"""Reusable survey-generation task entry points (CLI + RQ workers).

# change for b2c questionarie

Segment-only entry:
  1. classify_segment — one LLM call + B2C gate
  2. if rejected → write rejection artifact, enqueue NOTHING
  3. if accepted → fan out fixed 5 geographic region jobs (skip scope_classify)
     then A6 global fan-in

RQ prefer: ``classify_and_fanout`` as the first job; it dynamically enqueues
regional children + global depends_on.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Optional

from src import geography
from src.graph import build_classify_graph, build_region_graph
from src.nodes.assembler import to_json_string
from src.nodes.global_selector import publish_global_selection
from src.question_plan import (
    GEOGRAPHIC_REGIONS,
    RUN_GEOGRAPHIC_REGIONS,
    match_geographic_region,
    region_to_slug,
)
from src.survey_io import (
    build_region_payload,
    publish_region_file,
    publish_to_frontend,
    slugify,
    unique_output_path,
)

ProgressCallback = Optional[Callable[[str, dict], None]]

_STAGGER_CAP_SECONDS = float(os.getenv("STAGGER_CAP_SECONDS", "180"))
# change for b2c questionarie — cost/time optimization step 2. The 10
# region+country jobs in one segment run are fully independent (own state
# dict, own cache reads/writes); running them concurrently instead of one
# after another is the single biggest wall-clock win available without
# touching generation quality. Bounded so a single run cannot exceed the
# shared LLM rate limiter's realistic burst headroom.
_REGION_JOB_CONCURRENCY = int(os.getenv("REGION_JOB_CONCURRENCY", "4"))
_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def _maybe_stagger_start() -> None:
    """Desync simulator peaks across parallel RQ jobs (job.meta stagger_index)."""
    try:
        from rq import get_current_job
    except ImportError:
        return
    job = get_current_job()
    if job is None:
        return
    idx = (job.meta or {}).get("stagger_index")
    if idx is None:
        return
    step = float(os.getenv("STAGGER_SECONDS", "30"))
    delay = min(int(idx) * step, _STAGGER_CAP_SECONDS)
    if delay > 0:
        print(f"[stagger] job {job.id[:8]}… sleep {delay:.0f}s (index={idx})")
        time.sleep(delay)


def _classification_payload(final_state: dict) -> dict[str, Any]:
    """Normalize classification fields passed into regional jobs."""
    return {
        "normalized_segment": final_state.get("normalized_segment") or "",
        "segment_type": final_state.get("segment_type") or "b2c",
        "classification_confidence": final_state.get("classification_confidence") or "medium",
        "is_hybrid": bool(final_state.get("is_hybrid")),
        # change for b2c questionarie — A11: the planner and the methodology
        # note both need to know this is a dual-motion business.
        "sells_to_both": bool(final_state.get("sells_to_both")),
        "dual_motion_reason": final_state.get("dual_motion_reason") or "",
        "classification_reason": final_state.get("classification_reason") or "",
        "audience_profile": final_state.get("audience_profile") or {},
        "gate_decision": final_state.get("gate_decision") or "",
        "gate_passed": bool(final_state.get("gate_passed")),
    }


def _write_rejection(segment: str, payload: dict) -> str:
    """Persist rejected status so drops are explainable."""
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    slug = slugify(segment)
    path = _OUTPUT_DIR / f"{slug}_rejected.json"
    body = {
        "status": "rejected",
        "reason": payload.get("reason") or payload.get("rejection_reason") or "",
        "segment_type": payload.get("segment_type"),
        "classification_confidence": payload.get("classification_confidence"),
        "is_hybrid": payload.get("is_hybrid"),
        "gate_decision": payload.get("gate_decision"),
        "market_segment": segment,
        "normalized_segment": payload.get("normalized_segment"),
        "classification_reason": payload.get("classification_reason"),
    }
    path.write_text(json.dumps(body, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)


def _write_clarification(segment: str, payload: dict) -> str:
    """Persist the question we need answered, so a caller can re-run with it.

    # change for b2c questionarie — A11
    """
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    slug = slugify(segment)
    path = _OUTPUT_DIR / f"{slug}_needs_clarification.json"
    body = {
        "status": "needs_clarification",
        "clarifying_question": payload.get("reason") or "",
        "market_segment": segment,
        "normalized_segment": payload.get("normalized_segment"),
        "segment_type": payload.get("segment_type"),
        "classification_confidence": payload.get("classification_confidence"),
        "gate_decision": payload.get("gate_decision"),
        "classification_reason": payload.get("classification_reason"),
    }
    path.write_text(json.dumps(body, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)


async def classify_segment_async(
    segment: str,
    *,
    on_progress: ProgressCallback = None,
) -> dict:
    """Run scope_classify once. Returns accepted classification or rejected status."""
    # change for b2c questionarie
    graph = build_classify_graph()
    initial = {"market_segment": segment, "revision_count": 0}
    final_state: dict = {}
    async for event in graph.astream(initial, stream_mode="updates"):
        for node, update in event.items():
            if isinstance(update, dict):
                final_state.update(update)
                if on_progress:
                    on_progress(node, update)

    status = final_state.get("status") or (
        "accepted" if final_state.get("gate_passed") else "rejected"
    )

    # change for b2c questionarie — A11: unclassifiable input stops here and
    # asks. Kept ahead of the rejection branch so it cannot be recorded as a
    # refusal — "we don't know" and "we said no" are different answers.
    if status == "needs_clarification":
        question = (
            final_state.get("clarifying_question")
            or final_state.get("rejection_reason")
            or (final_state.get("final_survey") or {}).get("clarifying_question")
            or "What does this sell, and who pays for it?"
        )
        path = _write_clarification(segment, {**final_state, "reason": question})
        return {
            "status": "needs_clarification",
            "reason": question,
            "clarifying_question": question,
            "market_segment": segment,
            "normalized_segment": final_state.get("normalized_segment"),
            "gate_decision": final_state.get("gate_decision"),
            "classification_reason": final_state.get("classification_reason"),
            "path": path,
            "regions_enqueued": 0,
        }

    if status == "rejected" or not final_state.get("gate_passed"):
        reason = (
            final_state.get("rejection_reason")
            or (final_state.get("final_survey") or {}).get("reason")
            or final_state.get("classification_reason")
            or "Rejected by B2C gate."
        )
        path = _write_rejection(segment, {**final_state, "reason": reason})
        return {
            "status": "rejected",
            "reason": reason,
            "segment_type": final_state.get("segment_type"),
            "classification_confidence": final_state.get("classification_confidence"),
            "is_hybrid": final_state.get("is_hybrid"),
            "gate_decision": final_state.get("gate_decision"),
            "market_segment": segment,
            "normalized_segment": final_state.get("normalized_segment"),
            "classification_reason": final_state.get("classification_reason"),
            "path": path,
            "regions_enqueued": 0,
        }

    return {
        "status": "accepted",
        "classification": _classification_payload(final_state),
        "market_segment": segment,
        # change for b2c questionarie — honour RUN_REGIONS. This previously
        # always fanned out all 5 continents, so the env var only narrowed which
        # regions A6 blended and could not be used to limit an actual run.
        "regions_to_simulate": list(RUN_GEOGRAPHIC_REGIONS),
    }


def classify_segment(segment: str) -> dict:
    """Sync classify entry (RQ / tests)."""
    return asyncio.run(classify_segment_async(segment))


async def run_region_pipeline(
    segment: str,
    region: str,
    classification: dict,
    *,
    country: str | None = None,
    on_progress: ProgressCallback = None,
) -> dict:
    """Regional pipeline starting at evidence_harvester (no scope_classify)."""
    graph = build_region_graph()
    initial = {
        "market_segment": segment,
        "region": region,
        # change for b2c questionarie — the geography the survey is written FOR.
        # Nodes that phrase questions or estimate answers use this; evidence and
        # the shared core still come from the parent region, which is what keeps
        # the two countries in a region comparable.
        "country": country or "",
        "geography_label": country or region,
        "regions_to_simulate": [region],
        "revision_count": 0,
        "normalized_segment": classification.get("normalized_segment") or segment,
        "segment_type": classification.get("segment_type") or "b2c",
        "classification_confidence": classification.get("classification_confidence") or "high",
        "is_hybrid": bool(classification.get("is_hybrid")),
        # change for b2c questionarie — A11
        "sells_to_both": bool(classification.get("sells_to_both")),
        "dual_motion_reason": classification.get("dual_motion_reason") or "",
        "classification_reason": classification.get("classification_reason") or "",
        "audience_profile": classification.get("audience_profile") or {},
        "gate_passed": True,
        "gate_decision": classification.get("gate_decision") or "accepted_preseeded",
        "status": "accepted",
        "classification_preseeded": True,
    }

    final_state: dict = {}
    last_node = None
    async for event in graph.astream(initial, stream_mode="updates"):
        for node, update in event.items():
            last_node = node
            if isinstance(update, dict):
                final_state.update(update)
                if on_progress:
                    on_progress(node, update)
    final_state["_last_node"] = last_node
    return final_state


async def run_one_region_async(
    segment: str,
    region: str,
    classification: dict | None = None,
    *,
    country: str | None = None,
    on_progress: ProgressCallback = None,
) -> dict:
    """Run the pipeline for one geography and publish its file.

    # change for b2c questionarie — with ``country`` set, the survey is written
    # FOR that country and filed under its own slug. The parent region still
    # drives evidence and the shared core questions, so the two countries in a
    # region stay comparable; only the local module and the answer
    # distributions differ.
    """
    geo = match_geographic_region(region)
    if not geo:
        raise ValueError(
            f"run_one_region requires a geographic region, got {region!r}. "
            f"Expected one of {GEOGRAPHIC_REGIONS}"
        )
    if not classification:
        raise ValueError(
            "run_one_region requires classification from classify_segment "
            "(regional jobs must not re-run scope_classify)."
        )

    final_state = await run_region_pipeline(
        segment, geo, classification, country=country, on_progress=on_progress,
    )
    survey = final_state.get("final_survey")
    if not survey:
        last = final_state.get("_last_node")
        raise RuntimeError(f"No final survey produced (last node: {last})")
    if survey.get("status") == "rejected":
        raise RuntimeError("Region job received rejected classification — aborting.")

    slug = slugify(segment)
    # The file is named for the geography it actually covers.
    region_slug = geography.slug(country) if country else region_to_slug(geo)
    json_text = to_json_string(survey)
    path = unique_output_path(f"{slug}_{region_slug}")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json_text)

    payload = build_region_payload(
        survey, segment=segment, slug=slug, region_slug=region_slug,
    )
    published = publish_region_file(
        slug, region_slug, payload, segment=segment,
    )
    publish_to_frontend(slug, json_text)

    # change for b2c questionarie -- CHECK (B2C_MASTER_RULES_FINAL /
    # JSON_FORMAT_SPECIFICATION): the delivered `survey` dict no longer
    # carries a metadata block (total_questions/warnings) -- read those off
    # the new structure.totalQuestions and final_state directly instead.
    grounded_pct = final_state.get("grounded_pct", 0.0)
    n_core = sum(
        1 for q in (final_state.get("questions") or [])
        if q.get("question_layer") == "core"
    )
    n_mod = sum(
        1 for q in (final_state.get("questions") or [])
        if q.get("question_layer") == "module"
    )
    total_questions = (survey.get("structure") or {}).get("totalQuestions", 0)

    return {
        "segment": segment,
        "region": geo,
        "country": country or "",
        "region_slug": region_slug,
        "slug": slug,
        "path": str(path),
        "published": published,
        "questions": total_questions,
        "core_questions": n_core,
        "module_questions": n_mod,
        "grounded_pct": grounded_pct,
        "status": "ok",
    }


def run_one_region(
    segment: str,
    region: str,
    classification: dict | None = None,
    country: str | None = None,
) -> dict:
    """Sync RQ entry: one geography job (classification required)."""
    _maybe_stagger_start()
    return asyncio.run(
        run_one_region_async(segment, region, classification, country=country)
    )


def _enqueue_regional_jobs(segment: str, classification: dict) -> dict:
    """From classify worker: enqueue one job per country (top 2 per region)."""
    import redis
    from rq import Queue, Retry

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    queue_name = os.getenv("RQ_QUEUE_NAME", "surveys")

    def _parse_timeout(raw: str) -> int:
        import re
        raw = (raw or "60m").strip().lower()
        if raw.isdigit():
            return int(raw)
        m = re.fullmatch(r"(\d+(?:\.\d+)?)([smhd])", raw)
        if m:
            n, unit = float(m.group(1)), m.group(2)
            mult = {"s": 1, "m": 60, "h": 3600, "d": 86400}
            return int(n * mult[unit])
        return int(float(raw))

    job_timeout = _parse_timeout(os.getenv("RQ_JOB_TIMEOUT", "60m"))

    conn = redis.from_url(redis_url)
    q = Queue(queue_name, connection=conn)

    regional_ids = []
    # change for b2c questionarie — the queue path fans out to COUNTRIES, the
    # same as the CLI: the top 2 countries in each selected region, and no
    # Global blend. Keeping the queue on the old region+global shape would have
    # made the two entry points produce different deliverables.
    regions = list(RUN_GEOGRAPHIC_REGIONS)
    jobs = [(r, c) for r in regions for c in geography.countries_for(r, top_only=True)]
    for stagger, (r, c) in enumerate(jobs):
        job = q.enqueue(
            run_one_region,
            segment,
            r,
            classification,
            country=c,
            job_timeout=job_timeout,
            retry=Retry(max=2, interval=[30, 120]),
            meta={"stagger_index": stagger, "region": r, "country": c,
                  "phase": "country"},
        )
        regional_ids.append(job.id)
        print(f"enqueued {segment!r} / {c} ({r}) -> {job.id}")

    return {
        "regional_job_ids": regional_ids,
        "regions": regions,
        "countries": [c for _, c in jobs],
    }


def classify_and_fanout(segment: str) -> dict:
    """RQ first job: classify once; on accept enqueue 5 regions + global."""
    # change for b2c questionarie
    result = classify_segment(segment)
    if result.get("status") == "rejected":
        print(f"[b2c-gate] REJECTED {segment!r}: {result.get('reason')}")
        return result

    classification = result["classification"]
    print(
        f"[b2c-gate] ACCEPTED {segment!r} "
        f"({classification.get('segment_type')}/"
        f"{classification.get('classification_confidence')}) — fan-out 5 regions"
    )
    enq = _enqueue_regional_jobs(segment, classification)
    return {
        "status": "accepted",
        "market_segment": segment,
        "classification": classification,
        "regions_enqueued": len(enq["regions"]),
        **enq,
    }


async def run_one_async(
    segment: str,
    *,
    on_progress: ProgressCallback = None,
) -> dict:
    """CLI path: classify once → 5 region jobs → A6 global. Segment only."""
    print(f"\n── Classify (B2C gate) for {segment!r} ──")
    clf = await classify_segment_async(segment, on_progress=on_progress)
    # change for b2c questionarie — A11: neither non-accepted outcome may fan out.
    if clf.get("status") in ("rejected", "needs_clarification"):
        label = "NEEDS CLARIFICATION" if clf["status"] == "needs_clarification" else "REJECTED"
        print(f"[b2c-gate] {label}: {clf.get('reason')}")
        print(f"  Wrote: {clf.get('path')}")
        return clf

    classification = clf["classification"]
    # change for b2c questionarie — the deliverable is COUNTRY files: the top 2
    # countries in each region, and nothing else. Five regions therefore produce
    # ten files. The region itself is no longer published (it was an average of
    # markets nobody sells into) and neither is the Global blend.
    regions = list(RUN_GEOGRAPHIC_REGIONS)
    jobs: list[tuple[str, str]] = []
    for r in regions:
        for c in geography.countries_for(r, top_only=True):
            jobs.append((r, c))

    # change for b2c questionarie — cost/time optimization step 2: these jobs
    # have no cross-iteration data dependency (each builds its own state dict
    # and only touches the on-disk cache, which is safe for concurrent reads
    # and idempotent on concurrent writes — see cache_store.py). Run them
    # concurrently, bounded by a semaphore so this run cannot outpace the
    # shared LLM rate limiter. Results are collected back in the ORIGINAL
    # `jobs` order (not completion order) so `results[-1]` below keeps
    # selecting the same "last job" as the old sequential loop did.
    semaphore = asyncio.Semaphore(max(1, _REGION_JOB_CONCURRENCY))

    async def _run_job(i: int, r: str, c: str) -> dict:
        async with semaphore:
            print(f"\n── Job {i + 1}/{len(jobs)}: {c} ({r}) ──")
            return await run_one_region_async(
                segment, r, classification, country=c, on_progress=on_progress,
            )

    results = await asyncio.gather(
        *(_run_job(i, r, c) for i, (r, c) in enumerate(jobs))
    )

    return {
        "segment": segment,
        "slug": slugify(segment),
        "path": results[-1]["path"] if results else "",
        "published": results[-1].get("published") if results else None,
        "published_regions": [r["region_slug"] for r in results],
        "questions": results[-1].get("questions", 0) if results else 0,
        "grounded_pct": results[-1].get("grounded_pct", 0) if results else 0,
        "warnings": [],
        "region_results": results,
        "classification": classification,
        "status": "ok",
    }


def run_one_segment(segment: str) -> dict:
    """Sync entry for CLI/workers that run classify+fanout in-process."""
    _maybe_stagger_start()
    return asyncio.run(run_one_async(segment))


def run_global_selection(segment: str) -> dict:
    """Fan-in job — Phase A6: select best questions + blend regional %."""
    return publish_global_selection(segment)


def deliberately_fail_for_retry_test() -> None:
    """Always raises — used only by scripts/test_rq_retry.py (Phase 5)."""
    raise RuntimeError("intentional failure for RQ retry test")


def phase5_ok_marker() -> dict:
    """Trivial success job — proves other work continues while a sibling fails."""
    return {"status": "ok", "phase5": True}
