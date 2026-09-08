"""Phase A6 — global questionnaire = SELECT + blend (no new generation).

# change for b2c questionarie

Pools published regional survey files, dedups near-exact questions, scores by
coverage / layer / grounding / hygiene, picks QUESTIONS_PER_TAB_TARGET per tab,
and blends answer percentages with population-style region weights.

Does NOT call the LangGraph pipeline or invent questions.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from src.question_plan import (
    CORE_PER_TAB,
    MODULE_PER_TAB,
    PUBLISH_REGION_SLUGS,
    QUESTIONS_PER_TAB_TARGET,
    REGION_BLEND_WEIGHTS,
    REGION_QUESTION_MODE,
)
from src import narrative
from src.question_similarity import (
    LOOSE_JACCARD,
    is_near_exact,
    norm_alnum,
    token_set_jaccard,
)
from src.survey_io import (
    FRONTEND_SURVEYS_DIR,
    _insight_from_data,
    atomic_write_text,
    publish_to_frontend,
    slugify,
    write_index_json,
)

# Geographic publish slugs only (exclude global).
_GEO_SLUGS = [s for s in PUBLISH_REGION_SLUGS if s != "global"]

_SECTION_ORDER = ("profile", "behavior", "preferences", "satisfaction")
_SECTION_LABELS = {
    "profile": "Consumer Profile",
    "behavior": "Buying Behavior",
    "preferences": "Preferences & Expectations",
    "satisfaction": "Satisfaction & Future Intent",
}

_DOUBLE_BARREL = re.compile(
    r"(?i)\b(and|or)\b.+\b(and|or)\b|\band\s+(also|how|what|why)\b"
)
_LEADING = re.compile(
    r"(?i)^(don'?t you|isn'?t it|wouldn'?t you|surely|obviously)\b"
)


def _hygiene_ok(text: str, options: list) -> float:
    """0–1 hygiene score (mechanical)."""
    score = 1.0
    t = text or ""
    if _DOUBLE_BARREL.search(t):
        score -= 0.35
    if _LEADING.search(t):
        score -= 0.35
    if not options or len(options) < 2:
        score -= 0.4
    return max(0.0, score)


def _load_region_payload(folder: str, region_slug: str) -> dict | None:
    path = os.path.join(folder, f"{region_slug}.json")
    if not os.path.isfile(path):
        return None
    try:
        return json.loads(open(path, encoding="utf-8").read())
    except (json.JSONDecodeError, OSError):
        return None


def _iter_frontend_questions(payload: dict, region_slug: str) -> list[dict]:
    """Flatten frontend questions with section + region provenance."""
    out = []
    for sec in ((payload.get("frontend") or {}).get("sections") or []):
        sid = sec.get("id") or "profile"
        for q in sec.get("questions") or []:
            out.append({
                "section_id": sid,
                "region": region_slug,
                "id": q.get("id"),
                "question": q.get("question") or "",
                "options": list(q.get("options") or []),
                "chart": q.get("chart") or "vbar",
                "data": list(q.get("data") or []),
                "insight": q.get("insight") or "",
                # change for b2c questionarie — carry persona/% rationale into A6
                "distributionNote": q.get("distributionNote") or "",
                "multiSelect": bool(q.get("multiSelect")),
                "ordered": bool(q.get("ordered")),
                # change for b2c questionarie — A12: the response type and its
                # instruction must survive the global rebuild, or the reader
                # cannot tell a select-all from a single pick and a question
                # summing past 100% looks like an arithmetic error.
                "questionType": q.get("questionType") or "",
                "responseInstruction": q.get("responseInstruction") or "",
                "sumsTo100": q.get("sumsTo100", True),
                "question_layer": (
                    q.get("questionLayer") or q.get("question_layer") or "full"
                ),
                "is_grounded": bool(q.get("isGrounded") or q.get("is_grounded")),
                "funnel_position": q.get("funnelPosition") or q.get("funnel_position"),
                # change for b2c questionarie — Phase A8: storyline provenance.
                # Each region planned its own arc, so beat_canonical is the only
                # reliable way to line the same narrative moment up across five
                # independently-planned regions.
                "beat_id": q.get("beatId") or q.get("beat_id") or "",
                "beat_canonical": q.get("beatCanonical") or q.get("beat_canonical"),
                "beat_title": q.get("beatTitle") or q.get("beat_title") or "",
                "narrative_order": (
                    q.get("narrativeOrder")
                    or q.get("narrative_order")
                    or q.get("funnelPosition")
                    or 999
                ),
            })
    return out


def _cluster_near_exact(
    items: list[dict], segment: str = ""
) -> list[dict]:
    """Group near-exact questions; return cluster dicts with regional instances."""
    clusters: list[dict] = []
    for item in items:
        placed = False
        for cl in clusters:
            if cl["section_id"] != item["section_id"]:
                continue
            if is_near_exact(
                cl["question"], item["question"],
                cl["options"], item["options"],
                segment=segment,
            ):
                cl["instances"].append(item)
                cl["regions"].add(item["region"])
                if item.get("is_grounded"):
                    cl["is_grounded"] = True
                # change for b2c questionarie — keep first non-empty distributionNote
                if item.get("distributionNote") and not cl.get("distributionNote"):
                    cl["distributionNote"] = item["distributionNote"]
                layer = item.get("question_layer") or "full"
                if layer == "core" or (
                    layer == "module" and cl["question_layer"] == "full"
                ):
                    cl["question_layer"] = layer
                # change for b2c questionarie — Phase A8: keep the first canonical
                # anchor seen, and the earliest story position across regions.
                if item.get("beat_canonical") and not cl.get("beat_canonical"):
                    cl["beat_canonical"] = item["beat_canonical"]
                if item.get("beat_title") and not cl.get("beat_title"):
                    cl["beat_title"] = item["beat_title"]
                try:
                    cl["narrative_order"] = min(
                        int(cl.get("narrative_order") or 999),
                        int(item.get("narrative_order") or 999),
                    )
                except (TypeError, ValueError):
                    pass
                placed = True
                break
        if not placed:
            clusters.append({
                "section_id": item["section_id"],
                "question": item["question"],
                "options": list(item["options"]),
                "chart": item.get("chart") or "vbar",
                "multiSelect": item.get("multiSelect"),
                "ordered": item.get("ordered"),
                # change for b2c questionarie — A12
                "questionType": item.get("questionType") or "",
                "responseInstruction": item.get("responseInstruction") or "",
                "sumsTo100": item.get("sumsTo100", True),
                "question_layer": item.get("question_layer") or "full",
                "is_grounded": bool(item.get("is_grounded")),
                "funnel_position": item.get("funnel_position"),
                # change for b2c questionarie
                "distributionNote": item.get("distributionNote") or "",
                "beat_id": item.get("beat_id") or "",
                "beat_canonical": item.get("beat_canonical"),
                "beat_title": item.get("beat_title") or "",
                "narrative_order": item.get("narrative_order") or 999,
                "regions": {item["region"]},
                "instances": [item],
            })
    return clusters


def _score_cluster(cl: dict, n_regions: int) -> float:
    n_regions = max(1, n_regions)
    coverage = len(cl["regions"]) / n_regions
    layer = cl.get("question_layer") or "full"
    layer_bonus = {"core": 0.35, "module": 0.12, "full": 0.18}.get(layer, 0.15)
    grounded = 0.20 if cl.get("is_grounded") else 0.0
    hygiene = 0.15 * _hygiene_ok(cl["question"], cl["options"])
    return coverage * 0.30 + layer_bonus + grounded + hygiene


def _blend_data(instances: list[dict]) -> list[dict]:
    """Population-weighted mean of regional {label,value} rows; renormalize."""
    if not instances:
        return []
    present = []
    for inst in instances:
        data = inst.get("data") or []
        if not data:
            continue
        w = REGION_BLEND_WEIGHTS.get(inst["region"], 1.0 / max(1, len(instances)))
        present.append((inst["region"], w, data))
    if not present:
        return list(instances[0].get("data") or [])

    best = max(present, key=lambda t: len(t[2]))
    labels = [d.get("label") for d in best[2]]
    label_set = {norm_alnum(lb) for lb in labels if lb}
    for _, _, data in present:
        for d in data:
            lb = d.get("label")
            if lb and norm_alnum(lb) not in label_set:
                labels.append(lb)
                label_set.add(norm_alnum(lb))

    blended = []
    for lb in labels:
        key = norm_alnum(lb)
        acc = 0.0
        w_used = 0.0
        for _, w, data in present:
            match = next(
                (d for d in data if norm_alnum(d.get("label") or "") == key),
                None,
            )
            if match is None:
                continue
            acc += float(match.get("value") or 0) * w
            w_used += w
        if w_used <= 0:
            continue
        blended.append({"label": lb, "value": round(acc / w_used, 1)})

    s = sum(d["value"] for d in blended)
    if blended and 90.0 <= s <= 110.0 and s != 100.0:
        factor = 100.0 / s
        for d in blended:
            d["value"] = round(d["value"] * factor, 1)
        drift = round(100.0 - sum(d["value"] for d in blended), 1)
        if drift and blended:
            top = max(blended, key=lambda d: d["value"])
            top["value"] = round(top["value"] + drift, 1)
    return blended


def _note_from_cluster(cl: dict) -> str:
    """First non-empty distributionNote from cluster or its instances."""
    # change for b2c questionarie
    note = (cl.get("distributionNote") or "").strip()
    if note:
        return note
    for inst in cl.get("instances") or []:
        note = (inst.get("distributionNote") or "").strip()
        if note:
            return note
    return ""


def _beat_key(cl: dict) -> str:
    """Identity of the storyline moment a cluster occupies."""
    key = (cl.get("beat_canonical") or cl.get("beat_id") or "").strip()
    return key or f"~{norm_alnum(cl.get('question') or '')[:24]}"


def _pick_clusters(
    clusters: list[dict],
    *,
    n_regions: int,
    target: int,
    prefer_core: bool,
    taken: list[dict] | None = None,
    segment: str = "",
) -> list[dict]:
    """Select up to ``target`` clusters for one section.

    # change for b2c questionarie — Phase A8
    Selection is beat-aware: early passes take at most ONE cluster per storyline
    beat, so the global questionnaire covers the whole arc instead of
    over-sampling a single moment. ``taken`` carries clusters already chosen for
    OTHER sections, so a question can never repeat anywhere in the global file.
    """
    taken = list(taken or [])
    scored = sorted(
        clusters,
        key=lambda c: (
            -_score_cluster(c, n_regions),
            int(c.get("narrative_order") or 999),
        ),
    )
    selected: list[dict] = []
    used_beats: set[str] = set()

    def _accept(cl: dict) -> bool:
        # Imported lazily: assembler owns the duplicate rules, and importing it
        # at module load would drag the whole assembly path into A6.
        from src.nodes.assembler import (
            _SAME_BEAT_OPTIONS,
            _SAME_BEAT_TEXT,
            _option_token_overlap,
        )

        for s in selected + taken:
            if is_near_exact(
                s["question"], cl["question"], s["options"], cl["options"],
                segment=segment,
            ):
                return False
            if token_set_jaccard(s["question"], cl["question"], segment) >= LOOSE_JACCARD:
                return False
            # change for b2c questionarie — the final fill pass ignores beat
            # boundaries, which let two restatements of one beat ("what starts
            # you thinking about buying" / "the main reason you started looking
            # for") both reach global.json. Same beat + shared option vocabulary
            # is a restatement, whatever the wording.
            beat = cl.get("beat_canonical") or cl.get("beat_id")
            s_beat = s.get("beat_canonical") or s.get("beat_id")
            if (
                beat
                and beat == s_beat
                and _option_token_overlap(cl["options"], s["options"], segment)
                >= _SAME_BEAT_OPTIONS
                and token_set_jaccard(s["question"], cl["question"], segment)
                >= _SAME_BEAT_TEXT
            ):
                return False
        return True

    def _take(pool: list[dict], limit: int, one_per_beat: bool) -> None:
        for cl in pool:
            if len(selected) >= limit:
                return
            key = _beat_key(cl)
            if one_per_beat and key in used_beats:
                continue
            if not _accept(cl):
                continue
            selected.append(cl)
            used_beats.add(key)

    if prefer_core:
        cores = [c for c in scored if c.get("question_layer") == "core"]
        modules = [c for c in scored if c.get("question_layer") != "core"]
        _take(cores, min(CORE_PER_TAB, target), True)
        _take(modules, min(CORE_PER_TAB + MODULE_PER_TAB, target), True)
    _take(scored, target, True)
    _take(scored, target, False)

    # Order by the canonical arc, so five independently-planned regional
    # storylines still publish as one coherent story.
    selected.sort(
        key=lambda c: (
            narrative.canonical_rank(c.get("section_id"), c.get("beat_canonical")),
            int(c.get("narrative_order") or 999),
            -_score_cluster(c, n_regions),
        )
    )
    return selected[:target]


def select_global_questionnaire(
    regional_payloads: dict[str, dict],
    *,
    segment: str = "",
    section_order: list[str] | None = None,
    section_labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a global frontend payload from regional files (pure selection)."""
    all_items: list[dict] = []
    for slug, payload in regional_payloads.items():
        if slug == "global" or not payload:
            continue
        all_items.extend(_iter_frontend_questions(payload, slug))

    if not all_items:
        raise ValueError("No regional questions available for global selection")

    n_regions = len({i["region"] for i in all_items})
    prefer_core = REGION_QUESTION_MODE == "core_plus_module"

    # change for b2c questionarie — section ids and their order now come from
    # the regional payloads themselves, because a market may name its own
    # sections. _SECTION_ORDER remains the fallback for older payloads.
    order = section_order or list(_SECTION_ORDER)
    by_section: dict[str, list] = {s: [] for s in order}
    for cl in _cluster_near_exact(all_items, segment=segment):
        by_section.setdefault(cl["section_id"], []).append(cl)

    sections_out = []
    qnum = 0
    # change for b2c questionarie — accumulates across sections so the global
    # questionnaire cannot repeat a question in a different tab.
    chosen_all: list[dict] = []
    for sid in order:
        clusters = by_section.get(sid) or []
        picked = _pick_clusters(
            clusters,
            n_regions=n_regions,
            target=QUESTIONS_PER_TAB_TARGET,
            prefer_core=prefer_core,
            taken=chosen_all,
            segment=segment,
        )
        chosen_all.extend(picked)
        qs = []
        for position, cl in enumerate(picked, start=1):
            qnum += 1
            data = _blend_data(cl["instances"])
            q_out = {
                "id": f"q{qnum}",
                "qNum": qnum,
                "question": cl["question"],
                "options": cl["options"] or [d["label"] for d in data],
                "chart": cl.get("chart") or "vbar",
                "data": data,
                "insight": _insight_from_data(data),
                # change for b2c questionarie — persona/% rationale (insight stays Top response)
                "distributionNote": _note_from_cluster(cl),
                "questionLayer": cl.get("question_layer") or "full",
                # change for b2c questionarie — Phase A8 storyline position
                "narrativeOrder": position,
                "funnelPosition": position,
                "beatId": cl.get("beat_id") or "",
                "beatTitle": cl.get("beat_title") or "",
                "beatCanonical": cl.get("beat_canonical"),
                # change for b2c questionarie — A12
                "questionType": cl.get("questionType") or "",
                "responseInstruction": cl.get("responseInstruction") or "",
                "sumsTo100": cl.get("sumsTo100", True),
                "sourceRegions": sorted(cl["regions"]),
            }
            if cl.get("multiSelect"):
                q_out["multiSelect"] = True
            if cl.get("ordered"):
                q_out["ordered"] = True
            if cl.get("is_grounded"):
                q_out["isGrounded"] = True
            qs.append(q_out)
        if qs:
            sections_out.append({
                "id": sid,
                "label": (section_labels or {}).get(sid)
                         or _SECTION_LABELS.get(sid, sid.replace("_", " ").replace("-", " ").title()),
                "questions": qs,
            })

    for sid, clusters in by_section.items():
        if sid in order:
            continue
        picked = _pick_clusters(
            clusters, n_regions=n_regions, target=QUESTIONS_PER_TAB_TARGET,
            prefer_core=prefer_core, taken=chosen_all, segment=segment,
        )
        chosen_all.extend(picked)
        qs = []
        for position, cl in enumerate(picked, start=1):
            qnum += 1
            data = _blend_data(cl["instances"])
            qs.append({
                "id": f"q{qnum}",
                "qNum": qnum,
                "question": cl["question"],
                "options": cl["options"] or [d["label"] for d in data],
                "chart": cl.get("chart") or "vbar",
                "data": data,
                "insight": _insight_from_data(data),
                # change for b2c questionarie
                "distributionNote": _note_from_cluster(cl),
                "questionLayer": cl.get("question_layer") or "full",
                # change for b2c questionarie — Phase A8 storyline position
                "narrativeOrder": position,
                "funnelPosition": position,
                "beatId": cl.get("beat_id") or "",
                "beatTitle": cl.get("beat_title") or "",
                "beatCanonical": cl.get("beat_canonical"),
                # change for b2c questionarie — A12
                "questionType": cl.get("questionType") or "",
                "responseInstruction": cl.get("responseInstruction") or "",
                "sumsTo100": cl.get("sumsTo100", True),
                "sourceRegions": sorted(cl["regions"]),
            })
        if qs:
            sections_out.append({
                "id": sid,
                "label": sid.replace("-", " ").title(),
                "questions": qs,
            })

    return {
        "frontend": {"sections": sections_out},
        "selection": {
            "mode": REGION_QUESTION_MODE,
            "source_regions": sorted({i["region"] for i in all_items}),
            "total_questions": qnum,
            "method": "select_and_blend",
        },
    }


def load_regional_payloads(slug: str, surveys_dir: str | None = None) -> dict[str, dict]:
    """Load geographic region JSON files under surveys/<slug>/.

    # change for b2c questionarie — when RUN_REGIONS is set, only blend those
    (avoids stale continents from earlier runs leaking into global.json).
    """
    from src.question_plan import RUN_GEOGRAPHIC_REGIONS, region_to_slug

    root = surveys_dir or FRONTEND_SURVEYS_DIR
    folder = os.path.join(root, slug)
    wanted = {region_to_slug(r) for r in RUN_GEOGRAPHIC_REGIONS} if RUN_GEOGRAPHIC_REGIONS else set(_GEO_SLUGS)
    out: dict[str, dict] = {}
    for geo in _GEO_SLUGS:
        if geo not in wanted:
            continue
        payload = _load_region_payload(folder, geo)
        if payload:
            out[geo] = payload
    # Fallback: if RUN_REGIONS filter yields nothing, load whatever is on disk.
    if not out:
        for geo in _GEO_SLUGS:
            payload = _load_region_payload(folder, geo)
            if payload:
                out[geo] = payload
    return out


def publish_global_selection(
    segment: str,
    *,
    slug: str | None = None,
    surveys_dir: str | None = None,
) -> dict[str, Any]:
    """Fan-in: select+blend → write global.json + finalize index.json."""
    slug = slug or slugify(segment)
    root = surveys_dir if surveys_dir is not None else FRONTEND_SURVEYS_DIR
    folder = os.path.join(root, slug)

    regional = load_regional_payloads(slug, surveys_dir=root)
    if not regional:
        raise FileNotFoundError(
            f"No regional survey files under {folder} — run region jobs first"
        )

    # change for b2c questionarie — a market names its own sections, so take
    # the order and labels from the regional payloads rather than a fixed tuple.
    section_order: list[str] = []
    section_labels: dict[str, str] = {}
    for _slug in _GEO_SLUGS:
        payload_r = regional.get(_slug) or {}
        for sec in ((payload_r.get("frontend") or {}).get("sections") or []):
            sid = sec.get("id")
            if sid and sid not in section_order:
                section_order.append(sid)
                if sec.get("label"):
                    section_labels[sid] = sec["label"]
        if section_order:
            break

    selected = select_global_questionnaire(
        regional, segment=segment,
        section_order=section_order or None,
        section_labels=section_labels or None,
    )
    # change for b2c questionarie — Phase A8: carry the storyline into global.json.
    # Regions plan independently, so publish the arc of the first region that has
    # one; the canonical anchors make it representative of the blended order.
    storyline: dict = {}
    storyline_source = None
    storyline_summary = ""
    definition: dict = {}
    # change for b2c questionarie — A11: the study-level blocks (purpose, the
    # segments, the scoring model that assigns them, and the fielding notes)
    # were being dropped here. Global rebuilds metadata from scratch, so
    # anything not named survived only in the regional files — leaving the
    # headline deliverable with 30 percentages and no way to score them.
    study: dict = {}
    _STUDY_KEYS = (
        "studyPurpose", "segmentation_lens", "must_cover_topics", "segments",
        "segmentModel", "fieldingNotes",
    )
    for _slug in _GEO_SLUGS:
        payload_r = regional.get(_slug) or {}
        meta = payload_r.get("metadata") or {}
        if meta.get("storyline") and not storyline:
            storyline = meta["storyline"]
            storyline_source = meta.get("storyline_source")
            storyline_summary = meta.get("storyline_summary") or ""
        for key in _STUDY_KEYS:
            if meta.get(key) and not study.get(key):
                study[key] = meta[key]
        if payload_r.get("surveyDefinition") and not definition:
            definition = dict(payload_r["surveyDefinition"])
        if storyline and definition and len(study) == len(_STUDY_KEYS):
            break

    # change for b2c questionarie — A11: global selects a SUBSET of the regional
    # questions, so a weight can point at a question that did not survive
    # selection. Drop those, or the scoring table would reference questions the
    # reader cannot find in the document.
    if study.get("segmentModel"):
        kept_ids = {
            q.get("id")
            for sec in (selected["frontend"].get("sections") or [])
            for q in (sec.get("questions") or [])
        }
        model = dict(study["segmentModel"])
        weights = [
            w for w in (model.get("weights") or [])
            if w.get("question_id") in kept_ids
        ]
        dropped = len(model.get("weights") or []) - len(weights)
        model["weights"] = weights
        if dropped:
            model["notes"] = [
                *(model.get("notes") or []),
                f"{dropped} scoring weight(s) dropped: they scored questions "
                "that were not selected into the global questionnaire.",
            ]
        study["segmentModel"] = model

    # change for b2c questionarie — restate the definition for the global view.
    # The block is copied from the first regional file, so every sentence naming
    # that region has to be rewritten or the Global page claims to describe
    # "buyers in North America".
    if definition:
        source_region = definition.get("region") or ""
        definition["region"] = "Global"
        definition["title"] = f"{segment} — consumer survey (Global)"

        def _globalise(text: str) -> str:
            out = str(text or "")
            if source_region and source_region != "Global":
                out = out.replace(f"in {source_region}", "worldwide")
                out = out.replace(f"Adults in {source_region}", "Adults worldwide")
                out = out.replace(source_region, "all surveyed regions")
            return out

        for key in ("whatThisSurveyIs", "whoseContextThisIs", "whyTheseRespondents"):
            if definition.get(key):
                definition[key] = _globalise(definition[key])

        definition["whatThisSurveyIs"] = (
            (definition.get("whatThisSurveyIs") or "")
            + " This global view is not a separate fieldwork exercise: it selects the "
            "strongest questions from the regional surveys and blends their "
            "percentages using population-style regional weights."
        ).strip()

    payload = {
        "segment": segment,
        "slug": slug,
        "region": "global",
        **({"surveyDefinition": definition} if definition else {}),
        "frontend": selected["frontend"],
        "metadata": {
            "total_questions": selected["selection"]["total_questions"],
            "selection": selected["selection"],
            "storyline": storyline,
            "storyline_source": storyline_source,
            "storyline_summary": storyline_summary,
            # change for b2c questionarie — A11: the segmentation study travels
            # with the global file. The scoring model is question-id keyed and
            # global keeps the regional ids, so the weights still resolve.
            **study,
            "warnings": [],
        },
    }

    os.makedirs(folder, exist_ok=True)
    global_path = os.path.join(folder, "global.json")
    atomic_write_text(global_path, json.dumps(payload, indent=2, ensure_ascii=False))

    ready = ["global"] + [s for s in _GEO_SLUGS if s in regional]
    order = {s: i for i, s in enumerate(PUBLISH_REGION_SLUGS)}
    ready = sorted(set(ready), key=lambda r: order.get(r, 999))
    index_path = write_index_json(
        folder, segment=segment, slug=slug, regions=ready,
    )

    legacy = None
    if surveys_dir is None or os.path.abspath(root) == os.path.abspath(FRONTEND_SURVEYS_DIR):
        legacy = publish_to_frontend(slug, json.dumps({
            "frontend": payload["frontend"],
            "survey": {
                "market_segment": segment,
                "region": "Global",
                "metadata": payload["metadata"],
            },
        }, indent=2, ensure_ascii=False))

    return {
        "segment": segment,
        "slug": slug,
        "path": global_path,
        "index": index_path,
        "legacy": legacy,
        "regions_used": sorted(regional.keys()),
        "questions": payload["metadata"]["total_questions"],
        "status": "ok",
    }
