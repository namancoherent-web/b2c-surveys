"""CLI runner for the agentic survey generator.

Usage:
    python run.py "<market segment>"

Segment-only: classifies B2C once, hard-stops pure B2B, else fans out the fixed
5 continents + global blend (A6).
"""

import asyncio
import json
import sys

from src.tasks import run_one_async


def _progress_line(node: str, update: dict) -> str:
    """One short status line summarizing a node's contribution."""
    if node == "scope_classify":
        gate = "PASS" if update.get("gate_passed") else "STOP"
        return (
            f"✓ scope_classify     → {update.get('segment_type')} "
            f"({update.get('classification_confidence')}) [{gate}] "
            f"{update.get('gate_decision')}"
        )
    if node == "reject_stop":
        return f"✗ reject_stop       → {update.get('status')}: {update.get('rejection_reason')}"
    if node == "evidence_harvester":
        pool = update.get("evidence_pool") or []
        numeric = sum(1 for f in pool if f.get("is_numeric"))
        by_origin = {}
        for f in pool:
            by_origin[f["origin"]] = by_origin.get(f["origin"], 0) + 1
        origins = ", ".join(f"{k}:{v}" for k, v in sorted(by_origin.items()))
        return f"✓ evidence_harvester → {len(pool)} facts ({numeric} numeric) [{origins}]"
    if node == "evidence_synthesizer":
        mkp = update.get("market_knowledge_profile") or {}
        return (
            f"✓ evidence_synthesizer → confidence={mkp.get('evidence_confidence')}, "
            f"pains={len(mkp.get('top_pain_points') or [])}, "
            f"thin={update.get('grounding_thin')}"
        )
    if node == "persona_generator":
        cat = update.get("persona_catalog") or {}
        n = len(cat.get("personas") or [])
        regs = len(cat.get("weights_by_region") or {})
        return f"✓ persona_generator  → {n} archetypes, {regs} regional weight packs"
    if node == "story_planner":
        # change for b2c questionarie — Phase A8. Sections are named per market,
        # so read them off the blueprint rather than assuming the canonical four
        # (which reported [0/0/0/0] once section ids became market-specific).
        from src import narrative

        bp = update.get("survey_blueprint") or {}
        sections = narrative.sections_for(bp)
        counts = "/".join(
            str(len(narrative.beats_for_tab(bp, s["section_id"]))) for s in sections
        )
        names = " · ".join(s.get("label") or s["section_id"] for s in sections)
        return (
            f"✓ story_planner      → {bp.get('source')} arc, "
            f"{len(sections)} sections [{counts}] beats\n"
            f"                       {names}"
        )
    if node == "question_architect":
        counts = update.get("counts_by_tab") or {}
        total = sum(counts.values())
        per_tab = "/".join(str(v) for v in counts.values())
        cache = "cache-hit" if update.get("questionnaire_cache_hit") else "generated"
        return f"✓ question_architect → {total} questions [{per_tab}] ({cache})"
    if node == "survey_simulator":
        return (
            f"✓ survey_simulator   → {update.get('grounded_question_count', 0)} confident Qs, "
            f"avg_conf={update.get('simulation_confidence_avg', 0)}%"
        )
    if node == "validator_critic":
        c = update.get("critique") or {}
        verdict = "passed" if c.get("passed") else "needs revision"
        return (
            f"✓ validator_critic   → score {c.get('score')}, {verdict}, "
            f"grounded_pct {c.get('grounded_pct')}%"
        )
    if node == "assembler":
        return "✓ assembler          → final survey assembled"
    return f"✓ {node}"


def _cli_progress(node: str, update: dict) -> None:
    try:
        print(_progress_line(node, update))
    except Exception:  # noqa: BLE001
        print(f"[ok] {node}")


def _print_summary(survey: dict, path: str) -> None:
    # change for b2c questionarie -- CHECK (B2C_MASTER_RULES_FINAL /
    # JSON_FORMAT_SPECIFICATION): the delivered JSON is now
    # surveyScope/behaviouralPersonas/structure/segments -- no more
    # survey.metadata block to read the CLI summary off of. Derive the same
    # summary info directly from structure/segments instead.
    structure = survey.get("structure") or {}
    segments = survey.get("segments") or []
    print("\n" + "=" * 60)
    print(f"  Saved: {path}")
    print(f"  Total questions: {structure.get('totalQuestions')}")
    print(
        "  Counts by segment: "
        + str({s.get("label"): s.get("questions") for s in structure.get("sections") or []})
    )
    print(f"  Behavioural personas: {len(survey.get('behaviouralPersonas') or [])}")
    all_qs = [q for s in segments for q in s.get("questions", [])]
    if all_qs:
        multi = sum(1 for q in all_qs if q.get("type") == "multi_select")
        print(f"  Multi-select questions: {multi}/{len(all_qs)}")
    print("=" * 60)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    if len(sys.argv) > 2:
        print("Usage: python run.py \"<market segment>\"")
        print("Region is no longer accepted — fan-out is always the fixed 5 continents.")
        sys.exit(2)

    if len(sys.argv) > 1:
        segment = sys.argv[1]
    else:
        segment = input("Market segment: ").strip()

    if not segment:
        print("No market segment provided.")
        sys.exit(1)

    # change for b2c questionarie — reflect RUN_REGIONS instead of always "5"
    from src.question_plan import RUN_GEOGRAPHIC_REGIONS

    n_regions = len(RUN_GEOGRAPHIC_REGIONS)
    print(
        f"\nGenerating survey for: {segment!r}  "
        f"(classify → {n_regions} region{'s' if n_regions != 1 else ''} + global)\n"
    )
    try:
        result = asyncio.run(run_one_async(segment, on_progress=_cli_progress))
        # change for b2c questionarie — A11: three outcomes, not two. Both
        # non-accepted paths exit non-zero so a batch script can tell that no
        # survey was produced without parsing stdout.
        if result.get("status") == "needs_clarification":
            print("\n" + "=" * 60)
            print("  NEED MORE INFORMATION — nothing generated")
            print(f"  input:    {segment!r}")
            print(f"\n  {result.get('clarifying_question') or result.get('reason')}\n")
            print("  Re-run with a fuller description, e.g.:")
            print('    python run.py "<what it sells, and who pays for it>"')
            print(f"  artifact: {result.get('path')}")
            print("=" * 60)
            sys.exit(3)

        if result.get("status") == "rejected":
            print("\n" + "=" * 60)
            print("  OUT OF SCOPE — no survey generated, no region files written")
            print(f"  segment_type: {result.get('segment_type')}")
            print(f"\n  {result.get('reason')}\n")
            print(f"  artifact:     {result.get('path')}")
            print("=" * 60)
            sys.exit(4)

        path = result.get("path") or ""
        if path and path.endswith("global.json"):
            print("\n" + "=" * 60)
            print(f"  Global selection saved: {path}")
            print(f"  Questions: {result.get('questions')}")
            print(f"  Regions: {result.get('published_regions')}")
            print("=" * 60)
        elif path:
            with open(path, encoding="utf-8") as fh:
                survey = json.load(fh)
            _print_summary(survey, path)
        if result.get("published"):
            print(f"  Published to frontend: {result['published']}")
            print(f"  Preview: http://localhost:3000/survey/{result['slug']}/global")
            for rs in result.get("published_regions") or []:
                if rs != "global":
                    print(f"           http://localhost:3000/survey/{result['slug']}/{rs}")
    except Exception as exc:  # noqa: BLE001
        print(f"\nRun failed: {exc}")
        raise


if __name__ == "__main__":
    main()
