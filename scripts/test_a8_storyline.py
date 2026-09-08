"""Phase A8 smoke — narrative ordering + absolute uniqueness.

# change for b2c questionarie

Guards the two rules that must never regress:

  1. STORY ORDER — a question is placed by the storyline beat it fills, not by
     the order the model happened to emit it in. The opening question of a
     section is always an opening question.
  2. ABSOLUTE UNIQUENESS — no question repeats anywhere in the survey, across
     sections, even reworded. Quota never protects a duplicate.

Runs with no LLM and no network. A dummy connection string is set only so
``src.config`` import-time validation passes; nothing connects.

Usage:
  python scripts/test_a8_storyline.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("NEON_CONNECTION_STRING", "postgresql://u:p@localhost/db")
os.environ.setdefault("DEEPSEEK_API_KEY", "sk-dummy-not-used")

from src import narrative  # noqa: E402
from src.graph import build_region_graph  # noqa: E402
from src.models import AnsweredQuestion, SurveyQuestion  # noqa: E402
from src.nodes.assembler import (  # noqa: E402
    _enforce_absolute_uniqueness,
    build_survey_json,
)
from src.nodes.question_architect import (  # noqa: E402
    PROMPT_TEMPLATE,
    _format_beats_block,
    _nearest_beat_id,
)

SEG = "Organic Milk"
_FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        _FAILURES.append(name)


def test_allocation() -> None:
    print("\n[1] beats → question allocation")
    beats = narrative.canonical_beats("buying_behavior")
    alloc = narrative.allocate_questions(beats, 8)
    check("all questions allocated", sum(n for _, n in alloc) == 8, str(alloc))
    check("earlier beats win the remainder", alloc[0][1] >= alloc[-1][1])
    short = narrative.allocate_questions(beats, 3)
    check(
        "a short tab still opens at beat 1",
        [b["beat_id"] for b, _ in short] == [b["beat_id"] for b in beats[:3]],
    )
    chunks = narrative.chunk_allocations(alloc, 8)
    check("chunks respect the output cap", all(sum(n for _, n in c) <= 8 for c in chunks))
    check(
        "chunking preserves story order",
        [b["beat_id"] for c in chunks for b, _ in c] == [b["beat_id"] for b in beats],
    )


def test_ordering() -> None:
    print("\n[2] scrambled model output → story order")
    beats = narrative.canonical_beats("consumer_profile")
    ids = [b["beat_id"] for b in beats]
    scrambled = [
        {"text": "Q-need", "beat_id": ids[4], "funnel_position": 1},
        {"text": "Q-freq", "beat_id": ids[2], "funnel_position": 1},
        {"text": "Q-entry", "beat_id": ids[0], "funnel_position": 1},
        {"text": "Q-occ-b", "beat_id": ids[1], "funnel_position": 2},
        {"text": "Q-occ-a", "beat_id": ids[1], "funnel_position": 1},
    ]
    out = narrative.order_questions(scrambled, beats)
    check(
        "questions run in beat order, then within-beat hint",
        [q["text"] for q in out] == ["Q-entry", "Q-occ-a", "Q-occ-b", "Q-freq", "Q-need"],
        str([q["text"] for q in out]),
    )
    check("narrative_order is dense 1..N", [q["narrative_order"] for q in out] == [1, 2, 3, 4, 5])
    check("beat_title backfilled", all(q["beat_title"] for q in out))

    mixed = [
        {"text": "Q-bogus", "beat_id": "no_such_beat", "funnel_position": 1},
        {"text": "Q-entry", "beat_id": ids[0], "funnel_position": 1},
    ]
    check(
        "unknown beat sinks to the end instead of crashing",
        [q["text"] for q in narrative.order_questions(mixed, beats)] == ["Q-entry", "Q-bogus"],
    )


def test_cross_region_remap() -> None:
    print("\n[3] shared CORE remapped into a region's own storyline")
    region_beats = [
        {"beat_id": "who_drinks_it", "title": "Who drinks it",
         "canonical": "household_role", "order": 1},
        {"beat_id": "morning_ritual", "title": "Morning ritual",
         "canonical": "usage_occasion", "order": 2},
        {"beat_id": "weekly_volume", "title": "Weekly volume",
         "canonical": "usage_frequency", "order": 3},
    ]
    core = [
        {"text": "core-freq", "beat_id": "usage_frequency", "beat_canonical": "usage_frequency"},
        {"text": "core-occasion", "beat_id": "usage_occasion", "beat_canonical": "usage_occasion"},
    ]
    narrative.remap_questions_to_beats(core, region_beats, "consumer_profile")
    check(
        "core mapped onto local beats via the canonical anchor",
        [q["beat_id"] for q in core] == ["weekly_volume", "morning_ritual"],
        str([q["beat_id"] for q in core]),
    )
    merged = narrative.order_questions(
        core + [{"text": "module-who", "beat_id": "who_drinks_it"}], region_beats,
    )
    check(
        "regional module lands in its beat, not appended last",
        [q["text"] for q in merged] == ["module-who", "core-occasion", "core-freq"],
        str([q["text"] for q in merged]),
    )


def test_blueprint_normalization() -> None:
    print("\n[4] planner output normalization")
    bp = narrative.normalize_blueprint(
        {"storyline": [
            {"tab": "consumer_profile", "beats": [
                {"beat_id": "Only One!", "title": "Only one", "canonical": "category_entry"}]},
            {"tab": "bogus_tab", "beats": [{"beat_id": "a", "title": "A"}]},
        ]},
        segment=SEG, region="Europe",
    )
    cp = narrative.beats_for_tab(bp, "consumer_profile")
    check("thin section topped up from the canonical arc", len(cp) >= narrative.MIN_BEATS_PER_TAB)
    check("beat ids slugified", cp[0]["beat_id"] == "only_one", cp[0]["beat_id"])
    check("all four sections present", all(narrative.beats_for_tab(bp, t) for t in narrative.TAB_ORDER))
    check("unknown section dropped", "bogus_tab" not in bp["storyline"])
    bad = narrative.normalize_blueprint(
        {"storyline": [{"tab": "buying_behavior",
                        "beats": [{"beat_id": "x", "title": "X", "canonical": "invented"}]}]},
    )
    check("invalid canonical nulled",
          narrative.beats_for_tab(bad, "buying_behavior")[0]["canonical"] is None)


def test_uniqueness() -> None:
    print("\n[5] absolute uniqueness")
    freq = ["Weekly", "Monthly", "Rarely", "Never"]
    answered = [
        {"id": "cp1", "tab": "consumer_profile", "narrative_order": 1, "type": "single_choice",
         "text": "How often do you buy organic milk?", "options": freq},
        {"id": "cp2", "tab": "consumer_profile", "narrative_order": 2, "type": "single_choice",
         "text": "Which meals do you drink organic milk with?",
         "options": ["Breakfast", "Lunch", "Dinner", "Other"]},
        {"id": "bb1", "tab": "buying_behavior", "narrative_order": 1, "type": "single_choice",
         "text": "How often do you buy organic milk?", "options": freq},
        {"id": "bb2", "tab": "buying_behavior", "narrative_order": 2, "type": "single_choice",
         "text": "How frequently do you purchase organic milk?", "options": freq},
        {"id": "bb3", "tab": "buying_behavior", "narrative_order": 3, "type": "single_choice",
         "text": "Which retail channel do you use to shop for organic milk?",
         "options": ["Supermarket", "Online grocery", "Local dairy", "Other"]},
    ]
    kept, notes = _enforce_absolute_uniqueness(answered, SEG)
    ids = [q["id"] for q in kept]
    check("exact duplicate in another section removed", "bb1" not in ids, str(ids))
    check("reworded duplicate removed", "bb2" not in ids, str(ids))
    check("distinct questions survive", {"cp1", "cp2", "bb3"} <= set(ids), str(ids))
    check("the first occurrence is the one kept", "cp1" in ids)
    check("every removal is reported", len(notes) == 2, str(notes))

    # False-positive guard: a shared scale is not a repeated question.
    likert = ["Very dissatisfied", "Somewhat dissatisfied", "Neither",
              "Somewhat satisfied", "Very satisfied"]
    scales = [
        {"id": f"sf{i}", "tab": "satisfaction_future_intent", "narrative_order": i,
         "type": "likert_5", "text": f"How satisfied are you with the {attr} of organic milk?",
         "options": likert}
        for i, attr in enumerate(("taste", "price", "shelf life"), start=1)
    ]
    kept2, _ = _enforce_absolute_uniqueness(scales, SEG)
    check("different attributes on one shared scale all survive",
          len(kept2) == 3, str([q["id"] for q in kept2]))


def test_assembler_end_to_end() -> None:
    print("\n[6] assembler publishes in story order")
    bp = narrative.default_blueprint(segment=SEG, region="Europe")
    beats = narrative.beats_for_tab(bp, "consumer_profile")

    def q(qid, beat, order, text):
        labels = ["A", "B", "C", "Other"]
        return {
            "id": qid, "tab": "consumer_profile", "text": text, "type": "single_choice",
            "chart_type": "bar", "options": labels, "beat_id": beat,
            "beat_canonical": beat, "narrative_order": order, "funnel_position": order,
            "question_layer": "core", "evidence_aligned": False, "is_grounded": False,
            "distribution_note": "",
            "answers": [{"label": lb, "percentage": p, "grounded_in": None,
                         "source_market": None, "grounding_label": "", "confidence": "medium"}
                        for lb, p in zip(labels, [40.0, 30.0, 20.0, 10.0])],
        }

    state = {
        "normalized_segment": SEG, "segment_type": "b2c", "region": "Europe",
        "regions_to_simulate": ["Europe"], "survey_blueprint": bp,
        "audience_profile": {}, "critique": {}, "counts_by_tab": {},
        # deliberately handed to the assembler in the WRONG order
        "answered_questions": [
            q("cp9", beats[4]["beat_id"], 5, "Late need-state question"),
            q("cp3", beats[2]["beat_id"], 3, "Middle frequency question"),
            q("cp1", beats[0]["beat_id"], 1, "Opening category question"),
        ],
    }
    survey = build_survey_json(state)
    profile = next(s for s in survey["frontend"]["sections"] if s["id"] == "profile")
    texts = [x["question"] for x in profile["questions"]]
    check("published order follows the storyline",
          texts == ["Opening category question", "Middle frequency question",
                    "Late need-state question"], str(texts))
    check("qNum ascends with the story", [x["qNum"] for x in profile["questions"]] == [1, 2, 3])
    check("narrativeOrder published", [x["narrativeOrder"] for x in profile["questions"]] == [1, 2, 3])
    check("beatId published", all(x.get("beatId") for x in profile["questions"]))
    meta = survey["survey"]["metadata"]
    check("storyline survives into published metadata", "profile" in (meta.get("storyline") or {}))


def test_wiring_and_models() -> None:
    print("\n[7] graph wiring, prompt, and model round-trip")
    graph = build_region_graph().get_graph()
    edges = {(e.source, e.target) for e in graph.edges}
    check("story_planner sits before the architect",
          ("persona_generator", "story_planner") in edges
          and ("story_planner", "question_architect") in edges)

    beats = narrative.canonical_beats("buying_behavior")
    block = _format_beats_block(narrative.allocate_questions(beats, 8),
                               free_choice=False, total=8)
    try:
        from src.nodes.question_architect import (
            _must_cover_block,
            _segments_block,
        )

        demo_bp = {
            "study_purpose": "Identify which of 4 behavioural segments each respondent is in.",
            "must_cover_topics": ["account sharing", "monthly spend"],
            "segments": [
                {"segment_id": "heavy", "name": "Heavy Buyers", "description": "Buy often."},
                {"segment_id": "value", "name": "Value Seekers", "description": "Price-led."},
            ],
        }
        rendered = PROMPT_TEMPLATE.format(
            tab="buying_behavior", segment=SEG, region="Europe", today="2026-08-03",
            regions="Europe", guidance="how they buy", beats_block=block,
            study_purpose=narrative.study_purpose(demo_bp),
            segments_block=_segments_block(demo_bp),
            must_cover_block=_must_cover_block(demo_bp),
            figures_json="[]", voice_json="[]", persona_json="[]",
            count=8, mode_line="", feedback_line="",
        )
        ok = True
    except (KeyError, IndexError, ValueError) as exc:
        ok, rendered = False, str(exc)
    check("architect prompt formats", ok, str(rendered)[:160])
    if ok:
        check("beat ids reach the prompt", "purchase_trigger" in rendered)
        check("per-beat quota stated", "write EXACTLY" in rendered)
        check("uniqueness rule stated", "ABSOLUTE UNIQUENESS" in rendered)
        # change for b2c questionarie — Phase A9 consultancy format
        check("study purpose reaches the prompt", "STUDY PURPOSE" in rendered)
        check("segments reach the prompt", "Value Seekers" in rendered)
        # change for b2c questionarie — A12: category coverage
        check("must-cover topics reach the prompt",
              "CANNOT CREDIBLY OMIT" in rendered
              and "account sharing" in rendered)
        check("option-relevance rule stated",
              "THE OPTIONS MUST ANSWER THE STEM" in rendered)
        check("stem window stated", "TARGET: 8-16 words" in rendered)
        # change for b2c questionarie — A11: too terse is a failure too
        check("ambiguity failure shown", "so short it is ambiguous" in rendered)
        check("self-contained rule stated", "SELF-CONTAINED" in rendered)
        # change for b2c questionarie — A10 simple/sequential questions
        check("simplicity rules stated", "SIMPLICITY RULES" in rendered)
        check("standard shape rule stated",
              "A STANDARD, PREDICTABLE SHAPE" in rendered)

    # change for b2c questionarie — A10: screening is a fielding concern, not
    # part of the instrument the client reads.
    _secs = narrative.sections_for(
        narrative.default_blueprint(SEG, "Europe"), include_standard=True)
    check("no screening section in the instrument",
          not any("screen" in (s.get("section_id") or "").lower() for s in _secs))
    check("profiling still closes the survey",
          "profil" in (_secs[-1].get("section_id") or "").lower())

    check("beat_id repair keeps an exact match", _nearest_beat_id("channel", beats) == "channel")
    check("beat_id repair handles an empty id",
          _nearest_beat_id("", beats) == beats[0]["beat_id"])

    sq = SurveyQuestion(id="cp1", tab="consumer_profile", text="t", type="single_choice",
                        options=["a", "b"], chart_type="bar", evidence_aligned=False,
                        beat_id="category_entry")
    aq = AnsweredQuestion(**sq.model_dump(), answers=[], sample_size=2500,
                          question_layer="core", narrative_order=3)
    check("question_layer survives simulation", aq.question_layer == "core")
    check("narrative_order survives simulation", aq.narrative_order == 3)
    check("defaults stay safe when absent",
          AnsweredQuestion(**sq.model_dump(), answers=[], sample_size=2500).question_layer == "full")


def test_adaptive_sections() -> None:
    print("\n[8] market-specific section names")
    bp = narrative.normalize_blueprint(
        {
            "sections": [
                {"section_id": "Streaming Habits", "label": "Streaming Habits",
                 "canonical": "consumer_profile", "remit": "how they watch"},
                {"section_id": "subscribe_manage", "label": "Subscribing & Managing",
                 "canonical": "buying_behavior", "remit": "sign up, switch, cancel"},
                {"section_id": "content_expectations", "label": "Content Expectations",
                 "canonical": "preferences_expectations", "remit": "what they want"},
                {"section_id": "value_churn", "label": "Value & Churn",
                 "canonical": "satisfaction_future_intent", "remit": "stay or go"},
                {"section_id": "dupe", "label": "Duplicate canonical",
                 "canonical": "consumer_profile"},
            ],
        },
        segment="Video Streaming", region="Europe",
    )
    ids = narrative.section_ids(bp)
    check("custom section ids kept", ids[0] == "streaming_habits", str(ids))
    check("duplicate canonical rejected", len(ids) == 4, str(ids))
    check("labels are category-specific",
          narrative.section_label(bp, "value_churn") == "Value & Churn")
    check("canonical anchor preserved",
          narrative.section_canonical(bp, "subscribe_manage") == "buying_behavior")
    check("renamed section inherits canonical beats",
          len(narrative.beats_for_tab(bp, "content_expectations")) >= 3)
    check("plan_for drives question targets",
          [t for t, _ in narrative.plan_for(bp, 8)] == ids)
    # Custom sections publish under their own id; canonical ones keep legacy ids.
    check("custom section publishes own id",
          narrative.frontend_section_id("streaming_habits") == "streaming_habits")
    check("canonical section keeps legacy id",
          narrative.frontend_section_id("consumer_profile") == "profile")
    check("default blueprint still yields the canonical four",
          narrative.section_ids(narrative.default_blueprint()) == list(narrative.TAB_ORDER))


def test_best_revision_kept() -> None:
    print("\n[9] assembler ships the best revision, not the last")
    bp = narrative.default_blueprint(segment=SEG, region="Europe")
    beats = narrative.beats_for_tab(bp, "consumer_profile")

    def q(qid, text):
        labels = ["A", "B", "C", "Other"]
        return {
            "id": qid, "tab": "consumer_profile", "text": text, "type": "single_choice",
            "chart_type": "bar", "options": labels, "beat_id": beats[0]["beat_id"],
            "beat_canonical": beats[0]["beat_id"], "narrative_order": 1,
            "funnel_position": 1, "question_layer": "core", "evidence_aligned": False,
            "is_grounded": False, "distribution_note": "",
            "answers": [{"label": lb, "percentage": p, "grounded_in": None,
                         "source_market": None, "grounding_label": "", "confidence": "medium"}
                        for lb, p in zip(labels, [40.0, 30.0, 20.0, 10.0])],
        }

    state = {
        "normalized_segment": SEG, "segment_type": "b2c", "region": "Europe",
        "regions_to_simulate": ["Europe"], "survey_blueprint": bp,
        "audience_profile": {}, "counts_by_tab": {},
        # final pass is WORSE than the best pass
        "critique": {"score": 0.031},
        "answered_questions": [q("cp1", "The degraded final-pass question")],
        "best_score": 0.742,
        "best_critique": {"score": 0.742},
        "best_answered": [q("cp1", "The good first-pass question")],
    }
    survey = build_survey_json(state)
    texts = [
        x["question"]
        for s in survey["frontend"]["sections"] for x in s["questions"]
    ]
    check("best pass restored", texts == ["The good first-pass question"], str(texts))
    warns = " ".join(survey["survey"]["metadata"]["warnings"])
    check("restoration is reported in warnings", "0.742" in warns and "0.031" in warns, warns)

    # And when the final pass is the best, nothing is swapped.
    state2 = dict(state, critique={"score": 0.9}, best_score=0.5,
                  best_critique={"score": 0.5})
    survey2 = build_survey_json(state2)
    texts2 = [
        x["question"]
        for s in survey2["frontend"]["sections"] for x in s["questions"]
    ]
    check("best-final pass is left alone",
          texts2 == ["The degraded final-pass question"], str(texts2))


def test_broken_grid_detector() -> None:
    print("\n[10] broken-grid detector")
    from src.nodes.validator_critic import _is_broken_grid_question as grid
    likert = ["Very dissatisfied", "Dissatisfied", "Neutral", "Satisfied", "Very satisfied"]
    attrs = ["Battery life", "Accuracy", "Comfort", "Design", "Integration"]
    stem = "How satisfied are you with each of the following aspects?"
    check("scale-only grid flagged", grid(stem, likert, "likert_5"))
    check("attribute grid asked as single pick flagged", grid(stem, attrs, "single_choice"))
    check("attribute grid as multi-select is fine", not grid(stem, attrs, "multiple_choice"))
    check("ordinary question not flagged",
          not grid("How often do you buy running shoes?",
                   ["Weekly", "Monthly", "Rarely"], "single_choice"))


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    test_allocation()
    test_ordering()
    test_cross_region_remap()
    test_blueprint_normalization()
    test_uniqueness()
    test_assembler_end_to_end()
    test_wiring_and_models()
    test_adaptive_sections()
    test_best_revision_kept()
    test_broken_grid_detector()
    print("\n" + "=" * 58)
    if _FAILURES:
        print(f"  A8 smoke FAILED ({len(_FAILURES)}): {_FAILURES}")
        sys.exit(1)
    print("  A8 smoke PASSED")
