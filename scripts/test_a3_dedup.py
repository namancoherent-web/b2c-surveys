"""Phase A3 smoke — near-exact vs loose dedup + global similarity (no LLM)."""
from src.nodes.assembler import _drop_duplicates
from src.nodes.validator_critic import (
    _heuristic_duplicates,
    _near_exact_duplicates,
    validator_critic,
)
from src.question_plan import CATEGORY_PLAN, QUESTIONS_PER_TAB_MIN
from src.question_similarity import (
    is_near_exact,
    too_similar_to_any,
    token_set_jaccard,
)
from src.nodes.question_architect import _Q_SCHEMA_V


def _q(qid, tab, text, options=None):
    opts = options or ["Daily", "Weekly", "Monthly", "Rarely", "Never"]
    return {
        "id": qid,
        "tab": tab,
        "text": text,
        "type": "single_choice",
        "options": opts,
        "funnel_position": 1,
        "answers": [
            {"label": o, "percentage": round(100 / len(opts), 1),
             "grounded_in": None, "source_market": None,
             "grounding_label": "", "confidence": "medium"}
            for o in opts
        ],
        "sample_size": 2500,
        "distribution_note": "",
        "is_grounded": False,
        "question_confidence": "medium",
    }


def test_similarity():
    assert is_near_exact(
        "How often do you use an air fryer?",
        "How often do you use an air fryer?",
    )
    assert is_near_exact(
        "How often do you use your air fryer at home?",
        "How often do you use an air fryer at home?",
        segment="air fryer",
    )
    assert too_similar_to_any(
        "How often do you purchase organic milk at the grocery store?",
        ["How often do you purchase organic milk at the grocery store"],
        segment="organic milk",
    )
    assert too_similar_to_any(
        "How satisfied are you with the taste of organic milk?",
        ["How satisfied are you with taste of organic milk?"],
        segment="organic milk",
    )
    assert not is_near_exact(
        "How often do you buy air fryers?",
        "How satisfied are you with air fryer cleaning?",
        segment="air fryer",
    )
    assert _Q_SCHEMA_V.startswith("questionnaire_v")
    print("similarity OK", token_set_jaccard(
        "How often do you use an air fryer at home?",
        "How often do you use your air fryer at home?",
        "air fryer",
    ))


def test_near_exact_uncapped_and_fail():
    import src.nodes.validator_critic as vc
    vc._llm_judge = lambda *a, **k: vc._JudgeResult(duplicates=[], issues=[])

    # Build min-sized survey with one near-exact cross-tab pair
    full = []
    tabs = [t for t, _ in CATEGORY_PLAN]
    for tab in tabs:
        for i in range(QUESTIONS_PER_TAB_MIN):
            full.append(_q(
                f"{tab}_{i}", tab,
                f"In the last 30 days, how often did you use product path {tab} {i}?",
            ))
    # Plant near-exact paraphrase across tabs
    full[0]["text"] = "How often do you use an air fryer at home?"
    full[QUESTIONS_PER_TAB_MIN]["text"] = "How often do you use your air fryer at home?"

    hard = _near_exact_duplicates(full, segment="air fryer")
    assert any(len(c) >= 2 for c in hard), hard

    out = validator_critic({
        "answered_questions": full,
        "revision_count": 0,
        "segment": "air fryer",
        "normalized_segment": "air fryer",
        "evidence_pack": {"claims": []},
        "counts_by_tab": {t: QUESTIONS_PER_TAB_MIN for t, _ in CATEGORY_PLAN},
    })
    c = out["critique"]
    assert c["passed"] is False
    assert c.get("hard_duplicates")
    assert any("DEDUP" in f for f in c["feedback"])
    print("near-exact fail OK", c["hard_duplicates"][:1])


def test_assembler_hard_drop():
    answered = [
        _q("cp1", "consumer_profile", "How often do you use an air fryer?"),
        _q("cp2", "consumer_profile", "How often do you use an air fryer?"),  # exact dup
        _q("bb1", "buying_behavior", "Where do you usually buy air fryers?",
           ["Online", "Store", "Other"]),
    ]
    # Pad profile to look like a full tab for soft floor logic
    for i in range(3, 12):
        answered.append(_q(f"cp{i}", "consumer_profile", f"Unique profile question number {i} about lifestyle?"))

    kept = _drop_duplicates(
        answered,
        duplicates=[["cp1", "cp2"]],
        hard_duplicates=[["cp1", "cp2"]],
    )
    ids = {q["id"] for q in kept}
    assert "cp1" in ids
    assert "cp2" not in ids
    print("assembler hard drop OK")


if __name__ == "__main__":
    test_similarity()
    test_near_exact_uncapped_and_fail()
    test_assembler_hard_drop()
    print("A3 smoke PASSED")
