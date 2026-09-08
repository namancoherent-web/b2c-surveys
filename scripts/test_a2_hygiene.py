"""Phase A2 smoke checks — hygiene helpers + funnel sort (no LLM)."""
from src.models import SurveyQuestion
from src.nodes.assembler import build_survey_json
from src.nodes.validator_critic import (
    _has_escape_option,
    _likert_balanced,
    _likert_direction,
    _looks_double_barreled,
    _looks_leading,
    _needs_escape_hatch,
    validator_critic,
)
from src.nodes.question_architect import _Q_SCHEMA_V


def test_helpers():
    assert _looks_double_barreled("How would you rate price and quality?")
    assert not _looks_double_barreled("How often do you use an air fryer?")
    assert _looks_leading("Wouldn't you agree this product is great?")
    assert not _looks_leading("How would you rate this product?")
    assert _needs_escape_hatch("single_choice", ["Brand A", "Brand B", "Brand C"])
    assert not _has_escape_option(["Brand A", "Brand B"])
    assert _has_escape_option(["Brand A", "Other"])
    assert not _needs_escape_hatch(
        "single_choice",
        ["Daily", "Weekly", "Monthly", "Rarely", "Never"],
    )
    ok, _ = _likert_balanced(
        "likert_5",
        [
            "Strongly disagree",
            "Disagree",
            "Neither agree nor disagree",
            "Agree",
            "Strongly agree",
        ],
    )
    assert ok
    bad, reason = _likert_balanced(
        "likert_5",
        ["Great", "Good", "OK", "Meh", "Bad"],
    )
    assert not bad, reason
    assert _likert_direction(
        ["Strongly disagree", "Disagree", "Neither", "Agree", "Strongly agree"]
    ) == "neg_to_pos"
    assert _Q_SCHEMA_V.startswith("questionnaire_v")
    q = SurveyQuestion(
        id="cp1",
        tab="consumer_profile",
        text="How often do you cook?",
        type="single_choice",
        options=["Daily", "Weekly", "Monthly", "Rarely", "Never"],
        chart_type="bar",
        evidence_aligned=False,
        funnel_position=3,
    )
    assert q.funnel_position == 3
    print("helpers OK")


def test_validator_hygiene_fail():
    answered = [
        {
            "id": "bb1",
            "tab": "buying_behavior",
            "text": "How would you rate price and service?",
            "type": "single_choice",
            "options": ["High", "Medium", "Low", "Other"],
            "funnel_position": 1,
            "answers": [
                {"label": "High", "percentage": 25.0, "grounded_in": None,
                 "source_market": None, "grounding_label": "", "confidence": "medium"},
                {"label": "Medium", "percentage": 25.0, "grounded_in": None,
                 "source_market": None, "grounding_label": "", "confidence": "medium"},
                {"label": "Low", "percentage": 25.0, "grounded_in": None,
                 "source_market": None, "grounding_label": "", "confidence": "medium"},
                {"label": "Other", "percentage": 25.0, "grounded_in": None,
                 "source_market": None, "grounding_label": "", "confidence": "medium"},
            ],
            "sample_size": 2500, "distribution_note": "", "is_grounded": False,
            "question_confidence": "medium",
        }
    ]
    # Pad to avoid quota fail drowning the signal — use MIN floor awareness
    from src.question_plan import CATEGORY_PLAN, QUESTIONS_PER_TAB_MIN

    tabs = [t for t, _ in CATEGORY_PLAN]
    full = []
    for tab in tabs:
        for i in range(QUESTIONS_PER_TAB_MIN):
            if tab == "buying_behavior" and i == 0:
                full.append(answered[0])
                continue
            full.append({
                "id": f"{tab}_{i}",
                "tab": tab,
                "text": f"In the last 30 days, how often did you use product option {i}?",
                "type": "single_choice",
                "options": ["Daily", "Weekly", "Monthly", "Rarely", "Never"],
                "funnel_position": i + 1,
                "answers": [
                    {"label": "Daily", "percentage": 20.0, "grounded_in": None,
                     "source_market": None, "grounding_label": "", "confidence": "medium"},
                    {"label": "Weekly", "percentage": 20.0, "grounded_in": None,
                     "source_market": None, "grounding_label": "", "confidence": "medium"},
                    {"label": "Monthly", "percentage": 20.0, "grounded_in": None,
                     "source_market": None, "grounding_label": "", "confidence": "medium"},
                    {"label": "Rarely", "percentage": 20.0, "grounded_in": None,
                     "source_market": None, "grounding_label": "", "confidence": "medium"},
                    {"label": "Never", "percentage": 20.0, "grounded_in": None,
                     "source_market": None, "grounding_label": "", "confidence": "medium"},
                ],
                "sample_size": 2500, "distribution_note": "", "is_grounded": False,
                "question_confidence": "medium",
            })
    import src.nodes.validator_critic as vc

    vc._llm_judge = lambda *a, **k: vc._JudgeResult(duplicates=[], issues=[])
    out = validator_critic({
        "answered_questions": full,
        "revision_count": 0,
        "segment": "test",
        "evidence_pack": {"claims": []},
        "counts_by_tab": {t: QUESTIONS_PER_TAB_MIN for t, _ in CATEGORY_PLAN},
    })
    c = out["critique"]
    assert c["passed"] is False
    assert any("HYGIENE" in f for f in c["feedback"])
    assert any("double-barrel" in " ".join(qi["issues"]) for qi in c["per_question"])
    print("validator hygiene fail OK")


def test_assembler_funnel_sort():
    state = {
        "answered_questions": [
            {
                "id": "cp2", "tab": "consumer_profile", "text": "Sensitive Q",
                "type": "single_choice", "options": ["A", "B", "Other"],
                "chart_type": "bar", "funnel_position": 2, "evidence_aligned": False,
                "options_from_evidence": False,
                "answers": [
                    {"label": "A", "percentage": 50.0, "grounded_in": None,
                     "source_market": None, "grounding_label": "", "confidence": "medium"},
                    {"label": "B", "percentage": 30.0, "grounded_in": None,
                     "source_market": None, "grounding_label": "", "confidence": "medium"},
                    {"label": "Other", "percentage": 20.0, "grounded_in": None,
                     "source_market": None, "grounding_label": "", "confidence": "medium"},
                ],
                "sample_size": 2500, "is_grounded": False, "distribution_note": "",
            },
            {
                "id": "cp1", "tab": "consumer_profile", "text": "Broad Q",
                "type": "single_choice", "options": ["Daily", "Weekly", "Monthly", "Rarely", "Never"],
                "chart_type": "bar", "funnel_position": 1, "evidence_aligned": True,
                "options_from_evidence": False,
                "answers": [
                    {"label": "Daily", "percentage": 20.0, "grounded_in": None,
                     "source_market": None, "grounding_label": "", "confidence": "medium"},
                    {"label": "Weekly", "percentage": 20.0, "grounded_in": None,
                     "source_market": None, "grounding_label": "", "confidence": "medium"},
                    {"label": "Monthly", "percentage": 20.0, "grounded_in": None,
                     "source_market": None, "grounding_label": "", "confidence": "medium"},
                    {"label": "Rarely", "percentage": 20.0, "grounded_in": None,
                     "source_market": None, "grounding_label": "", "confidence": "medium"},
                    {"label": "Never", "percentage": 20.0, "grounded_in": None,
                     "source_market": None, "grounding_label": "", "confidence": "medium"},
                ],
                "sample_size": 2500, "is_grounded": False, "distribution_note": "",
            },
        ],
        "critique": {"duplicates": [], "grounded_pct": 0},
        "normalized_segment": "test",
        "region": "Global",
    }
    survey = build_survey_json(state)["survey"]
    profile = next(t for t in survey["tabs"] if t["tab"] == "consumer_profile")
    assert profile["questions"][0]["text"] == "Broad Q"
    assert profile["questions"][1]["text"] == "Sensitive Q"
    assert profile["questions"][0]["funnel_position"] == 1
    print("assembler funnel sort OK")


if __name__ == "__main__":
    test_helpers()
    test_validator_hygiene_fail()
    test_assembler_funnel_sort()
    print("A2 smoke PASSED")
