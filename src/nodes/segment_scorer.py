"""Node 8 — segment_scorer.

# change for b2c questionarie — Phase A9 (segmentation study)

Builds the scoring model that turns a completed questionnaire into a segment
assignment: for each question, which answer options point to which segment and
how strongly, plus a tie-break rule.

This is what makes the output a segmentation STUDY rather than a description of
the category. Without it a client has 30 percentages and no way to say which
group a respondent belongs to.

Rules enforced here, not left to the model:
  - Profiling questions (age, income, household, NPS) are NEVER scoring inputs.
    They exist for cross-tabs. Feeding demographics into the segmentation would
    turn a behavioural segmentation into a demographic one.
  - Screening questions are excluded too — everyone who reached the survey
    passed them, so they carry no discriminating signal.
  - Only options that actually exist on the question can carry weight.
"""

from __future__ import annotations

import json

from pydantic import BaseModel, Field

from src import narrative
from src.llm import get_structured_llm
from src.state import SurveyState

_EXCLUDED_LAYERS = {"screening", "profiling"}


class _OptionWeight(BaseModel):
    question_id: str = Field(description="The question id this rule applies to.")
    option_label: str = Field(description="The answer option, copied EXACTLY.")
    segment_id: str = Field(description="Which segment this answer points to.")
    points: int = Field(description="How strongly, 1 (weak) to 3 (decisive).")


class _ScoringModel(BaseModel):
    weights: list[_OptionWeight] = Field(
        description="One entry per (question, option, segment) signal worth scoring.",
    )
    tie_break: str = Field(
        default="",
        description=(
            "One sentence: which single question decides when a respondent ties "
            "across two segments, and why that question is the most direct "
            "behavioural self-report."
        ),
    )
    notes: list[str] = Field(default_factory=list)


PROMPT = """\
You are building the SEGMENTATION SCORING MODEL for a completed B2C
questionnaire on "{segment}".

STUDY PURPOSE: {purpose}

SEGMENTS TO ASSIGN (a respondent ends up in exactly one):
{segments_block}

Below are the survey's questions with their exact answer options. For each
option that genuinely signals membership of a segment, emit a weight.

RULES (never violate):
1. Copy `question_id` and `option_label` EXACTLY as given. A weight on an option
   that does not exist is discarded.
2. `points`: 3 = decisive on its own (a stated behaviour pattern that defines
   the segment), 2 = strong, 1 = mild supporting signal.
3. Only score options that DISCRIMINATE. If an option is equally likely across
   all segments, leave it unscored — most options should be unscored.
4. Every segment must be reachable: each one needs at least two scoring options
   across the survey, and at least one worth 3 points.
5. An option may point to more than one segment when it genuinely does, but
   prefer a single clear assignment.
6. `tie_break`: name the ONE question that settles a tie and say why — it should
   be the most direct behavioural self-report in the survey.

QUESTIONS (JSON):
{questions_json}
"""


def _segments_block(segments: list[dict]) -> str:
    lines = []
    for i, s in enumerate(segments, start=1):
        lines.append(f"  {i}. [{s.get('segment_id')}] {s.get('name')}")
        if s.get("description"):
            lines.append(f"       {s['description']}")
    return "\n".join(lines)


def _scoreable(questions: list[dict]) -> list[dict]:
    """Behavioural questions only — never screening or profiling."""
    return [
        q for q in questions
        if (q.get("question_layer") or "") not in _EXCLUDED_LAYERS
    ]


def segment_scorer(state: SurveyState) -> dict:
    """Produce the response -> segment scoring table."""
    blueprint = state.get("survey_blueprint") or {}
    segments = narrative.segments_for(blueprint)
    answered = state.get("answered_questions") or []
    segment_name = (
        state.get("normalized_segment") or state.get("market_segment") or "the category"
    )

    if not segments or not answered:
        return {"segment_model": {}}

    scoreable = _scoreable(answered)
    if not scoreable:
        return {"segment_model": {}}

    valid_ids = {s["segment_id"] for s in segments}
    payload = [
        {
            "question_id": q.get("id"),
            "section": q.get("tab"),
            "text": q.get("text"),
            "type": q.get("type"),
            "options": q.get("options") or [],
        }
        for q in scoreable
    ]

    prompt = PROMPT.format(
        segment=segment_name,
        purpose=narrative.study_purpose(blueprint) or "Segment the category's buyers.",
        segments_block=_segments_block(segments),
        questions_json=json.dumps(payload, ensure_ascii=False),
    )

    try:
        result: _ScoringModel = get_structured_llm(
            _ScoringModel, temperature=0.1, max_tokens=8000
        ).invoke(prompt)
    except Exception as exc:  # noqa: BLE001 — a failed model must not sink the run
        return {
            "segment_model": {
                "segments": segments,
                "weights": [],
                "tie_break": "",
                "notes": [f"Scoring model unavailable ({type(exc).__name__})."],
            }
        }

    # Keep only weights that reference a real question, a real option on that
    # question, and a real segment — the model does drift on exact strings.
    by_id = {q.get("id"): q for q in scoreable}
    kept, dropped = [], 0
    for w in result.weights:
        q = by_id.get(w.question_id)
        if not q or w.segment_id not in valid_ids:
            dropped += 1
            continue
        options = [str(o) for o in (q.get("options") or [])]
        match = next(
            (o for o in options if o.strip().lower() == w.option_label.strip().lower()),
            None,
        )
        if match is None:
            dropped += 1
            continue
        kept.append({
            "question_id": w.question_id,
            "question_text": q.get("text"),
            "option_label": match,
            "segment_id": w.segment_id,
            "points": max(1, min(3, int(w.points))),
        })

    reachable = {w["segment_id"] for w in kept}
    notes = list(result.notes or [])
    if dropped:
        notes.append(f"{dropped} weight(s) discarded — unknown question, option or segment.")
    unreachable = [s["name"] for s in segments if s["segment_id"] not in reachable]
    if unreachable:
        notes.append(
            "No scoring signal for: " + ", ".join(unreachable)
            + " — these segments cannot be assigned from this questionnaire."
        )

    return {
        "segment_model": {
            "segments": segments,
            "weights": kept,
            "tie_break": (result.tie_break or "").strip(),
            "notes": notes[:10],
            "excluded": (
                "Screening and Profiling questions are excluded from scoring: "
                "screeners carry no signal (everyone passed) and demographics "
                "are for cross-tabs, not segmentation."
            ),
        }
    }
