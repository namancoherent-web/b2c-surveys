"""LangGraph wiring for the persona-simulation survey pipeline.

# change for b2c questionarie

Classify-once flow:
    classify graph:  scope_classify → [gate] → END | reject_stop → END
    region graph:    evidence_harvester → … → assembler → END
                     (skips scope_classify; classification pre-seeded in state)

Full graph (optional single-shot): scope_classify → [gate] → harvest… | reject_stop
"""

import os

from langgraph.graph import END, StateGraph

from src.nodes.assembler import assembler
from src.nodes.evidence_harvester import evidence_harvester
from src.nodes.evidence_synthesizer import evidence_synthesizer
from src.nodes.persona_generator import persona_generator
from src.nodes.question_architect import question_architect
from src.nodes.scope_classify import clarify_stop, reject_stop, scope_classify
from src.nodes.segment_scorer import segment_scorer
from src.nodes.story_planner import story_planner
from src.nodes.survey_simulator import survey_simulator
from src.nodes.validator_critic import validator_critic
from src.state import SurveyState

# change for b2c questionarie — 3 passes were not enough once the validator
# started catching duplicates and grammar as well as structure: a run with 16
# flagged questions used all three and shipped with most of them intact. The
# assembler already keeps the BEST-scoring revision, so extra passes can only
# help; each one costs ~30s.
MAX_REVISIONS = int(os.getenv("MAX_REVISIONS", "6"))


def route_after_scope(state) -> str:
    """B2C gate: proceed | reject | clarify.

    # change for b2c questionarie — A11: an input we could not classify gets its
    # own terminal, so "we don't know" is never recorded as "we said no".
    """
    if state.get("gate_passed"):
        return "proceed"
    if state.get("status") == "needs_clarification":
        return "clarify"
    return "reject"


def route_after_validator(state) -> str:
    """Return the next node name based on the critique."""
    critique = state.get("critique") or {}

    if critique.get("passed"):
        return "assembler"
    if (state.get("revision_count") or 0) >= MAX_REVISIONS:
        return "assembler"

    targets = {qi.get("fix_target") for qi in (critique.get("per_question") or [])}
    if "estimator" in targets:
        targets.add("simulator")
    feedback_blob = " ".join(str(f) for f in (critique.get("feedback") or [])).lower()
    needs_architect = (
        "architect" in targets
        or "quota" in feedback_blob
        or "under target" in feedback_blob
        or "irrelevant" in feedback_blob
        or "geo-residence" in feedback_blob
        or "geography" in feedback_blob
        # change for b2c questionarie -- CHECK (user directive, 2026-09-08):
        # validator_critic now emits "coverage: ..." feedback when one of the
        # 16 required question types is missing or the NPS scale is malformed.
        # Only the architect can write a missing question, so this MUST route
        # here -- without it the gate would mark the pass dirty but never
        # actually regenerate, and the run would burn all 6 revisions still
        # missing the same question.
        or "coverage:" in feedback_blob
        # change for b2c questionarie -- CHECK (spec: change-spec Part C.5 /
        # v3 Part D, 2026-09-11): register defects (contractions, "I ..."
        # options, over-formal wording, length ceilings) and brand-policy
        # defects are both rewrites of question TEXT and OPTION LABELS, so
        # like coverage they can only be fixed by the architect.
        or "register:" in feedback_blob
        or "brand:" in feedback_blob
    )
    if needs_architect:
        return "question_architect"
    if "personas" in targets:
        return "persona_generator"
    if "simulator" in targets:
        return "survey_simulator"
    return "assembler"


def _add_pipeline_nodes(g: StateGraph) -> None:
    """Shared harvest → assemble chain (no scope_classify)."""
    g.add_node("evidence_harvester", evidence_harvester)
    g.add_node("evidence_synthesizer", evidence_synthesizer)
    g.add_node("persona_generator", persona_generator)
    # change for b2c questionarie — Phase A8: plan the narrative arc before
    # any question is written, so question order is deliberate, not emergent.
    g.add_node("story_planner", story_planner)
    g.add_node("question_architect", question_architect)
    g.add_node("survey_simulator", survey_simulator)
    g.add_node("validator_critic", validator_critic)
    # change for b2c questionarie — Phase A9: build the segmentation
    # scoring model once the questions are final.
    g.add_node("segment_scorer", segment_scorer)
    g.add_node("assembler", assembler)

    g.add_edge("evidence_harvester", "evidence_synthesizer")
    g.add_edge("evidence_synthesizer", "persona_generator")
    g.add_edge("persona_generator", "story_planner")
    g.add_edge("story_planner", "question_architect")
    g.add_edge("question_architect", "survey_simulator")
    g.add_edge("survey_simulator", "validator_critic")
    g.add_conditional_edges(
        "validator_critic",
        route_after_validator,
        {
            "question_architect": "question_architect",
            "persona_generator": "persona_generator",
            "survey_simulator": "survey_simulator",
            "assembler": "segment_scorer",
        },
    )
    g.add_edge("segment_scorer", "assembler")
    g.add_edge("assembler", END)


def build_classify_graph():
    """Segment-only classify + B2C gate. Accepted → END; rejected → reject_stop."""
    # change for b2c questionarie
    g = StateGraph(SurveyState)
    g.add_node("scope_classify", scope_classify)
    g.add_node("reject_stop", reject_stop)
    g.add_node("clarify_stop", clarify_stop)
    g.set_entry_point("scope_classify")
    g.add_conditional_edges(
        "scope_classify",
        route_after_scope,
        {
            "proceed": END,
            "reject": "reject_stop",
            "clarify": "clarify_stop",
        },
    )
    g.add_edge("reject_stop", END)
    g.add_edge("clarify_stop", END)
    return g.compile()


def build_region_graph():
    """One geographic region job — starts at evidence_harvester (no re-classify)."""
    # change for b2c questionarie
    g = StateGraph(SurveyState)
    _add_pipeline_nodes(g)
    g.set_entry_point("evidence_harvester")
    return g.compile()


def build_graph():
    """Full graph with B2C gate then regional pipeline (legacy / debug)."""
    g = StateGraph(SurveyState)
    g.add_node("scope_classify", scope_classify)
    g.add_node("reject_stop", reject_stop)
    g.add_node("clarify_stop", clarify_stop)
    _add_pipeline_nodes(g)

    g.set_entry_point("scope_classify")
    g.add_conditional_edges(
        "scope_classify",
        route_after_scope,
        {
            "proceed": "evidence_harvester",
            "reject": "reject_stop",
            "clarify": "clarify_stop",
        },
    )
    g.add_edge("reject_stop", END)
    g.add_edge("clarify_stop", END)
    return g.compile()
