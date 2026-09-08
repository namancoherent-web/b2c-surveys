"""LangGraph shared state for the survey-generation pipeline.

Persona-simulation architecture:

    scope → harvest → synthesize(MKP) → personas → questionnaire
         → survey simulator → consistency validator → assembler

``evidence_pool`` uses an ``operator.add`` reducer so parallel harvest branches
accumulate.

# change for b2c questionarie — Phase A5: region jobs produce region-native
questionnaires (core + module). ``questions_by_region`` aligns with
``answered_by_region``. Graph wiring is unchanged; region is driven via state.
"""

import operator
from typing import Annotated, Dict, List, TypedDict


class SurveyState(TypedDict, total=False):
    # Node 1 — scope_classify (runs once per segment, before fan-out)
    market_segment: str
    normalized_segment: str
    segment_type: str  # "b2c" | "b2b" | "other"
    classification_confidence: str  # "high" | "medium" | "low"
    is_hybrid: bool  # B2B market with a consumer-proxy end user (patients)
    # change for b2c questionarie — A11: sells to consumers AND businesses, each
    # paying on their own account. Distinct from is_hybrid — see scope_classify.
    sells_to_both: bool
    dual_motion_reason: str
    needs_clarification: bool
    clarifying_question: str
    region: str
    # change for b2c questionarie — the country this file is written FOR, and
    # the label to print for it. LangGraph drops any key not declared here, so
    # without these the country never reached the assembler and every country
    # file was titled with its parent region.
    country: str
    geography_label: str
    regions_to_simulate: List[str]
    classification_reason: str
    audience_profile: dict
    # change for b2c questionarie — B2C gate
    gate_passed: bool
    gate_decision: str
    status: str  # "accepted" | "rejected" | "needs_clarification"
    rejection_reason: str
    classification_preseeded: bool

    # Node 2 — evidence_harvester
    evidence_pool: Annotated[List[dict], operator.add]
    evidence_empty: bool

    # Node 3 — evidence_synthesizer (Market Knowledge Profile)
    market_knowledge_profile: dict
    grounding_thin: bool  # True when MKP confidence is low / evidence empty

    # Legacy indexer output (optional / unused by simulator path)
    evidence_index: dict

    # Node 4 — persona_generator
    persona_catalog: dict

    # Node 4b — story_planner (Phase A8)
    # change for b2c questionarie — ordered narrative beats per tab, planned per
    # segment AND per region; question order derives from this, never from raw
    # model output. See src/narrative.py.
    survey_blueprint: dict

    # Node 5 — questionnaire (question_architect)
    questions: List[dict]
    # change for b2c questionarie — per-region question sets (slug -> list)
    questions_by_region: Dict[str, List[dict]]
    counts_by_tab: dict
    questionnaire_id: str
    questionnaire_cache_hit: bool

    # Node 6 — survey_simulator
    answered_questions: List[dict]
    answered_by_region: dict  # region slug -> list of answered questions
    grounded_question_count: int  # retained name: count with medium+ sim confidence
    simulation_confidence_avg: float

    # Node 7 — consistency_validator
    critique: dict
    grounded_pct: float  # retained for assembler metadata; maps to sim confidence
    revision_count: int

    # change for b2c questionarie — best-so-far snapshot.
    # The revision loop is not monotonic: a pass can score 0.742 and the next
    # 0.031. Without this the assembler ships whatever the LAST pass produced,
    # discarding the best work. validator_critic writes these whenever a pass
    # improves; assembler restores them if the final pass is worse.
    best_score: float
    best_answered: List[dict]
    best_critique: dict
    best_counts_by_tab: dict

    # Node 8 — segment_scorer (Phase A9)
    # change for b2c questionarie — response -> segment scoring table.
    segment_model: dict

    # Node 9 — assembler
    final_survey: dict
