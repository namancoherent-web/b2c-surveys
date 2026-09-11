"""Pydantic v2 models.

Starts with only what Nodes 1-2 (scope_classify, evidence_harvester) need.
Additional models for questions, answers, critique, and the final survey grow
here as each node is built.
"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SegmentType(str, Enum):
    """Whether the market segment targets consumers or businesses."""

    B2C = "b2c"
    B2B = "b2b"
    OTHER = "other"


class ClassificationConfidence(str, Enum):
    """How sure the classifier is about segment_type."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class AudienceProfile(BaseModel):
    """The defined target audience for the survey."""

    demographics: dict = Field(
        default_factory=dict,
        description="Demographic/firmographic attributes (age, income, company size, etc.).",
    )
    behaviors: List[str] = Field(
        default_factory=list,
        description="Characteristic behaviors that define this audience.",
    )
    qualifiers: List[str] = Field(
        default_factory=list,
        description="Inclusion criteria — who counts as in-scope for this segment.",
    )
    exclusions: List[str] = Field(
        default_factory=list,
        description="Who is explicitly out of scope.",
    )
    rationale: str = Field(
        default="",
        description="Why this audience was defined the way it was.",
    )
    assumptions: List[str] = Field(
        default_factory=list,
        description="Assumptions made when inferring the audience from the raw segment.",
    )


class ScopeResult(BaseModel):
    """Output of Node 1 (scope_classify)."""

    normalized_segment: str = Field(
        description="Cleaned, canonical form of the raw market segment string.",
    )
    segment_type: SegmentType = Field(
        description=(
            "Detected market type based on who actually buys/uses the product: "
            "'b2c' if everyday consumers buy it, 'b2b' if businesses/institutions do, "
            "'other' if clearly non-consumer and not a commercial B2B goods market."
        ),
    )
    classification_confidence: ClassificationConfidence = Field(
        default=ClassificationConfidence.MEDIUM,
        description=(
            "high = clear pure B2C or pure B2B; medium/low = ambiguous or hybrid — "
            "prefer medium/low when unsure so the pipeline can still proceed."
        ),
    )
    is_hybrid: bool = Field(
        default=False,
        description=(
            "True when the commercial market is B2B-ish but a clear consumer / patient / "
            "end-user proxy exists (e.g. prescription drugs → patients). Hybrids should "
            "PROCEED with a consumer-proxy audience."
        ),
    )
    # change for b2c questionarie — A11: a DIFFERENT situation from is_hybrid.
    # is_hybrid = one buyer (a business) with a consumer end-user standing in
    # for them. sells_to_both = two genuine buyers, each paying for themselves.
    # Collapsing the two loses the fact that decides who gets surveyed.
    sells_to_both: bool = Field(
        default=False,
        description=(
            "True when the company genuinely sells to BOTH consumer and business "
            "buyers, each buying on their own account (e.g. a note-taking app on "
            "individual plans AND enterprise seats; a food delivery platform with "
            "consumer orders AND business catering accounts). This is NOT "
            "is_hybrid: here the consumer really is a paying customer, not a "
            "proxy for a business buyer."
        ),
    )
    dual_motion_reason: str = Field(
        default="",
        description=(
            "When sells_to_both, one or two sentences naming the consumer motion "
            "and the business motion separately, and which is the larger."
        ),
    )
    # change for b2c questionarie — A11: refuse to guess on thin input.
    needs_clarification: bool = Field(
        default=False,
        description=(
            "True when the input is too thin to classify — a bare company name "
            "with no description, or a phrase that could plausibly be a consumer "
            "or a business market. Set this INSTEAD of guessing; the pipeline "
            "stops and asks rather than generating the wrong study."
        ),
    )
    clarifying_question: str = Field(
        default="",
        description=(
            "When needs_clarification, the single most useful question to ask — "
            "the one whose answer would settle the classification. Ask about what "
            "the product does and who pays for it, never a generic 'tell me more'."
        ),
    )
    region: str = Field(
        default="Global",
        description="Legacy field — always 'Global' at classify time (fan-out is fixed).",
    )
    classification_reason: str = Field(
        description="Plain-language justification for the b2c vs b2b classification.",
    )
    audience_profile: AudienceProfile = Field(
        description="The defined consumer (or consumer-proxy) target audience for this segment.",
    )


# ---------------------------------------------------------------------------
# Node 3 — evidence_indexer
# ---------------------------------------------------------------------------
class RelevanceLevel(str, Enum):
    """How close the fact's source market is to the target segment."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class NumericFigure(BaseModel):
    """A real figure captured EXACTLY as stated — never rounded or rephrased."""

    measure: str = Field(description="What the number describes (e.g. 'market share of cheese segment').")
    number: float = Field(description="The numeric value as a float (e.g. 46.8).")
    unit: str = Field(description="Unit of the value: '%' | 'USD_mn' | 'USD' | 'CAGR_%' | etc.")
    period: Optional[str] = Field(
        default=None,
        description="Year or range the figure refers to, if stated (e.g. '2026', '2026-2033').",
    )
    verbatim: str = Field(
        description="The ORIGINAL figure string exactly as written (e.g. '46.8%', 'USD 7.4 Mn'). Preserve verbatim.",
    )


class IndexedFact(BaseModel):
    """One evidence fact, labeled with honest provenance + relevance for grounding."""

    model_config = ConfigDict(use_enum_values=True)

    fact: str = Field(description="The cleaned fact statement (copied from input).")
    origin: str = Field(
        description="Source origin: cmi_database | web | stat | reddit | exa | youtube | twitter (copied from input).",
    )
    source_ref: str = Field(description="Source reference: report id or URL (copied from input).")
    source_market: str = Field(
        description="The market the fact ACTUALLY describes (e.g. 'Sheep Milk', or the target segment itself).",
    )
    is_direct_match: bool = Field(
        description="True only if the fact is about THE target segment itself; False for related/adjacent markets (default False when unsure).",
    )
    relevance: RelevanceLevel = Field(
        description="How close source_market is to the target: direct=high; close substitute=medium; far=low.",
    )
    grounding_label: str = Field(
        description="Honest provenance to surface if this grounds an answer: '' if direct match, else 'based on related {source_market} market data'.",
    )
    tab: str = Field(
        description="Which survey tab the fact serves: consumer_profile | buying_behavior | preferences_expectations | satisfaction_future_intent | context.",
    )
    topic: str = Field(description="Clean specific topic label (e.g. 'regional market share', 'purchase frequency', 'price tier').")
    figure: Optional[NumericFigure] = Field(
        default=None,
        description="The extracted numeric figure if the fact carries one, else null.",
    )
    source_date: Optional[str] = Field(
        default=None,
        description="Published/updated date if known (copied from input), for recency preference.",
    )
    corroborated_by: List[str] = Field(
        default_factory=list,
        description="source_refs of other facts that state the same figure (corroboration).",
    )


class EvidenceIndex(BaseModel):
    """Legacy figure-index (kept for compatibility; simulator no longer pastes %)."""

    grounded_facts: List[IndexedFact] = Field(
        default_factory=list,
        description="All numeric facts (direct AND adjacent), each labeled.",
    )
    context_facts: List[IndexedFact] = Field(
        default_factory=list,
        description="Non-numeric relevant facts that inform plausible inference.",
    )
    by_tab: dict = Field(
        default_factory=dict,
        description="Map of tab -> list of indices into grounded_facts for fast lookup.",
    )
    grounded_count: int = Field(default=0, description="Total numeric facts available.")
    direct_match_count: int = Field(default=0, description="Direct-match numeric facts.")
    adjacent_count: int = Field(default=0, description="Adjacent-market numeric facts.")


# ---------------------------------------------------------------------------
# Evidence Synthesizer — Market Knowledge Profile
# ---------------------------------------------------------------------------
class BandLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class MarketMaturity(str, Enum):
    NASCENT = "nascent"
    GROWING = "growing"
    MATURE = "mature"
    SATURATED = "saturated"


class GrowthOutlook(str, Enum):
    CONTRACTING = "contracting"
    STABLE = "stable"
    GROWING = "growing"
    HIGH_GROWTH = "high_growth"


class TechAdoption(str, Enum):
    LAGGARD = "laggard"
    MAINSTREAM = "mainstream"
    EARLY = "early"
    CUTTING_EDGE = "cutting_edge"


class Sentiment(str, Enum):
    NEGATIVE = "negative"
    MIXED = "mixed"
    POSITIVE = "positive"


class MarketKnowledgeProfile(BaseModel):
    """Compressed market structure used by persona generation + survey simulation.

    Reddit may fill language/pain fields only — never quantitative bands.
    """

    model_config = ConfigDict(use_enum_values=True)

    market_maturity: MarketMaturity = Field(description="Category lifecycle stage.")
    avg_income_band: BandLevel = Field(description="Typical buyer purchasing power.")
    price_sensitivity: BandLevel = Field(description="How price-driven the category is.")
    growth_outlook: GrowthOutlook = Field(description="Near-term demand trajectory.")
    technology_adoption: TechAdoption = Field(description="How quickly buyers adopt new tech/features.")
    environmental_awareness: BandLevel = Field(description="Eco/sustainability salience.")
    brand_loyalty: BandLevel = Field(description="How locked-in buyers are to brands.")
    government_incentives: List[str] = Field(
        default_factory=list,
        description="Relevant incentives/policies (empty if none).",
    )
    top_pain_points: List[str] = Field(
        default_factory=list,
        description="Top consumer pains (may use Reddit/review language).",
    )
    top_purchase_drivers: List[str] = Field(
        default_factory=list,
        description="Top reasons people buy in this category.",
    )
    competitor_weaknesses: List[str] = Field(
        default_factory=list,
        description="Common competitor/brand weaknesses buyers cite.",
    )
    consumer_sentiment: Sentiment = Field(description="Overall consumer sentiment.")
    consumer_language: List[str] = Field(
        default_factory=list,
        description="Real consumer phrases for questionnaire option wording.",
    )
    feature_requests: List[str] = Field(
        default_factory=list,
        description="Common feature wishes from forums/reviews.",
    )
    regional_notes: dict = Field(
        default_factory=dict,
        description="region name -> short structural note (not percentages).",
    )
    evidence_confidence: BandLevel = Field(
        description="How confident we are in this profile given available evidence.",
    )
    citations: List[str] = Field(
        default_factory=list,
        description="source_refs supporting structural judgments (reports/stat/web).",
    )
    synthesis_notes: List[str] = Field(
        default_factory=list,
        description="Honesty notes: conflicts, thin coverage, Reddit excluded from bands.",
    )


# ---------------------------------------------------------------------------
# Persona Generator
# ---------------------------------------------------------------------------
class PersonaArchetype(BaseModel):
    """One weighted consumer archetype (not an individual respondent)."""

    id: str = Field(description="Stable id, e.g. 'eco_professional'.")
    name: str = Field(description="Display name, e.g. 'Eco Professional'.")
    age_band: str = Field(description="Age range, e.g. '28-40'.")
    income_band: str = Field(description="Income band, e.g. 'upper-middle'.")
    education: str = Field(description="Typical education level.")
    lifestyle: str = Field(description="Short lifestyle sketch.")
    tech_adoption: str = Field(description="laggard|mainstream|early|cutting_edge.")
    brand_loyalty: str = Field(description="low|medium|high.")
    price_sensitivity: str = Field(description="low|medium|high.")
    buying_motivation: List[str] = Field(default_factory=list)
    pain_points: List[str] = Field(default_factory=list)
    goals: List[str] = Field(default_factory=list)
    typical_behaviors: List[str] = Field(default_factory=list)


class PersonaCatalog(BaseModel):
    """10–20 archetypes plus per-region population weights (sum ≈ 100)."""

    personas: List[PersonaArchetype] = Field(description="10–20 archetypes for this segment.")
    weights_by_region: dict = Field(
        default_factory=dict,
        description="region -> {persona_id: weight_percent}. Each region sums to ~100.",
    )
    generation_notes: List[str] = Field(default_factory=list)
    persona_confidence: str = Field(
        default="medium",
        description="low|medium|high confidence in the archetype mix.",
    )


# ---------------------------------------------------------------------------
# Node 4 — question_architect / universal questionnaire
# ---------------------------------------------------------------------------
class QuestionType(str, Enum):
    """Closed (non-open-ended) question types the survey supports."""

    SINGLE_CHOICE = "single_choice"      # pick ONE; options exhaustive; Node 5 sums to 100%
    MULTIPLE_CHOICE = "multiple_choice"  # pick MANY; options independent; not summed to 100%
    RANKING = "ranking"                  # order the options
    LIKERT_5 = "likert_5"                # 5-point agreement/satisfaction scale
    LIKERT_7 = "likert_7"                # 7-point agreement/satisfaction scale


class SurveyQuestion(BaseModel):
    """One multiple-choice question. Answer percentages are added later by Node 5."""

    model_config = ConfigDict(use_enum_values=True)

    id: str = Field(description="Temporary id, e.g. 'cp1' (consumer_profile question 1).")
    tab: str = Field(description="Survey tab: consumer_profile | buying_behavior | preferences_expectations | satisfaction_future_intent.")
    text: str = Field(description="The question wording — clear, plain, jargon-free.")
    type: QuestionType = Field(description="Closed question type (see QuestionType).")
    options: List[str] = Field(
        description="Closed answer options (non-empty). Mutually exclusive for single_choice; independent for multiple_choice. Include an escape option ('Other'/'None'/'N/A') where relevant.",
    )
    chart_type: str = Field(
        description=(
            "How to render — one of: bar | horizontal_bar | radar | donut | stacked. "
            "Vary charts across the questionnaire; do not default everything to bar."
        ),
    )
    # change for b2c questionarie — within-tab funnel order (1 = broadest / easiest / safest)
    funnel_position: int = Field(
        default=1,
        description="Order WITHIN this question's storyline beat: 1 = broadest/easiest/safest; higher = narrower/harder/more sensitive.",
    )
    # change for b2c questionarie — Phase A8: storyline beat this question fills
    beat_id: str = Field(
        default="",
        description="The storyline beat this question fills. MUST be copied exactly from one of the beat ids listed for this chunk — it decides where the question lands in the survey's narrative order.",
    )
    serves_topic: Optional[str] = Field(
        default=None,
        description="The evidence-index topic this question's options align to, or null if none.",
    )
    evidence_aligned: bool = Field(
        description="True if the options map to an available grounded_fact topic (so Node 5 can attach real numbers).",
    )
    options_from_evidence: bool = Field(
        default=False,
        description="True if the OPTIONS were lifted from real consumer language/complaints/pain points in the evidence (independent of whether the percentages are grounded).",
    )

    @field_validator("funnel_position", mode="before")
    @classmethod
    def _coerce_funnel(cls, v):
        try:
            n = int(v)
            return n if n >= 1 else 1
        except (TypeError, ValueError):
            return 1

    @field_validator("type", mode="before")
    @classmethod
    def _coerce_type(cls, v):
        """Tolerate the LLM swapping a chart name into 'type' — coerce unknown
        question types to single_choice so one bad value can't fail the batch.

        CHECK 13/14 (B2C master rules, Section 26h) — this backend has no
        genuine ranking calculation (mean rank position, "% ranked #1"); a
        'ranking' type only ever displayed a plain percentage share
        indistinguishable from single_choice, which misled the reader into
        thinking a real rank order was measured. The prompt no longer asks
        for 'ranking', but coerce here too as a hard backstop in case the
        model emits it anyway.
        """
        valid = {e.value for e in QuestionType} - {QuestionType.RANKING.value}
        return v if isinstance(v, str) and v in valid else "single_choice"

    @field_validator("chart_type", mode="before")
    @classmethod
    def _coerce_chart(cls, v):
        """Coerce an unknown chart_type to a safe default."""
        return v if isinstance(v, str) and v in {
            "bar", "horizontal_bar", "radar", "donut", "stacked",
        } else "bar"


# change for b2c questionarie -- CHECK (spec: survey_wording_and_brand_policy
# _update.txt Part B / b2c_survey_question_structure_v3.txt Part C,
# 2026-09-11): brand names are now permitted, but ONLY when they survive a
# five-gate country check. A brand in a survey is a factual claim about a
# market -- if it is not actually sold in that country, every response
# distribution attached to it is fabricated and the survey is invalid. The
# model is explicitly NOT trusted as the source ("traceable to a
# verification step, not to the model's memory"), so the model's only job
# here is to JUDGE evidence that web search actually returned.
class BrandVerdict(BaseModel):
    """One candidate brand, judged against the five region gates."""

    brand: str = Field(
        description="Brand name exactly as marketed in the survey country.",
    )
    available: bool = Field(
        description="Gate 1 - sold through normal retail/distribution in that country.",
    )
    category_match: bool = Field(
        description="Gate 2 - sells THIS category in THAT country.",
    )
    active: bool = Field(
        description="Gate 3 - has not exited/withdrawn/discontinued the category there.",
    )
    local_name_correct: bool = Field(
        description="Gate 4 - written as it is actually marketed in that country.",
    )
    meaningful_presence: bool = Field(
        description="Gate 5 - real distribution, not a token listing.",
    )
    presence_rank: int = Field(
        default=99,
        description="1 = market leader. Used to order options by plausible presence.",
    )
    evidence_url: str = Field(
        default="",
        description="URL of the search result that supports this judgement.",
    )
    reason: str = Field(
        default="",
        description="One short sentence justifying the verdict.",
    )


class BrandVerification(BaseModel):
    """The verified brand set for one (country, category) pair."""

    verdicts: List[BrandVerdict] = Field(
        default_factory=list,
        description="One verdict per candidate brand considered.",
    )


class TabQuestionBatch(BaseModel):
    """The questions generated for a single tab by one LLM call."""

    tab: str = Field(description="The tab these questions belong to.")
    questions: List[SurveyQuestion] = Field(description="Generated questions for this tab.")
    notes: List[str] = Field(
        default_factory=list,
        description="Generation notes — e.g. evidence used, shortfalls, or thin-grounding caveats.",
    )


# change for b2c questionarie -- CHECK (user directive, 2026-09-10): a final
# category-fit review of the finished section. Slot coverage and wording
# rules are both enforced in code, but neither can tell whether a question
# is WORTH ASKING for this particular market -- that is a judgement call,
# and it is the one that produced the "Where do you usually do your laundry?"
# class of defect. This is the model grading its own finished work.
class QuestionFitVerdict(BaseModel):
    """One question's category-fit verdict from the self-review pass."""

    question_number: int = Field(
        description="1-based position of the question in the list shown for review.",
    )
    verdict: str = Field(
        description=(
            "Exactly one of: 'keep' (a strong question for this category) or "
            "'weak' (the honest answer would be near-identical for almost "
            "every respondent, or no business decision turns on it)."
        ),
    )
    reason: str = Field(
        default="",
        description="One short sentence — only required when verdict is 'weak'.",
    )
    replacement_text: str = Field(
        default="",
        description=(
            "When verdict is 'weak': a better question measuring the SAME "
            "underlying thing, written for this category. Empty when 'keep'."
        ),
    )
    replacement_options: List[str] = Field(
        default_factory=list,
        description=(
            "Answer options for replacement_text (4-7, mutually exclusive). "
            "Empty when 'keep'."
        ),
    )


class SectionFitReview(BaseModel):
    """Verdicts for every question in one finished section."""

    verdicts: List[QuestionFitVerdict] = Field(
        default_factory=list,
        description="One verdict per question reviewed, in the order shown.",
    )


# ---------------------------------------------------------------------------
# Node 4b — story_planner (Phase A8: narrative arc)
# change for b2c questionarie
# ---------------------------------------------------------------------------
class StoryBeat(BaseModel):
    """One ordered narrative slot inside a survey section."""

    beat_id: str = Field(
        description="Stable snake_case id for this beat, e.g. 'usage_occasion'.",
    )
    title: str = Field(description="Short human label, e.g. 'Usage occasions'.")
    intent: str = Field(
        description="What this beat must establish, in one sentence — the brief the question writer will follow.",
    )
    canonical: Optional[str] = Field(
        default=None,
        description=(
            "The closest CANONICAL beat id for this section (from the canonical arc given in the prompt), "
            "or null if this beat is genuinely category-specific with no canonical equivalent. "
            "Used to align the same narrative moment across regions — choose carefully."
        ),
    )


class StorySection(BaseModel):
    """One survey section, named for the category rather than generically."""

    section_id: str = Field(
        description="Stable snake_case id for this section, e.g. 'streaming_habits'.",
    )
    label: str = Field(
        description=(
            "Chapter heading of a research report, in standard market-research "
            "nomenclature: 3-6 words, usually a compound 'X & Y' — 'Consumption "
            "Behaviour & Viewing Context', 'Purchase Journey & Decision Drivers', "
            "'Satisfaction, Loyalty & Churn Risk'. Title Case, under ~45 characters. "
            "Never colloquial ('Habits', 'Wishlist') and never a bare one-word noun."
        ),
    )
    canonical: str = Field(
        description=(
            "The standard research job this section performs, from the fixed canonical "
            "list given in the prompt. No two sections may share one canonical."
        ),
    )
    remit: str = Field(
        default="",
        description="One sentence: what this section covers for this category.",
    )


class TabStoryline(BaseModel):
    """The ordered beats for one survey section."""

    tab: str = Field(
        description="The section_id these beats belong to — must match one you named in `sections`.",
    )
    beats: List[StoryBeat] = Field(
        description="Beats in NARRATIVE ORDER — first entry is asked first. 3-8 beats.",
    )


class RespondentSegment(BaseModel):
    """One behavioural/attitudinal segment the study exists to identify.

    # change for b2c questionarie — segmentation study output. Respondents never
    # see these labels; they are derived from a scoring model over the answers.
    """

    segment_id: str = Field(
        description="Stable snake_case id, e.g. 'value_optimizers'.",
    )
    name: str = Field(
        description=(
            "Research classification describing the defining BEHAVIOUR, in the "
            "same institutional register as the section headings — "
            "'High-Engagement Subscribers', 'Price-Driven Rotators', "
            "'Household Co-Viewers', 'Occasional Trialists'. Two or three words, "
            "Title Case. Never a marketing persona nickname ('Premium Bingers', "
            "'Content Hunters', 'Super Fans')."
        ),
    )
    description: str = Field(
        description=(
            "One or two sentences: how this group behaves in THIS category, what "
            "drives them, and how they differ from the others. Must be "
            "behavioural/attitudinal — never defined by age or income."
        ),
    )
    hypothesis: str = Field(
        default="",
        description="What we expect to be true of this group and want the data to confirm or kill.",
    )


class SurveyBlueprint(BaseModel):
    """The storyline for one segment in one region."""

    study_purpose: str = Field(
        default="",
        description=(
            "One sentence stating what this study is FOR — e.g. 'Identify which of "
            "4 behavioural segments each respondent belongs to, based on viewing "
            "habits, spend behaviour and content motivations.'"
        ),
    )
    # change for b2c questionarie — A11: name the lens BEFORE the segments, so
    # the cut is a stated methodological choice rather than whatever the model
    # happened to produce. Different categories segment on different axes.
    segmentation_lens: str = Field(
        default="",
        description=(
            "The primary lens these segments are cut on — usage/behavioural, "
            "jobs-to-be-done, price-sensitivity, psychographic, or a stated "
            "combination — AND one sentence on why it fits this category better "
            "than the alternatives."
        ),
    )
    # change for b2c questionarie — A12: a study that omits the defining issue
    # of its category reads as generic however well the questions are written.
    # The planner names those issues; the validator then checks the instrument
    # actually covers them. Declared per category, so nothing is hardcoded.
    must_cover_topics: List[str] = Field(
        default_factory=list,
        description=(
            "3-6 topics THIS category cannot credibly omit — the things a "
            "specialist would notice were missing. Each is 1-4 plain words that "
            "will appear in a question about it (e.g. for streaming: 'account "
            "sharing', 'ad-supported tier', 'monthly spend', 'devices used'; for "
            "running shoes: 'replacement cycle', 'injury', 'fit', 'price paid'). "
            "Name the category's real commercial issues, not generic research "
            "topics."
        ),
    )
    segments: List[RespondentSegment] = Field(
        default_factory=list,
        description=(
            "Exactly 4 behavioural segments this study must be able to tell apart. "
            "They must be mutually exclusive, collectively cover the market, and "
            "each be reachable through closed questions about behaviour."
        ),
    )
    sections: List[StorySection] = Field(
        default_factory=list,
        description="3-6 category-specific sections in narrative order.",
    )
    storyline: List[TabStoryline] = Field(
        description="One entry per section you named, keyed by your section_id.",
    )
    narrative_summary: str = Field(
        default="",
        description="One short paragraph describing the story this survey tells, start to finish.",
    )
    adaptation_notes: List[str] = Field(
        default_factory=list,
        description="What was changed from the canonical arc for this category/region, and why.",
    )


# ---------------------------------------------------------------------------
# Node 5 — answer_estimator
# ---------------------------------------------------------------------------
class AnswerOption(BaseModel):
    """One option's estimated share, with honest grounding provenance."""

    model_config = ConfigDict(use_enum_values=True)

    label: str = Field(description="The option label (must match the question option).")
    percentage: float = Field(description="Estimated percentage for this option (0-100).")
    grounded_in: Optional[str] = Field(
        default=None,
        description="source_ref of the real figure backing this option, or null if inferred.",
    )
    source_market: Optional[str] = Field(
        default=None,
        description="Market the backing figure describes; null if inferred.",
    )
    grounding_label: str = Field(
        default="",
        description="'' if direct or inferred; 'based on related {market} data' if borrowed from an adjacent market.",
    )
    confidence: str = Field(
        default="low",
        description="high (direct + recent/stat/corroborated) | medium (adjacent or context-backed) | low (inferred).",
    )

    @field_validator("confidence", mode="before")
    @classmethod
    def _coerce_conf(cls, v):
        return v if isinstance(v, str) and v in {"high", "medium", "low"} else "low"


class AnsweredQuestion(SurveyQuestion):
    """A SurveyQuestion with its estimated answer distribution attached."""

    answers: List[AnswerOption] = Field(description="Per-option estimated distribution.")
    sample_size: int = Field(description="Synthetic respondent base used for the distribution.")
    distribution_note: str = Field(
        default="",
        description="Short note: which figures grounded which options, and what was inferred.",
    )
    is_grounded: bool = Field(
        default=False,
        description="True if at least one option is grounded_in a real figure.",
    )
    question_confidence: str = Field(
        default="low",
        description="Rollup confidence for the question: high|medium|low.",
    )
    # change for b2c questionarie — carried through simulation so the assembler
    # and Phase A6 can still see layer + story position. Previously dropped here,
    # which silently forced every published question to question_layer='full'.
    question_layer: str = Field(
        default="full",
        description="core | module | full — which instrument layer produced this question.",
    )
    narrative_order: int = Field(
        default=0,
        description="Position of this question in its tab's storyline (1 = asked first).",
    )
    beat_title: str = Field(
        default="",
        description="Human label of the storyline beat this question fills.",
    )
    beat_canonical: Optional[str] = Field(
        default=None,
        description="Canonical beat id this question's beat maps to — the cross-region anchor.",
    )


# ---------------------------------------------------------------------------
# Node 6 — validator_critic
# ---------------------------------------------------------------------------
class QuestionIssue(BaseModel):
    """Flagged problems for one question, with where the fix belongs."""

    id: str = Field(description="The question id this issue applies to.")
    issues: List[str] = Field(description="Specific problems found (do not rewrite — only flag).")
    fix_target: str = Field(
        description=(
            "Where to route the fix: 'architect' (question/dedup/options), "
            "'simulator' (answer distributions / persona consistency), "
            "or legacy 'estimator' (treated as simulator)."
        ),
    )


class Critique(BaseModel):
    """The validator's audit of the answered survey — drives pass/loop routing."""

    score: float = Field(description="Quality score 0.0-1.0 (math + label integrity dominate).")
    passed: bool = Field(description="True if the survey passes the integrity gate (suggest score >= 0.85).")
    feedback: List[str] = Field(default_factory=list, description="Set-level actionable notes.")
    per_question: List[QuestionIssue] = Field(
        default_factory=list,
        description="Per-question issues with fix_target for targeted revision.",
    )
    duplicates: List[List[str]] = Field(
        default_factory=list,
        description="Clusters of duplicate question ids (first id is the representative to keep).",
    )
    # change for b2c questionarie — near-exact clusters always dropped (uncapped)
    hard_duplicates: List[List[str]] = Field(
        default_factory=list,
        description="Near-identical duplicate clusters (always remove extras; no soft cap).",
    )
    grounded_pct: float = Field(default=0.0, description="% of all answer-options grounded in a real figure.")
    direct_grounded_pct: float = Field(default=0.0, description="% of options grounded by a DIRECT match (empty label).")
    adjacent_grounded_pct: float = Field(default=0.0, description="% of options grounded but adjacent (labeled).")
    counts_by_tab: dict = Field(default_factory=dict, description="Questions per tab vs the plan.")
