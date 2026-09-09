"""Node 4 — questionnaire architect.

# change for b2c questionarie — Phase A5
For geographic region jobs with REGION_QUESTION_MODE=core_plus_module:
shared CORE questions (segment-wide cache) + regional MODULE questions
(region-specific option language). Fully_distinct / Global use a full tab fill.
No answer percentages here — simulator owns distributions.
"""

import os
import re
from concurrent.futures import ThreadPoolExecutor

from src import cache_store, config, narrative, standard_sections
from src.date_utils import current_date_context
from src.llm import get_structured_llm
from src.models import TabQuestionBatch
from src.nodes.validator_critic import CONSTRUCTS as _VALIDATOR_CONSTRUCTS
from src.question_plan import (
    CATEGORY_PLAN,
    CORE_PER_TAB,
    GENERATION_MODE,
    GEOGRAPHIC_REGIONS,
    MODULE_PER_TAB,
    QUESTIONNAIRE_CACHE_MIN_QUESTIONS,
    REGION_QUESTION_MODE,
    QUESTIONS_PER_TAB_TARGET,
    match_geographic_region,
    region_to_slug,
)
from src.question_similarity import LOOSE_JACCARD, too_similar_to_any
from src.state import SurveyState

# change for b2c questionarie — v10: sections are now market-specific, so cached
# questions are tagged to section ids that a new blueprint may not contain. v9
# entries (fixed four sections) and v8 (no beat_id) must not be reused — the
# shared CORE cache in particular is keyed per segment, not per blueprint, so a
# stale hit would silently drop every core question.
# v11: simplified question craft (6-12 word stems, plain words) and
# Screening removed from the instrument.
# v12: clarity over brevity — 8-16 word stems that name the category and
# stand alone. v11 optimised for shortness and produced ambiguous stems.
# v13: no "None / I do not use it" options (the audience definition already
# qualifies every respondent as a user), and must-have / trade-off items may
# not be forced single-choice.
# v14: the category's must-cover topics reach the architect prompt.
# v15: "Which of these best describes" banned (it appeared on 7 of 29
# questions), openers must vary, and single-pick options may not nest.
# v16: senior-analyst voice — category-specific, behaviour over self-image,
# one register throughout, and no two questions sharing meaning OR shape.
# v17: no behavioural question may re-ask a Profiling demographic, and
# "Which of these" is stripped as padding at publication.
# v18: CORE-layer questions may never hardcode a currency symbol/amount --
# the core cache is keyed by segment only (never geography), so a stale
# cache entry from before this rule was live could otherwise keep serving
# a hardcoded-USD core question to every country forever, silently
# defeating the fix. Confirmed live: the SAME USD spend question shipped
# on two consecutive runs after the currency fix landed, traced to a core
# cache entry populated days earlier under the unchanged v17 key. The
# version bump forces every core batch to regenerate under the new rule.
_Q_SCHEMA_V = "questionnaire_v18"

# change for b2c questionarie -- CHECK (user directive, 2026-09-08): short-
# survey format (16 questions: 4 per behavioural section, no anchors). Set
# QUESTIONS_PER_TAB_TARGET=4 in question_plan.py made this the ONLY mode --
# the satisfaction+NPS anchors used to sit OUTSIDE the 4 generated slots,
# which would make the satisfaction section 6 questions instead of 4. User's
# explicit choice: drop the anchors, let all 4 slots in that section be
# freshly generated (satisfaction/NPS/pain-point/future-intent guidance is
# now baked into _TAB_GUIDANCE below instead). REVERT NOTE: if this format
# is ever rolled back, restore by setting this to False and reverting
# question_plan.py's QUESTIONS_PER_TAB_MIN/MAX/TARGET back to env-driven
# 5/10/8 -- both changes must move together, the anchors were only safe to
# drop because the guidance list replaces what they used to guarantee.
_SHORT_SURVEY_NO_ANCHORS = True

# Temp-id prefixes per tab.
_TAB_PREFIX = {
    "consumer_profile": "cp",
    "buying_behavior": "bb",
    "preferences_expectations": "pe",
    "satisfaction_future_intent": "sf",
}


def _prefix_for(tab: str, blueprint: dict | None = None) -> str:
    """Short id prefix for a section.

    # change for b2c questionarie — sections can now be named for the market
    # ("streaming_habits"), so derive the prefix from the section id and fall
    # back to the canonical section's prefix to keep ids stable and readable.
    """
    if tab in _TAB_PREFIX:
        return _TAB_PREFIX[tab]
    canon = narrative.section_canonical(blueprint, tab)
    if canon in _TAB_PREFIX:
        return _TAB_PREFIX[canon]
    words = [w for w in re.split(r"[^a-z0-9]+", (tab or "").lower()) if w]
    if len(words) >= 2:
        return (words[0][0] + words[1][0])[:2]
    return (tab or "sec")[:2]

# What each tab should cover (kept distinct to avoid cross-tab duplication).
# change for b2c questionarie — consulting-grade; NO age/income/brand-name items
#
# change for b2c questionarie -- CHECK (user directive, 2026-09-09): the
# framework is a FIXED 23-question instrument -- 5 / 7 / 6 / 5 across the four
# behavioural sections, plus Profiling (age + gender only, added separately in
# standard_sections.py). Each tab's guidance below names the EXACT question
# types that must fill that section's slots, in order.
#
# Two things make this different from ordinary prompt guidance:
#   1. It is GATED, not advisory. validator_critic.REQUIRED_SLOTS mirrors this
#      list slot-for-slot and fails the pass until every one exists, so a
#      missing type forces a regeneration instead of shipping.
#   2. Section boundaries are strict. A question belonging to another
#      section's remit is a defect even if it is a good question -- the slot
#      lists below are mutually exclusive by construction.
_TAB_GUIDANCE = {
    "consumer_profile": (
        "category relationship and usage context only — NOT age, income, or "
        "where they live, and NOTHING about buying (that is Section 2). "
        "Every question must be specific to the target product. This section "
        "has EXACTLY 5 slots and ALL 5 are required — fill them with these 5 "
        "question types, one each, in this order: "
        "(1) USAGE FREQUENCY — how often / how many days they use it; "
        "(2) PRIMARY USE CASE — the main job it does for them, what they "
        "mainly use it for; "
        "(3) OWNERSHIP TENURE — literally how LONG they have had or been "
        "using their current one (\"how long have you had...\"). This is NOT "
        "a count of how many they own; "
        "(4) USAGE CONTEXT — the physical places, situations or occasions "
        "where they use it (\"where do you usually...\", \"when do you...\"). "
        "Must NOT be phrased as \"what do you use it for\" — that is slot 2; "
        "(5) ITEM VARIANT — which format, size, type or version they "
        "currently use, so user types can be told apart."
    ),
    "buying_behavior": (
        "the purchase journey for the target product — NEVER ask which brand "
        "or company they buy (no manufacturer/name lists), and nothing about "
        "how the product performs (that is Section 3). This section has "
        "EXACTLY 7 slots and ALL 7 are required — fill them with these 7 "
        "question types, one each, in this order: "
        "(1) ACQUISITION TRIGGER — the specific event that made them realise "
        "they needed one; "
        "(2) SHOPPING TIMEFRAME — how long they spent researching and "
        "comparing before buying; "
        "(3) INFORMATION SOURCES — where they went to learn about this kind "
        "of product (reviews, demos, people they asked); "
        "(4) ACQUISITION CHANNEL — the kind of shop, site or platform where "
        "they actually bought it; "
        "(5) PRICE PAID — how much they spent, in realistic local-currency "
        "bands; "
        "(6) TOP DECISION DRIVER — the single factor that mattered most; "
        "(7) ALTERNATIVE CONSIDERATION SET — what other options or ways of "
        "solving the same problem they seriously considered first."
    ),
    "preferences_expectations": (
        "how the product actually performs for them — never named competitor "
        "brands or companies, and nothing about where or why they bought it "
        "(that is Section 2). This section has EXACTLY 6 slots and ALL 6 are "
        "required — fill them with these 6 question types, one each, in this "
        "order: "
        "(1) CORE FUNCTION SATISFACTION — how well it does its main job; "
        "(2) USABILITY / EASE OF USE — how easy or fiddly it is to use, set "
        "up or operate; "
        "(3) QUALITY SIGNAL — what tells them one is well made; "
        "(4) ESSENTIAL vs NON-ESSENTIAL FEATURES — which features they could "
        "not do without (multi-select); "
        "(5) VALUE FOR MONEY — whether what they got was worth what they "
        "paid; "
        "(6) EXPECTATION MATCH — how it performs compared with what they "
        "expected before buying."
    ),
    "satisfaction_future_intent": (
        "outcomes with the target product — no age/income/brand-name items. "
        "This section has EXACTLY 5 slots and ALL 5 are required, and this "
        "section has NO standard/anchor questions inserted automatically, so "
        "leaving any of these out is a real gap: "
        "(1) OVERALL SATISFACTION — how satisfied they are overall; "
        "(2) TOP PAIN POINT — the single biggest frustration or unmet need "
        "while using it; "
        "(3) SWITCHING TRIGGER — what would realistically make them switch "
        "away to a different kind of solution. This is DISTINCT from slot 2: "
        "slot 2 is the problem they have TODAY, slot 3 is the change that "
        "would make them LEAVE; "
        "(4) FUTURE ACQUISITION INTENT — how likely they are to buy the same "
        "kind of product again when they next need one. Ask likelihood, not "
        "what would make them buy sooner; "
        "(5) CATEGORY ADVOCACY — how likely they are to recommend this kind "
        "of product to someone else. This MUST be single_choice with EXACTLY "
        "11 options, one per whole number 0 to 10 in order (\"0 - Not at all "
        "likely\", \"1\", \"2\", ... \"9\", \"10 - Extremely likely\") — that "
        "exact 11-point shape is what makes it the standard 0-10 scale at "
        "publication; any other option count will NOT be recognised."
    ),
}


def _guidance_for(tab: str, beats: list | None, blueprint: dict | None) -> str:
    """What this section must cover.

    # change for b2c questionarie — a market-specific section carries its own
    # ``remit`` from the planner; fall back to the canonical section's standing
    # guidance so the banned-content rules always come through.
    """
    remit = ""
    for s in narrative.sections_for(blueprint):
        if s.get("section_id") == tab:
            remit = (s.get("remit") or "").strip()
            break
    canon = narrative.section_canonical(blueprint, tab)
    base = _TAB_GUIDANCE.get(tab) or _TAB_GUIDANCE.get(canon, "")
    if remit and base:
        return f"{remit} In research terms this section is the {canon} stage: {base}"
    return remit or base

# Residence / screener geography is chosen in the UI (By Region), never as a survey item.
_GEO_RESIDENCE_RE = re.compile(
    r"(?i)\b("
    r"which\s+region\s+do\s+you\s+(live|reside)|"
    r"in\s+which\s+region\s+do\s+you\s+(currently\s+)?(live|reside)|"
    r"where\s+do\s+you\s+(currently\s+)?(live|reside)|"
    r"what\s+(is\s+your|region\s+are\s+you\s+in)|"
    r"your\s+(current\s+)?(country|region)\s+of\s+residence|"
    r"where\s+are\s+you\s+(based|located)|"
    r"select\s+your\s+(region|country)"
    r")\b"
)


def _is_geo_residence_question(text: str) -> bool:
    """True for UI screener geography — not a consumer insight question."""
    return bool(_GEO_RESIDENCE_RE.search(text or ""))


# change for b2c questionarie — ban personal demographics asked as survey items
_AGE_INCOME_RE = re.compile(
    r"(?i)\b("
    r"(what|which).{0,40}\b(age|age\s*group|age\s*band|how\s+old)\b|"
    r"\b(your|household'?s?|personal|annual|monthly)\s+(total\s+)?(income|salary|earnings)\b|"
    r"\bincome\s+(range|bracket|band|level|group)\b|"
    r"\bhow\s+old\s+are\s+you\b|"
    r"\bage\s+group\b|"
    r"\bhousehold'?s?\s+total\s+(annual\s+)?income\b"
    r")"
)

# change for b2c questionarie — ban asking for brand / company names
_BRAND_COMPANY_RE = re.compile(
    r"(?i)\b("
    r"which\s+(brand|brands|company|companies|manufacturer|manufacturers|label|labels)\b|"
    r"what\s+(brand|brands|company|companies|manufacturer)\b|"
    r"(favorite|favourite|preferred|usual|current|main)\s+brand\b|"
    r"brand\s+(do\s+you|you\s+(buy|use|prefer|choose|purchase|trust))\b|"
    r"name\s+of\s+(the\s+)?(brand|company|manufacturer)\b|"
    r"from\s+which\s+(brand|company|manufacturer)\b|"
    r"which\s+of\s+these\s+(brands|companies|manufacturers)\b"
    r")"
)


def _is_age_or_income_question(text: str) -> bool:
    """True for age / household-income demographics — banned from the instrument."""
    # change for b2c questionarie
    return bool(_AGE_INCOME_RE.search(text or ""))


def _is_brand_name_question(text: str, options: list | None = None) -> bool:
    """True when the Q asks for a brand/company name (stem or option list)."""
    # change for b2c questionarie
    if _BRAND_COMPANY_RE.search(text or ""):
        return True
    if re.search(r"(?i)\b(select|choose|pick)\s+(your\s+)?(brand|company)\b", text or ""):
        return True
    if re.search(r"(?i)\b(brand|company|manufacturer)\b", text or "") and options:
        brandish = sum(
            1 for o in options
            if re.match(r"^[A-Z][A-Za-z0-9&\-'.]{1,24}$", str(o or "").strip())
        )
        if brandish >= 3:
            return True
    return False


# Generate a tab's quota in small chunks so DeepSeek's ~8k output cap can't
# truncate a large call into a short one (BUG 5). # change for b2c questionarie
# Chunks now follow the storyline: each call covers a contiguous run of beats.
QUESTION_CHUNK = 8
MAX_TAB_ATTEMPTS = 6  # total LLM calls per tab (quota mode refills)
MAX_CHUNK_ATTEMPTS = 2  # retries for one beat-chunk before moving on


def _norm_text(text: str) -> str:
    return "".join(ch for ch in (text or "").lower() if ch.isalnum())


# --- Storyline helpers (Phase A8) -------------------------------------------
# change for b2c questionarie


def _format_beats_block(
    allocations: list, *, free_choice: bool, total: int
) -> str:
    """Render the storyline brief the model must fill for this chunk."""
    lines = []
    for beat, n in allocations:
        head = f"  - beat_id: {beat['beat_id']}   \"{beat.get('title') or beat['beat_id']}\""
        if not free_choice:
            head += f"   → write EXACTLY {n} question(s) for this beat"
        lines.append(head)
        intent = (beat.get("intent") or "").strip()
        if intent:
            lines.append(f"        this beat must establish: {intent}")
    if free_choice:
        lines.append(
            f"\n  Choose where to spend this chunk: write {total} question(s) in "
            "TOTAL across the beats above — at most ONE per beat — placing them "
            "in the beats where this region's reality differs most from "
            "elsewhere. Set beat_id to the beat you chose."
        )
    return "\n".join(lines)


def _remaining_allocation(chunk: list, got: list) -> list:
    """Re-plan a chunk's per-beat quota from what a retry still owes."""
    have: dict = {}
    for q in got:
        have[q.get("beat_id")] = have.get(q.get("beat_id"), 0) + 1
    remaining = [
        (beat, n - have.get(beat["beat_id"], 0))
        for beat, n in chunk
        if n - have.get(beat["beat_id"], 0) > 0
    ]
    return remaining or list(chunk)


def _nearest_beat_id(raw: str, beats: list) -> str:
    """Repair a wrong/missing beat_id onto one of this chunk's beats."""
    key = _norm_text(raw)
    if key:
        for b in beats:
            if _norm_text(b["beat_id"]) == key:
                return b["beat_id"]
        raw_tokens = {t for t in re.findall(r"[a-z]+", (raw or "").lower()) if len(t) > 2}
        best, best_hits = None, 0
        for b in beats:
            cand = f"{b['beat_id']} {b.get('title') or ''}".lower()
            hits = sum(1 for t in raw_tokens if t in cand)
            if hits > best_hits:
                best, best_hits = b["beat_id"], hits
        if best:
            return best
    return beats[0]["beat_id"] if beats else ""


def _must_cover_block(blueprint: dict | None) -> str:
    """Topics this category cannot omit, for the architect prompt.

    # change for b2c questionarie — A12. Empty for a blueprint that declared
    # none, so the prompt is unchanged rather than carrying a stray heading.
    """
    topics = narrative.must_cover_topics(blueprint)
    if not topics:
        return ""
    listed = "\n".join(f"  - {t}" for t in topics)
    return (
        "\nTHIS CATEGORY CANNOT CREDIBLY OMIT:\n" + listed + "\n"
        "Somewhere across the whole survey each of these must be asked "
        "about — in a question stem or in an answer option. They are the "
        "issues a specialist would notice were missing. Cover the ones that "
        "belong in THIS section; the other sections cover the rest.\n"
    )


def _segments_block(blueprint: dict | None) -> str:
    """Render the behavioural segments the questions must discriminate."""
    segs = narrative.segments_for(blueprint)
    if not segs:
        return ("  (none defined — write questions that separate heavy from light "
                "users, price-led from quality-led, and loyal from switching.)")
    lines = []
    for i, sg in enumerate(segs, start=1):
        lines.append(f"  {i}. {sg.get('name')} [{sg.get('segment_id')}]")
        desc = (sg.get("description") or "").strip()
        if desc:
            lines.append(f"       {desc}")
    return "\n".join(lines)


def _avoid_block(prior_texts: list) -> str:
    """The 'never repeat these' block — absolute uniqueness is a hard rule."""
    if not prior_texts:
        return ""
    existing = "\n".join(f"    · {t}" for t in prior_texts[-80:])
    return (
        "- ALREADY ASKED — these questions already exist in this survey. Do NOT "
        "write any question that duplicates, paraphrases, narrows, broadens or "
        "otherwise collects the same information as ANY of them, in any section:\n"
        f"{existing}\n"
    )


# change for b2c questionarie — consulting funnel / tone / hygiene (Phase A2)
PROMPT_TEMPLATE = """\
You have spent twenty-five years writing consumer questionnaires, the last ten
running segmentation studies for clients in categories like this one. You are
writing the "{tab}" section of a study on {segment} ({region}). Today is {today}.

SEGMENT-NAME CASING: {segment} above may be Title Case, exactly as the study
author typed it (e.g. "Sport Shoes"). When you name the category mid-sentence
in a question or option, write it the way a person would say it out loud —
normal sentence casing ("sport shoes"), not the literal placeholder casing.
Capitalize it only where ordinary English capitalization already requires
that (start of a sentence, a genuine proper noun). Confirmed live: a survey
shipped with "Sport Shoes" (capitalized) reused mid-sentence throughout,
which reads as templated rather than written by a person — never repeat that.

WHAT TWENTY-FIVE YEARS SOUNDS LIKE — this is the single biggest difference
between a professional instrument and a generated one:

- YOU KNOW THE CATEGORY. Your questions name things that actually exist in this
  market and actually vary between buyers — real formats, real occasions, real
  frictions. A question that could be pasted into a survey about any other
  category is a wasted question.
    GENERIC: "How important is quality to you?"
    EXPERIENCED (milk): "Do you check the use-by date before putting milk in
                         the trolley?"
    EXPERIENCED (streaming): "When a show you follow ends, do you keep the
                              subscription running?"
- YOU ASK ABOUT BEHAVIOUR, NOT SELF-IMAGE. People describe themselves
  generously and report what they did accurately. Prefer the second.
    WEAK:   "How price-conscious are you when buying running shoes?"
    STRONG: "Did you wait for a discount before buying your last pair?"
- YOU ONLY ASK WHAT THE ANSWER WILL CHANGE. Before each question: if every
  respondent answered the same way, would the study be poorer? If not, cut it.
  This is not hypothetical — apply it hardest to OWNERSHIP-COUNT questions
  ("how many X do you own/have"). That question earns its slot ONLY when
  people genuinely vary — multiple pairs of shoes, several skincare products
  in a routine. For a product almost everyone owns exactly one of at a time
  (a phone, a pair of wireless earbuds, a refrigerator), "how many do you
  own" collapses to the same answer for nearly the whole sample and tells a
  brand manager nothing. Ask about REPLACEMENT or UPGRADE behaviour instead,
  which is where the real variation lives for a single-unit product.
    BAD (wireless earbuds — near-universal single ownership):
      "How many pairs of wireless earbuds do you currently own?"
    STRONG (same product, asks what actually varies):
      "How long have you had your current pair of wireless earbuds?"
      "How many pairs of wireless earbuds have you owned in total?"
  Before writing an ownership-count question, ask yourself: for THIS
  product, would answers actually spread across several different numbers,
  or would almost everyone say "one"? If the latter, don't ask it.
- STAY IN THE PRODUCT CATEGORY, NOT THE ACTIVITY OR LIFESTYLE AROUND IT. A
  question that could be pasted unchanged into a survey about a DIFFERENT
  category is off-category, even if it feels related.
    BAD (running shoes survey): "What is the main reason you run?" (train
      for races / stay fit / manage stress / socialize / save money on
      commuting) — this measures fitness/exercise motivation in general, not
      anything about the SHOES. It would work identically in a survey about
      running apps, gym memberships, or protein powder.
    STRONG (running shoes survey): tie the motivation to a shoe-relevant
      consequence — "training for a race" only belongs if the option set
      connects it to a performance-shoe need, not fitness in the abstract.
  Ask: could this exact question, with these exact options, appear unchanged
  in a survey for an unrelated category? If yes, cut it or reframe it around
  a consequence that is specific to THIS product.
- A MINOR HABIT IS NOT A QUESTION UNLESS IT TIES TO A DECISION. "How do you
  decide which pair to wear on a given day" measures a trivial daily-rotation
  habit with no connection to a purchase, positioning, or retention decision
  — apply the same "what business decision does this support" test the value
  test already requires, and cut or reframe toward something that IS
  purchase-relevant (e.g. how many pairs someone rotates BECAUSE of activity
  type, which is a genuine usage-breadth measure, not which pair they grab
  today).
- YOU DO NOT HEDGE OR PERFORM. No "we'd love to know", no exclamation marks, no
  jokes, no second-guessing the respondent. State the question and stop.
- ONE VOICE THROUGHOUT. Every question in this survey must read as though the
  same person wrote it in one sitting: same register, same directness, same
  level of formality. Do not slip into chatty phrasing for the easy questions
  and formal phrasing for the sensitive ones.

NO TWO QUESTIONS MAY SHARE A MEANING OR A SHAPE.
- MEANING: if two questions would be answered by the same fact about a person,
  they are one question. "What makes you keep a service" and "what would make
  you stay" are the same question. So are "where you compare options" and
  "where you first hear about them" if the option lists overlap.
- SHAPE: two questions may not share their option list, their opening three
  words, or their sentence pattern. Variety in construction is not decoration —
  it is what stops an instrument reading as machine-produced.

THE SECTION MUST TELL ONE CONTINUOUS STORY. Each question follows from the one
before it, the way a good interviewer moves. The section opens on the widest,
easiest question and narrows; behaviour comes before opinion; anything
sensitive (spend, cancelling, dissatisfaction) comes after the respondent is
warmed up. A reader should be able to see why each question sits where it does.

STUDY PURPOSE: {study_purpose}

SEGMENTS THIS STUDY MUST TELL APART:
{segments_block}
{must_cover_block}
Your questions are the instrument that separates those segments. A question that
every segment would answer the same way is a wasted question. Before writing
each one, ask: which segment does each answer option point to? If no option
discriminates, rewrite the question.

Tone: professional, plain, respondent-friendly. No jargon, no sales copy, no
leading or loaded phrasing. Every item must feel like something a consulting
firm would put in a client-ready consumer study — specific to {segment} and to
THIS section's remit, not a generic omnibus template.

QUESTION CRAFT — this is where most drafts go wrong. Read carefully.

1. CLEAR FIRST, SHORT SECOND. The respondent must understand the question on
   one reading, with no ambiguity about what or when is being asked. Say the
   whole thing — in plain words.
   TARGET: 8-16 words. Never more than 18.

   GOOD: "How many video streaming services do you currently pay for?"
   GOOD: "How many hours a week do you usually spend running?"
   GOOD: "How often do you buy a new pair of running shoes?"
   GOOD: "What made you choose the last pair of running shoes you bought?"

   BAD — bloated, hides the question inside a preamble:
        "In the last 30 days, which of the following best describes how you
         typically watch video streaming subscriptions?"
   BAD — so short it is ambiguous. What are "them"? Bought where, when?
        "How often do you buy them?"
        "How many do you have?"
   Both failures are equally wrong. Cutting words is not the goal; being
   understood on the first reading is.

   SIMPLICITY RULES:
   - Everyday words a 6-year-old would understand (see THE FINAL TEST later
     in this brief — that bar applies here too). No research vocabulary in
     "attributes", "drivers", "criteria", "consumption occasion", "purchase
     journey", "trade-off". Those words belong in the section heading, never
     in what a respondent reads.
   - SELF-CONTAINED. Name the product in the stem — a question must make sense
     on its own, without the section heading above it. Never "them", "it" or
     "this product" where the category name would do.
   - Give the timeframe when it changes the answer, but say it the plain way,
     never "in a typical week/day/month" — that phrasing is BANNED, it is
     stiff survey-speak, not how a person talks:
       BANNED: "In a typical week, how many days do you use X?"
       PLAIN:  "How many days a week do you use X?"
       BANNED: "In a typical month, how often do you buy X?"
       PLAIN:  "How often do you buy X each month?"
     Leave the timeframe out entirely when it does not change the answer.
   - ONE clause and one idea. If your stem needs a comma to survive, cut it down.
   - BANNED OPENERS. "Thinking about...", "Which of the following best
     describes...", and "Which of these best describes..." are the same padded
     formula. A stem built on them takes five words to start asking. Say the
     thing instead:
       BAD:  "Which of these best describes where you usually watch streaming?"
       GOOD: "Where do you usually watch streaming services?"
       BAD:  "Which of these best describes how you pay for your subscriptions?"
       GOOD: "Who pays for your streaming subscriptions?"
       BAD:  "Which best describes your household?"
       GOOD: "Who else lives in your household?"
   - VARY THE CONSTRUCTION. Across the whole survey, no more than TWO questions
     may open with the same three words. Rotate: How many / How often / How much
     / What / Which / Where / When / Who / Why / How likely. A questionnaire
     where seven questions start identically reads as generated, not written.
   - Start with the question word wherever you can.

7. ANSWER OPTIONS MUST BE MUTUALLY EXCLUSIVE ON A SINGLE PICK.
   Never offer nested thresholds on a single_choice question — "$5 or more",
   "$10 or more", "$15 or more" are all true at once for the same person, so
   there is no correct answer. Use disjoint bands ("$10-$14", "$15-$19",
   "$20-$29") or ask directly for the cut-off point.

2. A STANDARD, PREDICTABLE SHAPE. The survey must read like one instrument
   written by one person, asked in a fixed order:
   - The first question of a section is the widest and easiest one there.
   - Behaviour first (what they do), then choice (how they pick), then opinion
     (how they feel). Never the reverse.

   ONE TOPIC AT A TIME — THE READER MUST NEVER FEEL A JUMP.
   Consecutive questions must continue the same thought. A respondent should be
   able to say "we are still talking about X" until the section clearly moves
   on. Never interleave: do not ask about price, jump to fit, then return to
   price. Finish a topic completely, then move to the next.
   Concretely, within a section:
   - Group every question on the same topic together, back to back.
   - Order the groups so each one follows naturally from the last (how they
     shop → where they buy → what they pay).
   - The question you write for a beat MUST be about that beat's subject. If
     your question is about buying intent, it does not belong on a
     satisfaction beat — pick the beat that actually matches, or write a
     different question for the beat you were given.
   Read your questions back in order before returning them. If any question
   would make a respondent think "why are we asking this now?", move it.

   NEVER DRILL INTO ONE OPTION OF A QUESTION YOU JUST ASKED. If a question
   already offers a set of choices, do not follow it with a question about one
   of those choices — that answer is already captured, and it derails the flow
   from the general to a single detail and back out again.
   BAD:  Q2 "Where do you usually run?" (roads / trails / treadmill / track)
         Q3 "How often did you run on a treadmill?"   <- drills into one option
   GOOD: Q2 "Where do you usually run?"
         Q3 "What is the main reason you run?"        <- next distinct topic
   The section has very few slots. Spend each one on a NEW dimension of the
   story, never on a sub-question of the previous slot.
   - Keep option counts consistent: 4-6 options is the norm. Only exceed it when
     the list is genuinely longer (channels, features).
   - Reuse the same scale wording across the survey. If one satisfaction
     question uses "Very dissatisfied ... Very satisfied", they all must.

3. THE OPTIONS MUST ANSWER THE STEM. This is the most common failure.
   If the stem asks HOW MANY, the options are counts. If it asks HOW OFTEN, the
   options are frequencies. If it asks WHICH, the options are things.
   GOOD: "How many services do you pay for?" -> 1 / 2 / 3 / 4 / 5 or more
   GOOD: "How much per month?" -> Under $10 / $10-24 / $25-49 / $50-74 / $75+
   BAD:  "How often do you buy?" -> "I care about quality" / "Price matters"
   Read your stem and your options back to back. If the option is not a direct
   answer to the words in the stem, the question is broken.

4. NUMERIC BANDS, NOT VAGUE WORDS, wherever a number exists AND the respondent
   can actually know it. Bands must not overlap and must cover the full range.

   BUT NEVER ASK FOR A TOTAL ACROSS A LOOK-BACK WINDOW. "In the last 30 days,
   how many times did you…" / "how much did you spend in total last month" are
   unanswerable — nobody tallies their own workouts, orders or app spend, so
   every reply is a guess and the resulting percentages are noise dressed as data.
   BAD:  "In the last 30 days, how many times did you wear athletic apparel?"
   GOOD: "How often do you wear athletic apparel for a workout?"
         -> Always or almost always / Often / Sometimes / Rarely / Never
   BAD:  "How much did your household spend on meal kits in the last 30 days?"
   GOOD: "Which best describes what you usually spend on a single meal-kit order?"
         -> Under $40 / $40-59 / $60-79 / $80-99 / $100 or more
   A count is fine when the number is small, current and standing — "How many
   streaming services do you currently pay for?" is a fact someone holds in
   mind. The test: could the respondent answer confidently without doing
   arithmetic or consulting a receipt? If not, ask relative frequency, a typical
   amount, or a current state.

5. ONE JOB PER QUESTION. If you need both behaviour and attitude, write two.

6. USE THE RIGHT FORMAT:
   - single_choice for mutually exclusive facts (counts, bands, one pattern)
   - multiple_choice ONLY for genuine select-all (channels used, features owned)
   - likert_5 for agreement or satisfaction on ONE named thing
   - NEVER use ranking. This backend cannot produce a genuine ranking
     calculation (mean rank position, or "% who ranked this #1") — it can
     only simulate a plain percentage share, which is indistinguishable from
     a single_choice result and misleads the reader into thinking a real
     rank order was measured (CHECK 13/14, B2C master rules). For relative
     importance across several items, ask single_choice instead:
     "Which factor has the biggest influence on your final choice?"

   Match the format to how people actually answer. If a respondent would
   genuinely pick several, it is multiple_choice — forcing one pick throws away
   the other answers and biases the winner.
   - "Which are your MUST-HAVES" -> multiple_choice. Must-haves are plural.
   - "What would you give up / trade off" -> single_choice asking which ONE
     thing matters most, not ranking — a trade-off is a comparison, but this
     backend can only display it as a single biggest-influence pick.
   - "Which reasons apply" / "what have you experienced" -> multiple_choice.
   Never mix dimensions inside one option list: every option must answer the
   same question. "I rotate every few months" (frequency) and "I cancel only
   when unhappy" (reason) cannot sit in the same list — a respondent for whom
   both are true has no valid answer.

8. NEVER OFFER A "NOT A USER" OPTION. Everyone in this sample already qualifies
   as a category buyer or user — that is settled by the audience definition
   before anyone is asked a question. An option like "None", "I don't subscribe
   to any", "I don't use it" or "N/A" describes a person who is not in the
   sample, so any percentage against it is impossible data.
   - "Other" IS allowed and expected on categorical lists.
   - "None of these are important to me" IS allowed — that is an opinion, not a
     claim of non-usership.
   - "None" / "I don't buy this" is NOT allowed.

This tab should cover: {guidance}

THE SLOT LIST ABOVE IS CLOSED — THESE THEMES AND NOTHING ELSE.
The numbered question types in the guidance above are not a starting point
or a set of suggestions to build on. They are the COMPLETE and EXCLUSIVE
list of what this section may contain:
- Write exactly one question per numbered slot. No more, no fewer.
- Do NOT add a question on any other theme, however good or interesting it
  is. A well-written question on an unlisted theme is still WRONG and will
  be rejected — an unlisted theme is out of scope by definition, not a
  bonus.
- Do NOT write two questions for the same slot in different words. Two
  questions a respondent answers the same way are one question, and the
  second one has taken a slot that its own theme needed.
- Do NOT borrow a theme from another section. Each theme belongs to exactly
  one section; if an idea fits a slot listed under a different section, it
  is that section's job, not yours — leave it out here.
The check applied afterwards is mechanical: every listed theme must be
present, and nothing outside the list may appear. Anything else fails and
is regenerated.

EVERY QUESTION MUST STILL PASS THE 6-YEAR-OLD WORDING TEST.
Getting the theme right is only half the job. Each question must also be
written so plainly that a 6-year-old would understand every single WORD in
it — even though the topic (money, buying, brands) is clearly adult. Short
sentence, everyday words, no research or marketing vocabulary, no stiff
survey phrasing. If a word would make a child ask "what does that mean?",
replace it with the ordinary word an adult would use talking to a friend.

STORYLINE — this survey tells one continuous story, and this chunk fills
specific BEATS of it. The beats below are already in narrative order:

{beats_block}

STORYLINE RULES (never violate):
- Every question MUST set `beat_id` to EXACTLY one of the beat ids listed above.
  Copy the id character for character. A question with a wrong or missing
  beat_id lands in the wrong place in the survey.
- Write the stated number of questions for each beat. Do not merge two beats
  into one question and do not spend a beat's quota on another beat's topic.
- A question must serve ITS beat's intent and nothing else. If an idea belongs
  to a different beat, drop it — that beat has its own quota.
- `funnel_position` orders questions WITHIN a single beat only: 1 = the
  broadest/easiest/safest of that beat's questions, increasing for narrower or
  more sensitive ones. Do NOT use it to order across beats — beat order already
  does that, and cross-beat values are ignored.
- Never open a beat with age, income, brand names, or geography of residence
  (all banned below).

WHICH QUESTIONS BELONG IN THIS SURVEY.
Every question is asked in ORDINARY CONSUMER LANGUAGE — you are talking to a
shopper about the product, never to a business. But a question only earns its
slot if the answer tells someone something USEFUL about the product: what
people buy, why they picked it, what they pay, where they get it, what they
like, what annoys them, and what would make them change.

  ASK THINGS LIKE THIS — plain questions about the product itself:
    "What made you pick the one you bought?"
    "What would make you switch to a different one?"
    "How much do you usually pay for one?"
    "Where do you usually buy them?"
    "What annoys you most about the one you have?"
    "What would you give up to pay less?"
    "Which features do you actually use?"

  DO NOT ASK THINGS LIKE THIS — the answer tells nobody anything about the
  product, so the slot is wasted:
    "Who in your household decides which one to buy?"  <- household admin, not
        the product; and this respondent is already the buyer, so it is settled
    "What size is the screen, in mm?"                  <- a spec most owners
        cannot answer, and knowing it changes nothing
    "When do you put it on in the morning?"            <- a daily habit that
        says nothing about how the product was chosen or how it performs
    "Do you own one?"                                  <- already settled by
        who is being surveyed

The test is simple: does the answer say something about THE PRODUCT — how it is
chosen, paid for, used, judged, or replaced? If it is really about the
respondent's household arrangements, daily routine, or a spec they would have
to look up, cut it.

Prefer questions about a REAL PAST DECISION ("the one you bought last") over
opinions or guesses about the future — what someone actually did is far more
reliable than what they imagine they would do.

HOW A QUESTION AND ITS ANSWERS MUST BE PRESENTED (never violate):
- ONE IDEA PER QUESTION. Name in your head the single thing it measures —
  usage frequency, price sensitivity, satisfaction, switching trigger. If you
  cannot name it in two words, the question is unfocused. Published instruments
  map every item to exactly one construct; that is what makes them readable.
- NEVER present the same idea twice in different words. "What would you give up
  to get a lower price?" and "If it cannot have everything, what would you drop
  first?" are ONE question — a respondent answers them identically.
- ANSWER OPTIONS ARE A SET, NOT A LIST. Every option in one question must be
  the same kind of thing, at the same level of detail, and phrased in the same
  grammatical form. Do not mix a two-word option with a full sentence.
- ORDER THE OPTIONS THE WAY THEY ARE READ. Ordinal answers run low → high
  ("0 days" → "7 days", "Under €150" → "€600 or more"); never shuffle them.
  Unordered answers put the escape option ("Other", "None of these") LAST.
- ONE CURRENCY PER INSTRUMENT, AND IT IS {currency}. Every price, spend and
  income option in this survey is written in {currency} — never US dollars
  unless {currency} IS the dollar. Price BANDS must be realistic for this
  market too, not a converted US ladder: what a shopper in this market
  actually pays. Getting the symbol right but the amounts wrong is still wrong.

QUESTION TYPE MUST BE OBVIOUS FROM THE WORDING ALONE (Issue 9d):
If two questions share a near-identical sentence template but one is a single
pick and the other is select-all, a reader cannot tell them apart from the
text -- only from a percentage total or hidden metadata. Never let this
happen. If a multi-select question would otherwise read like its single-select
neighbour, change its wording so the type is visible in the sentence itself:
  BAD (identical shape, different types, no textual cue):
    "Which must a skincare product DO for you to keep using it?"   (single)
    "Which must a skincare product BE for you to consider buying?" (multi)
  FIXED: keep the single-select version as-is, and rewrite the multi-select
  one so "select all" is part of the sentence:
    "Which of these matter to you when you decide to buy? Select all that
     apply."
  NEVER punctuate it as "...? Select all that apply.?" -- write ONE question
  mark, at the very end, never two, and never a "." followed by "?".

PLAIN LANGUAGE — WRITE THE WAY PEOPLE TALK (never violate):
- EVERY question must be about {segment} — the product itself. Ask what they
  buy, use, look for, pay, like, dislike and would change ABOUT THE PRODUCT.
  Not lifestyle, not identity, not attitudes to life in general.
- ACTIVE VOICE, SECOND PERSON, PRESENT TENSE. "How often do you buy X?" — not
  "How frequently is X purchased by you?" or "X is bought how often?"
- Start with a plain question word: What / Which / How / Where / When / Who /
  Do / Have. A question that opens with a clause is already too complicated.
- SHORT. Aim for 8-14 words; never exceed 20. If a stem needs a comma to hold
  it together, it is two questions or it is over-written.
- Words a 12-year-old knows. No research or marketing vocabulary in anything
  the respondent reads:
    BANNED: purchase journey, decision driver, touchpoint, consideration set,
    value proposition, pain point, attribute, criteria, factor, dimension,
    segment, demographic, utilise, leverage, engagement, ecosystem, omnichannel
    SAY INSTEAD: buy, choose, what matters, problem, feature, use
- No hedging or throat-clearing: drop "In your opinion", "Generally speaking",
  "Thinking about your overall experience,". Ask the thing.
  BAD:  "Thinking about your most recent purchase occasion, which attributes
         were most influential in your decision-making process?"
  GOOD: "What made you pick the shoes you bought last?"
  BAD:  "How would you characterise the frequency of your usage?"
  GOOD: "How often do you wear them?"
- KEEP CONDITIONAL / TRADE-OFF SETUP CLAUSES SHORT. An "if" clause exists only
  to set up the trade-off -- it must not run longer than that. Cut it to the
  minimum words needed and put the real question first where possible.
  BAD:  "If a skincare product could not have everything, which would you
         give up first to pay less?"
  GOOD: "If you had to cut one thing to pay less, what would you give up
         first?"
  BAD:  "If a product delivers visible results but has a heavy or greasy
         texture, how likely are you to keep using it?"
  GOOD: "How likely are you to keep using a product that works well but
         feels heavy or greasy?"
- CUT FILLER THAT ADDS NO MEANING: "for yourself", "personally", "in your
  opinion", "for you" tacked onto a question that does not need it.
- EVERY OPTION MUST BE SOMETHING A REAL PERSON WOULD SAY OUT LOUD -- a
  complete natural phrase, never a clipped internal label.

- READ EVERY QUESTION ALOUD BEFORE YOU KEEP IT. If it does not sound like
  something one person would say to another, rewrite it. It must be a complete,
  grammatical English sentence — no missing words, no dropped nouns.
  NEVER start a question "Which must a…" / "Which should a…" / "Which does a…"
  — a word is missing every time. Write "Which of these must a X do…" instead.
  BROKEN: "Which must a smartwatch do for you to consider buying it?"
  FIXED:  "Which of these must a smartwatch do before you would buy one?"
  BROKEN: "What was the most important when you chose your smartwatch?"
  FIXED:  "What mattered most when you chose your smartwatch?"
  A missing word is the single most visible defect in a client deliverable.

- TALK TO A PERSON, NOT A RESPONDENT. Say "you" and "your". Use the everyday
  word for the product, the one a shopper would use in a shop.
  STIFF: "For which purposes is the device utilised?"
  HUMAN: "What do you use it for?"
  STIFF: "Indicate your preferred retail channel."
  HUMAN: "Where do you usually buy them?"

HYGIENE RULES (never violate):
- ONE idea per question — never double-barreled ("price and service", "quality
  and value"). Split into separate questions.
- NEUTRAL wording — prefer "How would you rate…", "In the last 30 days, how
  often…". Never "Wouldn't you agree…", "Don't you think…", or guilt/praise frames.
- MECE options — mutually exclusive, collectively exhaustive; numeric ranges
  must not overlap. For categorical single_choice lists, include exactly one
  escape hatch ('Other' / 'None of the above' / 'N/A') where the set is not an
  inherently complete scale (frequency, likert).
- BALANCED scales — likert_5 / likert_7 must be symmetric around a neutral
  midpoint, equal steps on each side, consistent polarity direction
  (e.g. Strongly disagree → Strongly agree). Use the same direction across
  the survey.
- ONE POLARITY FOR THE WHOLE SURVEY — every ordinal scale must run
  NEGATIVE → POSITIVE (worst option first, best option last), whatever the
  question type. This applies to scales you write as single_choice, not just
  likert_5/likert_7. If satisfaction runs "Very dissatisfied … Very satisfied"
  then likelihood must run "Very unlikely … Very likely" and importance must
  run "Not at all important … Extremely important". A survey that flips
  direction midway makes respondents mis-click and makes two sections
  non-comparable.
- USE THESE EXACT SCALE LABELS. Consumer-research standards — do not invent
  variants, and reuse the same wording every time the construct recurs:
    Frequency:    Always or almost always / Often / Sometimes / Rarely / Never
    Satisfaction: Very dissatisfied / Somewhat dissatisfied /
                  Neither satisfied nor dissatisfied / Somewhat satisfied /
                  Very satisfied
    Likelihood:   Very unlikely / Somewhat unlikely /
                  Neither likely nor unlikely / Somewhat likely / Very likely
    Importance:   Not at all important / Slightly important /
                  Moderately important / Very important / Extremely important
    Purchase intent: Definitely will not buy / Probably will not buy /
                  Might or might not buy / Probably will buy / Definitely will buy
  PREFER A CONSTRUCT-SPECIFIC SCALE OVER AGREE/DISAGREE. Instead of
  "I worry about ingredient safety" → Strongly disagree…Strongly agree, write
  "How concerned are you about ingredient safety?" → Not at all concerned /
  Not too concerned / Somewhat concerned / Very concerned. Agreement scales
  invite people to just say yes, so they measure agreeableness, not the topic.
- NO BROKEN MATRIX/GRID STEMS — never write "How important are each of the
  following…" / "rate each of the following…" unless the OPTIONS themselves
  ARE the attribute list (Price, Comfort, Durability…). If you need ratings per
  factor, write ONE likert_5 question PER factor (e.g. "How important is
  Comfort when buying {segment}?") with a standard importance scale as options.
  Never pair an "each of the following" stem with only a likert scale and no
  factors listed.
- CONCRETE timeframes — prefer "In the last 30 days…" / "In the past 12 months…"
  over vague "Generally…" / "Usually…".
- HARD UNIQUENESS — no two questions may collect the same or highly similar
  information (same construct, paraphrase, or overlapping options). If unsure,
  drop one. Cross-tab near-duplicates are forbidden.

BANNED CONTENT (never violate — drop the idea entirely):
- NEVER ask age, age group, how old, household income, salary, or earnings.
  Demographics belong in the target-customer definition, NOT as survey items.
- NEVER ask which brand / company / manufacturer the respondent buys, uses, or
  prefers. Do NOT list brand or company names as options. You may ask about
  category-level cues (e.g. "private label vs premium", "origin claim",
  "certified organic") without naming firms.
- NEVER ask where the respondent lives / resides / is based (UI handles region).

CRITICAL — SUBJECT RULE (never violate):
Every question's TEXT and OPTIONS must be about {segment} — the EXACT target
product. NEVER write a question about a different product category. Some evidence
below comes from RELATED/adjacent markets (see source_market) — use it ONLY to
learn which TOPICS can carry a real number later; it must NEVER change what the
question ASKS about. E.g. if the target is organic (dairy) milk, do NOT ask about
plant-based / oat / almond / soy / coconut milk — ask about organic dairy milk.

REGION / GEOGRAPHY RULE (never violate):
- NEVER ask where the respondent lives / resides / is based (e.g. "In which
  region do you currently live?", "Which region do you live in?", "Where are
  you located?"). The geography is already fixed for this file — every
  respondent is in it — so asking wastes a question and can only contradict.
- Do NOT use the platform regions ({regions}) as answer options for a
  residence screener, and never ask the respondent to name their country.
- Product origin / "made in" / market focus about the PRODUCT are OK if they
  are not asking about the respondent's home and do not list company names.
- You MAY write questions grounded in THIS market's local reality — the shops
  and channels people actually use there, prices in the local currency, local
  formats and pack sizes, seasons and climate. Naming a local retail channel or
  quoting a price in local currency is describing the market, not asking where
  someone lives, and it is what makes a country file worth reading.

GROUNDABLE FIGURES (optional topic inspiration only — percentages are NOT filled here):
{figures_json}

CONSUMER VOICE / PAIN LANGUAGE (from MKP + Reddit synthesis — LIFT option labels):
{voice_json}

PERSONA COVERAGE (ensure questions can discriminate these archetypes — do NOT
ask age/income/brand-name items even if personas mention them):
{persona_json}

Produce {count} closed-ended questions for this tab. QUALITY AND UNIQUENESS
COME FIRST — design what a top consumer-insights firm would ship.

{count} IS A CEILING, NOT A QUOTA. Write as many REAL questions as this section
genuinely needs, up to {count}. Stopping at 6 or 7 strong questions is a better
answer than {count} where the last one or two are padding.

NEVER INVENT A QUESTION TO REACH THE COUNT. If you find yourself rephrasing
something you already asked in order to fill a slot, stop and return fewer.
These are all the SAME question and only one may appear:
  "What matters most to you in a X?"
  "What was the most important thing when you chose your X?"
  "Which of these must a X do for you to buy it?"
  "How do you judge the quality of a X before buying?"
  "What most often disappoints you about your X?"
A reader who meets all five knows the survey was padded. Pick the single best
framing for this section and spend the other slots on genuinely new ground —
or leave them empty.

Every slot is expensive: each question must earn its place by separating the
segments or changing what a brand manager would do. If a question is merely
interesting, or its answer is already implied by another question, it does not
go in.

DO NOT ASK THE SAME CATEGORY LIST TWICE UNDER A DIFFERENT STEM. "Which products
have you used?" and "Which formats are you willing to use?" read as different
questions, but if their option lists are both a rundown of the same product
types (cleanser, serum, mask...), a respondent is answering the same list
twice. Once you have asked about a category's product-type breakdown, do not
ask it again from another angle in this survey.

DO NOT MIX ANSWER DIMENSIONS IN ONE OPTION LIST. Every option in a single list
must answer the same kind of question. "Only in the morning" (a time) and
"only when my skin feels dry" (a trigger) are not comparable answers to "when
do you apply it?" — pick one dimension and ask only that.
Before you write a "when do you..." question, decide ONE of these two framings
and use ONLY that framing's option style — never both in the same list:
  TIME-OF-DAY framing: "When do you usually use it?"
    -> Morning / Evening / Both morning and evening / No set time
  TRIGGER/OCCASION framing: "When are you most likely to use it?"
    -> As part of my morning routine / As part of my evening routine /
       After showering or bathing / When my skin feels dry or uncomfortable /
       Before going out or for special occasions
Never combine a time option ("morning") with a trigger option ("when it feels
dry") in the same list — that is the single most common mixed-dimension defect
and it must not recur.

NEVER LEAD A QUESTION WITH A TIMEFRAME CLAUSE UNLESS CHANGE OVER TIME IS THE
ACTUAL CONSTRUCT. "In the past year, how often have you used X?" and "In the
last 12 months, what disappointed you?" both bolt on a needless timeframe in
front of a question about a standing habit — delete the clause and ask the
habit directly: "How often do you usually use X?" / "What most often
disappoints you about X?"
A timeframe belongs ONLY when the question is genuinely about change over
time: "Compared to a year ago, how much are you now spending?" is legitimate
because the comparison IS the question. If removing the timeframe changes
nothing about what is being asked, the timeframe is filler — remove it.

NEVER OFFER AN OPTION THAT CONTRADICTS THE AUDIENCE DEFINITION. If the
respondent is defined as buying this product for themselves, an option like
"for children in my household only" describes someone outside the study
population — that person was already excluded, so the option can never be
chosen honestly.
1) Every question asks ONE clear, specific thing (never double-barreled), in
   plain consumer language, and would be directly useful to a brand manager
   acting on the answer for {segment}.
   PREFER SHORT, CONVERSATIONAL PHRASING over formal survey-style
   constructions — this applies to EVERY category and EVERY generation, not
   one client's file. Confirmed by manager review across a full 28-question
   instrument (China Sport Shoes) — every stiff/formal question in that
   study had a shorter, plainer equivalent that asked the exact same thing.
   Apply these conversions to every question you write, in any category:

   a) KILL "WHICH ACTIVITY / WHICH BEST DESCRIBES / WHICH SINGLE FACTOR /
      WHICH OF THESE" OPENERS. Say the thing directly instead:
        STIFF: "Which activity do you use {segment} for most?"
        PLAIN: "What do you mainly use {segment} for?"
        STIFF: "Which single factor has the biggest influence on your
                final choice?"
        PLAIN: "What matters most when choosing {segment}?"
        STIFF: "Which of these must {segment} have for you to consider
                buying it?"
        PLAIN: "What features must {segment} have for you to buy them?"

   b) COLLAPSE "HOW [ADJECTIVE] IS IT THAT... / HOW IMPORTANT IS..." INTO A
      DIRECT YES/NO OR PLAIN QUESTION where a scale isn't the actual point:
        STIFF: "How important is it that one pair works for more than one
                activity?"
        PLAIN: "Do you prefer {segment} that can be used for different
                activities?"
        STIFF: "How attached do you feel to a particular brand for
                {segment}?"
        PLAIN: "Do you usually stick to the same brand for {segment}?"

   c) REWRITE NPS/LIKERT-STYLE STEMS AS A PLAIN SENTENCE A PERSON WOULD SAY.
      "How likely are you to X" and "How interested would you be in X"
      almost always compress to "Would you X":
        STIFF: "How likely are you to recommend X to a friend or family
                member?"
        PLAIN: "Would you recommend X to others?"
        STIFF: "How interested would you be in a subscription that sends
                you a new pair every few months?"
        PLAIN: "Would you be interested in getting {segment} through a
                subscription?"
        STIFF: "How well do your {segment} usually perform compared to
                what you expected when you bought them?"
        PLAIN: "Do your {segment} usually perform as well as you
                expected?"

   d) CUT HEDGE WORDS THAT ADD NO MEANING: "most recent", "most likely",
      "most encourage you to... sooner rather than later" all pad the
      sentence without changing what is being asked. Say it plainly:
        STIFF: "What made you buy your most recent pair of {segment}?"
        PLAIN: "Why did you buy your last pair of {segment}?"
        STIFF: "What would most likely make you switch to a different
                {segment}?"
        PLAIN: "What would make you switch to different {segment}?"
        STIFF: "What would most encourage you to buy your next pair
                sooner rather than later?"
        PLAIN: "What would make you buy your next pair sooner?"
        STIFF: "How much did you pay for your most recent pair?"
        PLAIN: "How much did you spend on your last pair of {segment}?"

   e) REWRITE "IF YOU HAD A PROBLEM WITH X, WHAT WOULD YOU MOST LIKELY DO"
      STYLE CONDITIONALS as a direct question, condition first, question
      second, no hedge word:
        STIFF: "If you had a problem with {segment}, what would you most
                likely do?"
        PLAIN: "What would you do if you had a problem with {segment}?"

   f) NEVER RESTATE THE OBJECT A SECOND TIME WHEN "THEY/IT" IS UNAMBIGUOUS
      within the same short sentence — but ALWAYS name the category once:
        STIFF: "Where do you usually see {segment} options before you buy?"
        PLAIN: "Where do you usually look for {segment} before buying?"
        STIFF: "How long do you think a good pair of {segment} should
                last before wearing out?"
        PLAIN: "How long should a good pair of {segment} last?"

   THE FINAL TEST — WORDING SIMPLE ENOUGH FOR A 6-YEAR-OLD TO UNDERSTAND EVERY
   WORD, EVEN THOUGH THE TOPIC IS FOR ADULTS. This is not about dumbing down
   the QUESTION — a 6-year-old is not the respondent, and the topic (prices,
   brands, purchase decisions) stays fully adult. It is about the WORDS you
   use to ask it: no word in the sentence should need a dictionary, and the
   sentence should have no more moving parts than a child's bedtime-story
   sentence. If you had to read the question OUT LOUD to a 6-year-old and
   have them understand every individual word (not the adult topic, just the
   words), would they? If any word forces you to imagine explaining it first
   ("attribute", "criteria", "consideration", "most likely", "which of
   these"), that word fails the test — replace it with the word a child
   already knows: use, buy, pay, like, most, why, when, where, pick, give up.
     FAILS THE TEST: "Which single factor has the biggest influence on your
       final choice?" — "factor" and "influence" are not words a 6-year-old
       uses.
     PASSES THE TEST: "What matters most when choosing {segment}?" — every
       word here is one a small child already knows, even though the
       QUESTION is clearly for an adult buyer.
   Read every question back with this test before finalizing. Short, plain,
   one sentence, no hedge words, no "which of these/which best describes/
   most likely/most recent" scaffolding.
2) ABSOLUTE UNIQUENESS — no question may repeat, paraphrase, or overlap ANY
   other question in the entire survey. Not within this beat, not across beats,
   not across the four sections. Two questions that a respondent would answer
   the same way are the SAME question, however differently worded. This
   includes questions listed as already-created below. If in doubt, write a
   different question — a missing question is recoverable, a repeated one is not.
3) OPTIONS carry the quality: 4-7 concrete, mutually exclusive, collectively
   exhaustive choices in real consumer wording (lift specific labels from the
   CONSUMER VOICE evidence); numeric ranges must not overlap; at most ONE
   'Other'/'None' escape option where relevant — no vague catch-all stacks; no
   recycled generic templates (Brand/Price/Quality) across questions; NO named
   companies or brands as options.
4) Evidence alignment is OPPORTUNISTIC, not required: where a strong question
   NATURALLY matches an available figure, use the figure's real category values
   as options and set evidence_aligned=True + serves_topic. NEVER contort a
   question or degrade its options just to match evidence — a better question
   with inferred numbers beats a worse question with grounded ones.
5) Set `beat_id` (copied exactly from the storyline list) and `funnel_position`
   (order within that beat, integer ≥ 1) on EVERY question.

Rules for every question:
- Closed options only — NO open-ended. single_choice options must be mutually
  exclusive AND collectively exhaustive (add an escape option 'Other'/'None'/
  'N/A' where relevant). multiple_choice options are independent.
- 'type' is the ANSWER FORMAT — one of EXACTLY: single_choice, multiple_choice,
  likert_5, likert_7. Never put a chart name here. Never use 'ranking' — see
  the format rule above.
- 'chart_type' is how to VISUALIZE it — one of EXACTLY: bar, horizontal_bar,
  radar, donut, stacked. VARIETY REQUIRED across this tab: do not assign the
  same chart to every question. Guidance:
    · bar — single_choice frequency tallies
    · horizontal_bar — multiple_choice factor lists (or long labels)
    · donut — share/composition (3–6 mutually exclusive options)
    · radar — attribute-importance across ~5–7 dims
    · stacked — part-to-whole / mix of shares in one bar
  likert_5/likert_7 will be rendered as likert bars automatically.
  'type' and 'chart_type' are different fields — do not mix.
- Plain, jargon-free wording. Every question must be UNIQUE — no duplicate or
  paraphrased versions of any other question in the survey — and must clearly
  belong to THIS tab's remit (do not ask another tab's topic here).
PERSONALIZE the fill questions (evidence_aligned=False) using the CONSUMER VOICE:
- Lift SPECIFIC option labels from the real complaints/pain points/language above
  (e.g. "Sunscreen white cast", "Retinoid purge", "Fragrance irritation") — NOT
  generic templates ("Skin reactions", or Brand/Price/Ingredients/Packaging).
  Prefer real, specific wording; always phrased about {segment}. Never lift a
  brand or company name into options — this includes a parenthetical "(e.g.,
  X, Y)" example tacked onto an otherwise generic option. "Online marketplace
  (e.g., Shopee, Lazada)" is still a brand-name violation even though
  "Online marketplace" alone is fine — the example itself names real
  companies. A generic option must stay generic all the way through with NO
  named example, in every market: "Online marketplace", "Sports chain
  retailer", "Specialty store".
- When a question's OPTIONS are drawn from that evidence, set
  options_from_evidence=True (even if the answer percentages will be inferred).
  Options-from-evidence and answer-grounding are INDEPENDENT — do NOT fake
  grounding just because the options came from evidence.
{mode_line}{feedback_line}

Examples:
- evidence_aligned single_choice "Where do you usually buy {segment}?" with
  channel options (Online specialist retailer, Brand website, Department store,
  Mass retailer, Other) — NOT continent/region-of-residence options.
- PERSONALIZED (options_from_evidence=True) multiple_choice "Which of these
  frustrations have you had with {segment}?" with options lifted from forum
  complaints ("Battery dies too fast", "Strap irritation", "App sync failures",
  "Other"), chart=horizontal_bar.
- likert_5 "Overall, how satisfied are you with {segment}?" with
  Very dissatisfied / Somewhat dissatisfied / Neither / Somewhat satisfied /
  Very satisfied (balanced, neutral midpoint).
"""


def _tab_evidence_from_mkp(tab: str, mkp: dict, catalog: dict, index: dict):
    """Build questionnaire context from MKP/personas; optional legacy index."""
    figures = []
    # Prefer purchase drivers / incentives as "category value" inspiration — not %.
    for item in (mkp.get("top_purchase_drivers") or [])[:12]:
        figures.append({
            "topic": "purchase drivers",
            "measure": item,
            "verbatim": item,
            "unit": None,
            "source_market": None,
            "is_direct_match": True,
            "fact": item,
        })
    # Legacy indexer figures (optional) — topic inspiration only.
    grounded = index.get("grounded_facts") or []
    by_tab = index.get("by_tab") or {}
    for i in by_tab.get(tab, [])[:8]:
        if i < len(grounded):
            f = grounded[i]
            fig = f.get("figure") or {}
            figures.append({
                "topic": f.get("topic"),
                "measure": fig.get("measure"),
                "verbatim": fig.get("verbatim"),
                "unit": fig.get("unit"),
                "source_market": f.get("source_market"),
                "is_direct_match": f.get("is_direct_match"),
                "fact": (f.get("fact") or "")[:220],
            })

    voice = []
    for text in (mkp.get("top_pain_points") or []):
        voice.append({"topic": "complaints", "origin": "mkp", "text": text})
    for text in (mkp.get("consumer_language") or []):
        voice.append({"topic": "customer_language", "origin": "mkp", "text": text})
    for text in (mkp.get("feature_requests") or []):
        voice.append({"topic": "feature_requests", "origin": "mkp", "text": text})
    for f in (index.get("context_facts") or [])[:20]:
        voice.append({
            "topic": f.get("topic"),
            "origin": f.get("origin"),
            "text": (f.get("fact") or "")[:220],
        })
    voice = voice[:30]

    personas = []
    for p in (catalog.get("personas") or [])[:16]:
        personas.append({
            "name": p.get("name"),
            "price_sensitivity": p.get("price_sensitivity"),
            "pain_points": (p.get("pain_points") or [])[:3],
            "goals": (p.get("goals") or [])[:3],
        })
    return figures, voice, personas


def _first_construct_match(text: str) -> str | None:
    """First CONSTRUCTS entry (validator_critic.py) a question's text matches."""
    for name, pat in _VALIDATOR_CONSTRUCTS:
        if re.search(pat, text or ""):
            return name
    return None


def _generate_tab(tab: str, count: int, index: dict, segment: str, region: str,
                  today: str, mode: str, grounding_thin: bool, feedback: str,
                  avoid_texts=None, mkp=None, catalog=None, layer: str = "full",
                  beats=None, blueprint=None, geo_label: str = "") -> dict:
    """Generate one tab's questions beat by beat, in narrative order.

    ``layer``: "core" | "module" | "full"
    ``geo_label``: the geography this file is written for — a country when the
    job is a country job, otherwise the region. Only the module layer uses it.
    ``beats``: ordered storyline beats for this tab (Phase A8). Each LLM call
    covers a contiguous run of beats with an explicit per-beat quota, so the
    model never has to invent the survey's running order — it only orders
    questions within a single beat.
    # change for b2c questionarie — Phase A5 / A8
    """
    import json

    beats = list(beats or narrative.canonical_beats(tab))
    if not beats:
        beats = narrative.canonical_beats(tab)

    figures, voice, personas = _tab_evidence_from_mkp(tab, mkp or {}, catalog or {}, index or {})
    figures_json = json.dumps(figures, ensure_ascii=False)
    voice_json = json.dumps(voice, ensure_ascii=False)
    persona_json = json.dumps(personas, ensure_ascii=False)
    base_feedback = (
        f"- REVISION: a prior review flagged issues — address this feedback: {feedback}\n"
        if feedback else ""
    )
    # change for b2c questionarie -- CHECK (user directive, 2026-09-08): a
    # "coverage:" line means a REQUIRED question type is missing entirely.
    # Generic "address this feedback" buries it among style notes, and the
    # run then burns revisions re-polishing wording while the same slot stays
    # empty. Hoist it to the top as a hard, unmissable instruction.
    if feedback and "coverage:" in str(feedback).lower():
        _cov = [
            ln.strip() for ln in str(feedback).replace(";", "\n").splitlines()
            if "coverage:" in ln.lower()
        ]
        if _cov:
            base_feedback = (
                "- MANDATORY FIX FIRST — a REQUIRED question type is missing "
                "from this section. Writing it is the single most important "
                "job of this revision; do not spend slots on anything else "
                "until every required type below exists:\n"
                + "".join(f"    * {c}\n" for c in _cov)
                + base_feedback
            )
    llm = get_structured_llm(TabQuestionBatch, temperature=0.4, max_tokens=16000)

    # Regional modules pick their own beats (few questions, placed where the
    # region actually diverges); core/full layers get explicit per-beat quotas.
    free_choice = layer == "module"
    if free_choice:
        plan = [([(b, 0) for b in beats], count)]
    else:
        allocations = narrative.allocate_questions(beats, count)
        plan = [
            (chunk, sum(n for _, n in chunk))
            for chunk in narrative.chunk_allocations(allocations, QUESTION_CHUNK)
        ]

    avoid_texts = list(avoid_texts or [])
    collected: list[dict] = []
    notes: list[str] = []
    attempts = 0

    for chunk, chunk_target in plan:
        chunk_beats = [b for b, _ in chunk]
        chunk_beat_ids = {b["beat_id"] for b in chunk_beats}
        got: list[dict] = []
        chunk_tries = 0

        while (
            len(got) < chunk_target
            and chunk_tries < MAX_CHUNK_ATTEMPTS
            and attempts < MAX_TAB_ATTEMPTS
        ):
            chunk_tries += 1
            attempts += 1
            need = chunk_target - len(got)
            remaining = (
                [(b, 0) for b in chunk_beats] if free_choice
                else _remaining_allocation(chunk, got)
            )
            beats_block = _format_beats_block(
                remaining, free_choice=free_choice, total=need,
            )

            if mode == "quota":
                mode_line = f"- QUOTA MODE: produce all {need} questions for this chunk.\n"
            else:
                mode_line = f"- HONEST MODE: produce up to {need} well-supported questions; fewer is OK.\n"
            # change for b2c questionarie — core vs regional module
            if layer == "core":
                mode_line += (
                    "- CORE INSTRUMENT: these questions must be COMPARABLE across all "
                    "regions (same constructs). Use universal option sets.\n"
                    # change for b2c questionarie -- CHECK (Section 30, currency):
                    # core questions are cached ONCE per segment and reused across
                    # every country -- confirmed live: a core money question
                    # hardcoded USD bands ("Under $50"..."$200 or more") into a
                    # China-region file, because "universal option sets" was never
                    # explicitly said to mean NO currency symbol at all. The
                    # {currency} placeholder in this prompt is filled with
                    # whichever country happens to trigger the cache-miss that
                    # generates this core batch -- that country's currency is NOT
                    # universal and must never appear literally in a core option.
                    "- NEVER put a currency symbol or amount (no $, €, £, ¥, ₹, or "
                    "any number implying a specific currency) in a CORE question's "
                    "options. A money question at this layer must use RELATIVE "
                    "price framing that means the same thing in every currency: "
                    "\"Budget / Mid-range / Premium\", \"Cheapest available / "
                    "Mid-priced / Most expensive option\", or a plain frequency/"
                    "behaviour framing instead of a price band. The regional "
                    "MODULE layer is where a concrete, currency-denominated price "
                    "band belongs — never here.\n"
                )
            elif layer == "module":
                # change for b2c questionarie — when the survey is written for a
                # country, the module is what makes that file worth having as
                # its own document. Anchor it on the country, not the region:
                # a German respondent shops in German channels and pays in
                # euros, and a module that says "in Europe" is the generic
                # question the core already asked.
                mode_line += (
                    f"- LOCAL MODULE for {geo_label}: write these questions so a "
                    f"consumer in {geo_label} recognises their own market. Use the "
                    "retail channels people there actually use, the local currency "
                    "for any money question, local pack sizes/formats, and the "
                    "seasons or conditions that apply there. Lift OPTIONS from "
                    f"local consumer voice for {geo_label}. Never ask where the "
                    "respondent lives — everyone answering this is already there. "
                    "Do NOT list brand or company names.\n"
                )
            else:
                mode_line += (
                    "- UNIVERSAL QUESTIONNAIRE: questions must be REGION-INVARIANT "
                    "(same instrument for all regions). Do not write region-specific instruments.\n"
                )
            if grounding_thin:
                mode_line += "- Market profile is THIN: still design strong consumer questions; expect simulation to infer.\n"

            prior_texts = (
                avoid_texts
                + [c["text"] for c in collected]
                + [g["text"] for g in got]
            )
            # Testing phase: Sonnet gets its own tuned template (see
            # question_architect_prompts_sonnet.py for why); DeepSeek's
            # PROMPT_TEMPLATE below is completely untouched either way.
            active_template = PROMPT_TEMPLATE
            if config.LLM_PROVIDER == "anthropic":
                from src.nodes.question_architect_prompts_sonnet import (
                    PROMPT_TEMPLATE as _SONNET_TEMPLATE,
                )
                active_template = _SONNET_TEMPLATE
            prompt = active_template.format(
                tab=tab, segment=segment, region=region, today=today,
                currency=standard_sections.currency_for(geo_label or region),
                study_purpose=(
                    narrative.study_purpose(blueprint)
                    or f"Understand how consumers in {region} buy and use {segment}."
                ),
                segments_block=_segments_block(blueprint),
                must_cover_block=_must_cover_block(blueprint),
                regions=", ".join(GEOGRAPHIC_REGIONS),
                guidance=_guidance_for(tab, beats, blueprint),
                beats_block=beats_block,
                figures_json=figures_json, voice_json=voice_json,
                persona_json=persona_json,
                count=need, mode_line=mode_line,
                feedback_line=base_feedback + _avoid_block(prior_texts),
            )
            try:
                batch: TabQuestionBatch = llm.invoke(prompt)
            except Exception as exc:  # noqa: BLE001 — a failed chunk must not sink the tab
                notes.append(f"chunk generation failed: {exc}")
                break

            notes.extend(batch.notes)
            added = 0
            # change for b2c questionarie -- CHECK (live, Sport Shoes/China,
            # 2026-09-06): `running` used to be reset to `prior_texts` alone
            # at the top of EVERY chunk iteration, so a question accepted in
            # an EARLIER chunk of this same tab (e.g. "How satisfied are you
            # with your current sport shoes?" in beat 1) was invisible to
            # dedup when a LATER chunk of the same tab wrote "How satisfied
            # are you with the sport shoes you wear most often?" -- neither
            # the text-similarity check nor any construct check ever saw
            # both at once. Seeding with `collected` (everything kept from
            # prior chunks of THIS tab generation) closes that gap.
            running = list(prior_texts) + [c["text"] for c in collected]
            # change for b2c questionarie -- CHECK (same live run): text
            # similarity alone missed this pair because the wording differs
            # enough to dodge the Jaccard threshold, even though both hit
            # the SAME construct in validator_critic.py's CONSTRUCTS table
            # (overall_satisfaction / purchase_intent). Track which
            # constructs are already claimed by an accepted question in this
            # tab generation, and drop a new question that claims the same
            # one again -- construct collision is checked independently of,
            # and in addition to, the text-similarity check above.
            claimed_constructs: dict[str, str] = {}
            for prior_text in running:
                cname = _first_construct_match(prior_text)
                if cname and cname not in claimed_constructs:
                    claimed_constructs[cname] = prior_text
            for q in batch.questions:
                data = q.model_dump()
                text = data.get("text", "")
                if not text:
                    continue
                if too_similar_to_any(
                    text, running, segment=segment, threshold=LOOSE_JACCARD
                ):
                    notes.append(f"dropped near-duplicate during gen: {text[:80]}")
                    continue
                cname = _first_construct_match(text)
                if cname and cname in claimed_constructs:
                    notes.append(
                        f"dropped construct-duplicate ({cname}, already covered "
                        f"by \"{claimed_constructs[cname][:60]}\"): {text[:80]}"
                    )
                    continue
                if _is_geo_residence_question(text):
                    notes.append(f"dropped banned geo-residence question: {text[:80]}")
                    continue
                if _is_age_or_income_question(text):
                    notes.append(f"dropped banned age/income question: {text[:80]}")
                    continue
                if _is_brand_name_question(text, data.get("options") or []):
                    notes.append(f"dropped banned brand/company question: {text[:80]}")
                    continue
                # change for b2c questionarie — a wrong beat_id would misplace the
                # question in the story, so repair it onto this chunk's beats.
                bid = (data.get("beat_id") or "").strip()
                if bid not in chunk_beat_ids:
                    repaired = _nearest_beat_id(bid, chunk_beats)
                    notes.append(
                        f"beat_id {bid!r} not in chunk — repaired to {repaired!r}: {text[:60]}"
                    )
                    bid = repaired
                data["beat_id"] = bid
                data["question_layer"] = layer if layer != "full" else "full"
                running.append(text)
                if cname:
                    claimed_constructs.setdefault(cname, text)
                got.append(data)
                added += 1
                if len(got) >= chunk_target:
                    break
            if added == 0:
                break

        collected.extend(got)
        if len(got) < chunk_target:
            notes.append(
                f"beat shortfall in {tab}: {len(got)}/{chunk_target} across beats "
                f"{[b['beat_id'] for b in chunk_beats]}"
            )

    if len(collected) < count:
        notes.append(f"shortfall: {len(collected)}/{count} after {attempts} attempts")

    collected = collected[:count]
    # Narrative order is decided here, in code — never by the model's list order.
    collected = narrative.order_questions(collected, beats)

    prefix = _prefix_for(tab, blueprint)
    for n, q in enumerate(collected, start=1):
        q["id"] = f"{prefix}{n}"
        q["tab"] = tab
        if not q.get("options"):
            q["options"] = ["Other"]
    return {"tab": tab, "questions": collected, "notes": notes}


def _build_tabs(
    *,
    tab_targets: list[tuple[str, int]],
    kept_by_tab: dict,
    index: dict,
    segment: str,
    region: str,
    today: str,
    grounding_thin: bool,
    feedback: str,
    mkp: dict,
    catalog: dict,
    layer: str,
    blueprint: dict | None = None,
    seed_avoid: list[str] | None = None,
    geo_label: str = "",
) -> tuple[list, dict]:
    """Generate/refill tabs sequentially with a shared anti-dup list.

    ``global_avoid`` spans every tab, so uniqueness is enforced across the whole
    survey rather than within a section.
    # change for b2c questionarie — Phase A8: each tab is filled beat by beat
    # and re-ordered by the storyline before it is returned.
    """
    global_avoid: list[str] = list(seed_avoid or [])
    for tab, _ in tab_targets:
        for q in kept_by_tab.get(tab, []):
            t = q.get("text") or ""
            if t and t not in global_avoid:
                global_avoid.append(t)

    # change for b2c questionarie — cost/time optimization step 2. The 4 tabs
    # were run one after another, each a blocking LLM call. `global_avoid`
    # LOOKS like a cross-tab dependency (it grows across iterations below),
    # but every `_generate_tab` call only ever reads the snapshot taken
    # BEFORE the loop starts (built solely from `kept_by_tab`, i.e. survivors
    # from a prior pass) — the in-loop growth is write-only until the next
    # section's read, and each tab reads before any tab writes back. So the
    # 4 calls have no real cross-iteration data dependency and can run
    # concurrently against one shared frozen snapshot. Exact-text duplicates
    # across freshly-generated tabs are still caught downstream by the
    # existing within-survey duplicate detector in validator_critic.py,
    # which already runs regardless of how this loop is scheduled — that
    # detector was always the actual backstop for this case, not this list.
    frozen_avoid = list(global_avoid)

    def _plan_tab(tab: str, target: int):
        beats = narrative.beats_for_tab(blueprint, tab)
        existing = kept_by_tab.get(tab, [])[:target]
        need = target - len(existing)
        if need <= 0:
            return tab, existing, None
        # Refill the parts of the story that are still missing first, so a
        # revision cannot leave an early beat empty while late beats double up.
        # change for b2c questionarie — refill the uncovered beats FIRST, but
        # keep the rest available. Restricting refill to open beats alone made
        # a revision pile three questions into one beat when only one was open.
        covered = {q.get("beat_id") for q in existing if q.get("beat_id")}
        open_beats = [b for b in beats if b["beat_id"] not in covered]
        if not open_beats:
            open_beats = list(beats)
        elif need > len(open_beats):
            # Priority order: gaps first, then everything else, so
            # allocate_questions spreads the surplus instead of stacking it.
            open_beats = open_beats + [b for b in beats if b["beat_id"] in covered]
        return tab, existing, (need, open_beats)

    plans = [_plan_tab(tab, target) for tab, target in tab_targets]
    to_generate = [(tab, existing, spec) for tab, existing, spec in plans if spec]

    def _run(tab: str, need: int, open_beats: list) -> dict:
        return _generate_tab(
            tab, need, index, segment, region, today,
            GENERATION_MODE, grounding_thin, feedback,
            avoid_texts=frozen_avoid,
            mkp=mkp, catalog=catalog, layer=layer,
            beats=open_beats, blueprint=blueprint,
            geo_label=geo_label,
        )

    generated: dict = {}
    if to_generate:
        with ThreadPoolExecutor(max_workers=len(to_generate)) as ex:
            futures = {
                tab: ex.submit(_run, tab, need, open_beats)
                for tab, _existing, (need, open_beats) in to_generate
            }
            generated = {tab: fut.result() for tab, fut in futures.items()}

    results = {}
    for tab, existing, spec in plans:
        if not spec:
            results[tab] = existing
            continue
        tab_qs = existing + generated[tab]["questions"]
        results[tab] = tab_qs
        for q in tab_qs:
            t = q.get("text") or ""
            if t and t not in global_avoid:
                global_avoid.append(t)

    questions, counts_by_tab = [], {}
    for tab, _count in tab_targets:
        prefix = _prefix_for(tab, blueprint)
        beats = narrative.beats_for_tab(blueprint, tab)
        tab_qs = narrative.order_questions(results.get(tab, []), beats)
        for n, q in enumerate(tab_qs, start=1):
            q["id"] = f"{prefix}{n}"
            q["tab"] = tab
            q["question_layer"] = q.get("question_layer") or layer
        questions.extend(tab_qs)
        counts_by_tab[tab] = len(tab_qs)
    return questions, counts_by_tab


def question_architect(state: SurveyState) -> dict:
    """Generate questionnaire — core+module for region jobs (A5)."""
    segment = state.get("normalized_segment") or state.get("market_segment") or "the segment"
    region = state.get("region") or "Global"
    # change for b2c questionarie — the geography the module layer localises to.
    geo_label = state.get("country") or state.get("geography_label") or region
    index = state.get("evidence_index") or {}
    mkp = state.get("market_knowledge_profile") or {}
    catalog = state.get("persona_catalog") or {}
    grounding_thin = bool(state.get("grounding_thin"))
    today = current_date_context()["iso"]
    # change for b2c questionarie — Phase A8: the storyline decides question order
    blueprint = state.get("survey_blueprint") or narrative.default_blueprint(
        segment=segment, region=region,
    )
    slug = region_to_slug(region)
    # change for b2c questionarie — the module layer localises to geo_label
    # (the country when one is set), but the cache key used to be built from
    # slug (the PARENT region only). Two countries in the same region, e.g.
    # India and China in Asia Pacific, hashed to the same module cache key
    # even though their prompts and expected content differ by country —
    # the second job could silently reuse the first job's country-specific
    # module questions. The module slug must track geo_label, not region.
    module_slug = region_to_slug(geo_label) if geo_label else slug
    is_geo_job = match_geographic_region(region) is not None
    use_core_module = (
        REGION_QUESTION_MODE == "core_plus_module" and is_geo_job
    )

    critique = state.get("critique") or {}
    is_revision = bool(critique)
    feedback = critique.get("feedback") or critique.get("notes") or ""
    if isinstance(feedback, list):
        feedback = " ".join(feedback)
    # change for b2c questionarie — the per-question issues quote the EXACT
    # defect ("asks 'friend or colleague', a workplace frame") and the exact
    # offending words, but only the generic feedback list reached this prompt.
    # Without it, a revision fixed some things and reintroduced the same named
    # defects in the new questions, because it never saw what was specifically
    # wrong with what it wrote last time — only a generic "fix issues" note.
    per_q_notes = []
    for qi in (critique.get("per_question") or [])[:20]:
        for issue in (qi.get("issues") or [])[:1]:
            per_q_notes.append(f"[{qi.get('id')}] {issue}")
    if per_q_notes:
        feedback = (feedback + "\n\nSPECIFIC DEFECTS FROM THE LAST PASS — do not "
                    "repeat these in the new questions:\n" + "\n".join(per_q_notes))

    # change for b2c questionarie — cache key includes geography (+ mode).
    # Was `slug` (parent region only) — see module_slug comment above for why
    # that let two countries in the same region collide on this key.
    cache_key = cache_store.make_key(
        _Q_SCHEMA_V, segment, module_slug, REGION_QUESTION_MODE,
        (mkp.get("top_pain_points") or [])[:8],
        [p.get("id") for p in (catalog.get("personas") or [])],
        # change for b2c questionarie — a changed storyline must regenerate
        narrative.blueprint_digest(blueprint),
    )
    if not is_revision:
        cached = cache_store.get_json("questionnaire", cache_key)
        if (
            cached
            and cached.get("questions")
            and sum((cached.get("counts_by_tab") or {}).values())
            >= QUESTIONNAIRE_CACHE_MIN_QUESTIONS
        ):
            out = {
                "questions": cached["questions"],
                "counts_by_tab": cached.get("counts_by_tab") or {},
                "questionnaire_id": cache_key,
                "questionnaire_cache_hit": True,
            }
            if cached.get("questions_by_region"):
                out["questions_by_region"] = cached["questions_by_region"]
            elif is_geo_job:
                out["questions_by_region"] = {slug: cached["questions"]}
            return out

    drop_ids = set()
    if is_revision:
        for cluster in critique.get("duplicates") or []:
            for dup in cluster[1:]:
                drop_ids.add(dup)
        for qi in critique.get("per_question") or []:
            if qi.get("fix_target") == "architect":
                joined = " ".join(qi.get("issues") or []).lower()
                if any(k in joined for k in (
                    "duplicate", "subject", "off-segment", "open-ended",
                    "missing options", "geo-residence", "region of residence",
                    "irrelevant geography", "screener",
                    "double-barrel", "double barreled", "leading", "loaded",
                    "mece", "overlapping option", "unbalanced", "likert",
                    "escape", "other/none", "mutually exclusive",
                    "collectively exhaustive", "hygiene",
                    "age", "income", "brand name", "company name", "manufacturer",
                    # change for b2c questionarie — the newer detectors were not
                    # in this list, so a question flagged for grammar, jargon,
                    # a same-construct duplicate or a broken flow was KEPT and
                    # re-flagged on every pass. The loop could never converge:
                    # the architect rewrote clean questions while the actual
                    # offenders survived to the end.
                    "redundancy", "language:", "flow:", "measures the same",
                    "not clearly worded", "consistency:", "missing a word",
                    "research jargon", "passive voice", "throat-clearing",
                    "reads at about grade", "asks who decides", "logic:",
                    "contradicts the screener", "not one of the survey's "
                    "declared segments",
                )):
                    drop_ids.add(qi.get("id"))

    # change for b2c questionarie -- CHECK (Section 26e): the satisfaction-tab
    # anchors (overall satisfaction + NPS) are spliced into `questions` AFTER
    # _build_tabs() already ran, so the architect never saw "How likely are
    # you to recommend..." as an avoid-text while writing the satisfaction
    # tab -- it could, and on the Running Shoes survey did, write its own
    # near-duplicate NPS-style recommendation question with a different scale
    # (Likert vs the anchor's 0-10). The post-hoc "advocacy" construct
    # detector in validator_critic.py already catches this on a revision
    # pass, but seeding the anchor text up front stops it at first generation
    # instead of relying solely on a later drop-and-regenerate cycle.
    _anchor_avoid = [] if _SHORT_SURVEY_NO_ANCHORS else [
        q.get("text") or ""
        for q in standard_sections.satisfaction_anchor_questions(segment)
    ]

    kept_by_tab = {t: [] for t in narrative.section_ids(blueprint)}
    for q in state.get("questions") or []:
        if q.get("id") in drop_ids:
            continue
        if _is_geo_residence_question(q.get("text") or ""):
            continue
        if _is_age_or_income_question(q.get("text") or ""):
            continue
        if _is_brand_name_question(q.get("text") or "", q.get("options") or []):
            continue
        kept_by_tab.setdefault(q.get("tab"), []).append(q)

    # change for b2c questionarie -- CHECK (Section 30, escalated 28c): the
    # exact-text avoid list (global_avoid, built below) only stops the model
    # from re-writing the SAME sentence -- it has no concept of "a different
    # sentence measuring the same construct". Confirmed live: a known
    # construct-collision duplicate (fit_difficulty: confidence-scale vs
    # struggle/frequency-scale framing of the same fit problem) kept
    # reappearing across 6 revisions even after the offending question was
    # correctly dropped each time -- the SURVIVING question's construct was
    # never explicitly declared "already covered", so the architect kept
    # independently regenerating a fresh, differently-worded question that
    # collided with it again. Explicitly tell the model which constructs a
    # surviving question already covers before it writes anything new.
    # change for b2c questionarie -- CHECK (live, Sport Shoes/China): the
    # satisfaction-tab anchors (overall satisfaction + NPS "would you
    # recommend...") are spliced into `questions` at the very END of this
    # node, entirely outside kept_by_tab / the is_revision branch below --
    # so the architect never learned the anchor already covers "advocacy",
    # and kept independently writing its own "would you recommend the
    # brand..." question on every single revision. Because the anchor was
    # ALWAYS present from generation 1 onward, every pass was equally
    # dirty (known_duplicate_fail=True every time) and the best-of-worst
    # restore had no clean pass to prefer, so the duplicate always shipped.
    # Fix: declare the anchors' constructs as covered unconditionally, not
    # only on revisions -- the anchors are deterministic, not something a
    # prior pass "survived" into kept_by_tab.
    covered_constructs = {}
    for anchor_text in _anchor_avoid:
        for cname, cpat in _VALIDATOR_CONSTRUCTS:
            if re.search(cpat, anchor_text) and cname not in covered_constructs:
                covered_constructs[cname] = ("standard-anchor", anchor_text)
    if is_revision:
        for tab_qs in kept_by_tab.values():
            for q in tab_qs:
                qtext = q.get("text") or ""
                for cname, cpat in _VALIDATOR_CONSTRUCTS:
                    if re.search(cpat, qtext) and cname not in covered_constructs:
                        covered_constructs[cname] = (q.get("id"), qtext)
    if covered_constructs:
        lines = [
            f"- {name.replace('_', ' ')}: already covered by "
            f"[{qid}] \"{text[:70]}\" — do NOT write another question "
            "for this construct in any wording, scale, or framing"
            for name, (qid, text) in covered_constructs.items()
        ]
        feedback = (
            feedback + "\n\nCONSTRUCTS ALREADY COVERED BY SURVIVING "
            "QUESTIONS — do not add a differently-worded question that "
            "measures the same thing:\n" + "\n".join(lines)
        )

    if use_core_module and not is_revision:
        # Shared CORE (segment-level cache) + regional MODULE.
        core_key = cache_store.make_key(_Q_SCHEMA_V, segment, "core", REGION_QUESTION_MODE)
        core_cached = cache_store.get_json("questionnaire", core_key)
        if core_cached and core_cached.get("questions"):
            core_qs = core_cached["questions"]
        else:
            # change for b2c questionarie — CORE is written against the CANONICAL
            # arc so the shared instrument is identical across regions, then
            # remapped onto this market's sections at merge time.
            # It must use CANONICAL SECTION IDS as well as blueprint=None: with
            # market ids and no blueprint, every section fell back to
            # consumer_profile, so all four core sections generated
            # consumer-profile questions and the dedup discarded most of them.
            canonical_sections = list(narrative.canonical_section_ids())
            core_plan = [(c, CORE_PER_TAB) for c in canonical_sections]
            core_qs, _ = _build_tabs(
                tab_targets=core_plan,
                kept_by_tab={c: [] for c in canonical_sections},
                index=index, segment=segment, region="Global", today=today,
                grounding_thin=grounding_thin, feedback=feedback,
                mkp=mkp, catalog=catalog, layer="core", blueprint=None,
                seed_avoid=_anchor_avoid,
            )
            for q in core_qs:
                q["question_layer"] = "core"
            cache_store.set_json("questionnaire", core_key, {"questions": core_qs})

        # Core questions are tagged with CANONICAL section ids; index them that
        # way so each market section can claim the core for its canonical job.
        core_by_canonical: dict = {}
        for q in core_qs:
            core_by_canonical.setdefault(q.get("tab"), []).append(q)

        module_plan = [(t, MODULE_PER_TAB) for t in narrative.section_ids(blueprint)]
        module_qs, _ = _build_tabs(
            tab_targets=module_plan,
            kept_by_tab={t: [] for t in narrative.section_ids(blueprint)},
            index=index, segment=segment, region=region, today=today,
            grounding_thin=grounding_thin, feedback=feedback,
            mkp=mkp, catalog=catalog, layer="module", blueprint=blueprint,
            seed_avoid=[q.get("text") or "" for q in core_qs] + _anchor_avoid,
            geo_label=geo_label,
        )
        for q in module_qs:
            q["question_layer"] = "module"

        # change for b2c questionarie — Phase A8: merge core + module by STORY
        # position, not by appending. A regional module question about an early
        # beat now sits early, instead of being pinned to the end of the tab.
        questions, counts_by_tab = [], {}
        for tab in narrative.section_ids(blueprint):
            prefix = _prefix_for(tab, blueprint)
            beats = narrative.beats_for_tab(blueprint, tab)
            canon = narrative.section_canonical(blueprint, tab)
            core_part = [
                dict(q) for q in (core_by_canonical.get(canon) or [])[:CORE_PER_TAB]
            ]
            # Remap using the CANONICAL section the core was written for, so the
            # canonical beat ranks resolve.
            narrative.remap_questions_to_beats(core_part, beats, canon)
            module_part = [
                dict(q) for q in module_qs if q.get("tab") == tab
            ][:MODULE_PER_TAB]
            merged = narrative.order_questions(core_part + module_part, beats)
            for n, q in enumerate(merged, start=1):
                q["id"] = f"{prefix}{n}"
                q["tab"] = tab
                questions.append(q)
            counts_by_tab[tab] = len(merged)
    else:
        # Full distinct / Global / revision path — fill to TARGET.
        layer = "module" if (REGION_QUESTION_MODE == "fully_distinct" and is_geo_job) else "full"
        questions, counts_by_tab = _build_tabs(
            tab_targets=narrative.plan_for(blueprint, QUESTIONS_PER_TAB_TARGET),
            kept_by_tab=kept_by_tab,
            index=index, segment=segment, region=region, today=today,
            grounding_thin=grounding_thin, feedback=feedback,
            mkp=mkp, catalog=catalog, layer=layer, blueprint=blueprint,
            seed_avoid=_anchor_avoid,
            geo_label=geo_label,
        )

    # change for b2c questionarie — Phase A9: bookend the market sections with
    # the two standard sections every consultancy study carries. They are
    # templated (not planner-invented) so they stay consistent across markets,
    # and they run through simulation like any other question so the client gets
    # real distributions for the cross-tabs.
    # Screening is deliberately NOT emitted: qualification is a fielding
    # concern, not part of the questionnaire a client reads. The survey opens
    # on the first substantive question.
    profiling = [
        {
            **q, "tab": standard_sections.PROFILING_SECTION_ID,
            "chart_type": "bar", "evidence_aligned": False,
            "options_from_evidence": False, "question_layer": "profiling",
            "narrative_order": i + 1, "funnel_position": i + 1,
        }
        for i, q in enumerate(
            standard_sections.profiling_questions(
                segment,
                currency=standard_sections.currency_for(geo_label),
                geography=geo_label,
            )
        )
    ]
    # change for b2c questionarie — overall satisfaction and NPS anchor the
    # satisfaction section instead of trailing the demographics. They lead that
    # section (broadest judgement first), so the generated questions there
    # narrow into drivers, friction and switching behind them.
    if not _SHORT_SURVEY_NO_ANCHORS:
        sat_tab = None
        for tab_id in narrative.section_ids(blueprint):
            if "satisfaction" in tab_id or "intent" in tab_id:
                sat_tab = tab_id
                break
        if sat_tab:
            anchors = [
                {
                    **q, "tab": sat_tab, "chart_type": "bar",
                    "evidence_aligned": False, "options_from_evidence": False,
                    "question_layer": "core", "funnel_position": 0,
                    "narrative_order": 0,
                }
                for q in standard_sections.satisfaction_anchor_questions(segment)
            ]
            # Push the generated questions down so the anchors sit first.
            for q in questions:
                if q.get("tab") == sat_tab:
                    q["funnel_position"] = (q.get("funnel_position") or 1) + len(anchors)
            questions = questions + anchors
            counts_by_tab[sat_tab] = counts_by_tab.get(sat_tab, 0) + len(anchors)

    questions = questions + profiling
    counts_by_tab[standard_sections.PROFILING_SECTION_ID] = len(profiling)

    questions_by_region = {slug: questions} if is_geo_job else {}

    if not is_revision:
        cache_store.set_json("questionnaire", cache_key, {
            "questions": questions,
            "counts_by_tab": counts_by_tab,
            "questions_by_region": questions_by_region,
        })

    out = {
        "questions": questions,
        "counts_by_tab": counts_by_tab,
        "questionnaire_id": cache_key,
        "questionnaire_cache_hit": False,
    }
    if questions_by_region:
        out["questions_by_region"] = questions_by_region
    return out


if __name__ == "__main__":
    demo_index = {
        "grounded_facts": [
            {
                "fact": "North America holds 41% of organic milk sales, Europe 33%, Asia Pacific 18%, Rest of World 8%.",
                "origin": "stat", "source_ref": "https://example.gov/dairy",
                "source_market": "Organic Milk", "is_direct_match": True,
                "relevance": "high", "grounding_label": "", "tab": "consumer_profile",
                "topic": "regional market share",
                "figure": {"measure": "regional share", "number": 41.0, "unit": "%",
                           "period": "2025", "verbatim": "41%"},
                "source_date": "2025-11-01", "corroborated_by": [],
            },
            {
                "fact": "Premium organic milk retails at USD 4-6 per gallon; mid-tier 3-4; value under 3.",
                "origin": "cmi_database", "source_ref": "report:1618",
                "source_market": "Organic Milk", "is_direct_match": True,
                "relevance": "high", "grounding_label": "", "tab": "buying_behavior",
                "topic": "price tier",
                "figure": {"measure": "premium price band", "number": 5.0, "unit": "USD",
                           "period": None, "verbatim": "USD 4-6 per gallon"},
                "source_date": None, "corroborated_by": [],
            },
        ],
        "context_facts": [],
        "by_tab": {"consumer_profile": [0], "buying_behavior": [1]},
        "grounded_count": 2, "direct_match_count": 2, "adjacent_count": 0,
    }
    state = {
        "normalized_segment": "Organic Milk",
        "segment_type": "b2c",
        "region": "Global",
        "evidence_index": demo_index,
        "grounding_thin": False,
    }
    out = question_architect(state)
    counts = out["counts_by_tab"]
    qs = out["questions"]
    aligned = sum(1 for q in qs if q.get("evidence_aligned"))
    print(f"counts_by_tab: {counts}")
    print(f"total questions: {len(qs)} | evidence_aligned: {aligned}")
    for tab, _c in CATEGORY_PLAN:
        print(f"\n=== {tab} ===")
        for q in [x for x in qs if x["tab"] == tab][:6]:
            print(f"  [{q['id']}] aligned={q['evidence_aligned']} "
                  f"topic={q.get('serves_topic')!r} chart={q['chart_type']} type={q['type']}")
            print(f"       {q['text']}")
            print(f"       options: {q['options']}")
