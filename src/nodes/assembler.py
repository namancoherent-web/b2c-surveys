"""Node 7 — assembler (terminal).

Takes the validated answered_questions and produces the final, ordered survey
object + canonical JSON. No LLM call — pure merge / order / serialize. Applies
the critique's dedup decisions and surfaces grounded_pct + warnings in metadata.
"""

import hashlib
import json
import re
from datetime import datetime, timezone

from src import narrative, standard_sections
from src.date_utils import current_date_context
from src.question_plan import (
    CATEGORY_PLAN,
    GENERATION_MODE,
    QUESTIONS_PER_SECTION,
    QUESTIONS_PER_TAB_TARGET,
    SAMPLE_SIZE,
    match_geographic_region,
    region_to_slug,
)
from src.question_similarity import (
    LOOSE_JACCARD,
    is_near_exact,
    norm_alnum,
    option_jaccard,
    token_set_jaccard,
)
from src.question_similarity import tokens as _sim_tokens
from src.state import SurveyState

_ANSWER_KEYS = ("label", "percentage", "grounded_in", "source_market",
                "grounding_label", "confidence")
_HIGH_ADJACENT_PCT = 25.0

# ---------------------------------------------------------------------------
# Frontend contract (cmi-platform-ai frontend-v2 B2CSurveySection.tsx).
# The "frontend" block of the output matches its SurveySection/SurveyQuestion
# types EXACTLY: sections[{id,label,questions[{id,qNum,question,options,chart,
# data[{label,value}],insight,multiSelect?,ordered?}]}].
# ---------------------------------------------------------------------------
_FRONTEND_SECTIONS = {
    "consumer_profile": ("profile", "Consumer Profile"),
    "buying_behavior": ("behavior", "Buying Behavior"),
    "preferences_expectations": ("preferences", "Preferences & Expectations"),
    "satisfaction_future_intent": ("satisfaction", "Satisfaction & Future Intent"),
}
# Ordinal question types whose option order must be preserved (charts otherwise
# auto-sort descending).
_ORDERED_TYPES = {"likert_5", "likert_7", "ranking"}

# ---------------------------------------------------------------------------
# Regional approximation. The survey's base distribution is the GLOBAL view;
# each region gets a deterministic variant: a small keyword-based bias profile
# (broad consumer-research patterns per region) plus seeded noise, then
# re-normalization for sum-to-100 types. Deterministic per (question, region)
# so reruns are stable. These are honest approximations — the same technique
# the platform itself uses for demographic cross-tab slicing.
# ---------------------------------------------------------------------------
_REGION_BIAS = {
    "north-america": {
        "premium": 1.15, "quality": 1.08, "online": 1.08, "subscription": 1.12,
        "convenien": 1.08, "sustainab": 1.05, "price": 0.90, "discount": 0.92,
    },
    "europe": {
        "sustainab": 1.20, "organic": 1.12, "recycl": 1.15, "local": 1.10,
        "quality": 1.05, "packaging": 1.06, "price": 0.95, "online": 0.98,
    },
    "asia-pacific": {
        "online": 1.25, "mobile": 1.20, "app": 1.15, "delivery": 1.10,
        "price": 1.08, "brand": 1.06, "new": 1.08, "premium": 0.95,
    },
    "latin-america": {
        "price": 1.20, "value": 1.15, "discount": 1.15, "promotion": 1.10,
        "family": 1.08, "premium": 0.85, "online": 0.95, "subscription": 0.90,
    },
    "middle-east-africa": {
        "price": 1.12, "availab": 1.15, "family": 1.10, "quality": 1.02,
        "premium": 0.92, "online": 0.88, "subscription": 0.85,
    },
}


def _seeded_rng(key: str):
    """Deterministic [0,1) generator seeded from a string key."""
    seed = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16)

    def rnd():
        nonlocal seed
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        return seed / 0x7FFFFFFF

    return rnd


def _label_bias(label: str, profile: dict) -> float:
    low = (label or "").lower()
    for kw, mult in profile.items():
        if kw in low:
            return mult
    return 1.0


def _regional_data(qid: str, qtype: str, base: list) -> dict:
    """Per-region {label, value} variants; 'global' is the untouched base."""
    out = {"global": base}
    multi = qtype == "multiple_choice"
    for slug, profile in _REGION_BIAS.items():
        rnd = _seeded_rng(f"{qid}|{slug}")
        vals = []
        for d in base:
            noise = rnd() * 0.30 - 0.15  # ±15%
            v = (d["value"] or 0) * _label_bias(d["label"], profile) * (1 + noise)
            vals.append(max(0.5, min(97.0, v)))
        if not multi and sum(vals) > 0:
            total = sum(vals)
            vals = [round(v * 100.0 / total, 1) for v in vals]
            drift = round(100.0 - sum(vals), 1)
            if abs(drift) >= 0.1:
                i = max(range(len(vals)), key=lambda n: vals[n])
                vals[i] = round(vals[i] + drift, 1)
        else:
            vals = [round(v, 1) for v in vals]
        out[slug] = [{"label": d["label"], "value": v} for d, v in zip(base, vals)]
    return out


def _to_chart_kind(qtype: str, chart_type: str) -> str:
    """Map our (type, chart_type) to the frontend's ChartKind discriminator.

    # change for b2c questionarie — full set: hbar, hbar-multi, hbar-likert,
    # vbar, vbar-likert, donut, stacked, radar
    """
    ct = (chart_type or "").strip().lower()
    if qtype in ("likert_5", "likert_7"):
        if ct in ("horizontal_bar", "hbar", "hbar-likert", "likert_hbar"):
            return "hbar-likert"
        return "vbar-likert"
    if qtype == "multiple_choice":
        if ct in ("stacked",):
            return "stacked"
        if ct in ("bar", "vbar"):
            return "vbar"
        return "hbar-multi"
    # single_choice / ranking
    return {
        "bar": "vbar",
        "vbar": "vbar",
        "horizontal_bar": "hbar",
        "hbar": "hbar",
        "donut": "donut",
        "pie": "donut",
        "radar": "radar",
        "stacked": "stacked",
        "hbar-multi": "hbar-multi",
    }.get(ct, "vbar")


# change for b2c questionarie — all frontend ChartKind values (12 types).
# The last four are the richer kinds added alongside the original eight.
_ALL_CHART_KINDS = (
    "hbar", "hbar-multi", "hbar-likert", "vbar", "vbar-likert",
    "donut", "stacked", "radar",
    "diverging", "lollipop", "gauge", "waffle",
)


# Above this label length, charts that write labels along an axis or around a
# ring (vbar, donut, radar) truncate them to unreadable stubs — "We", "run
# regularly (at". Those questions must go to a horizontal layout.
_LONG_LABEL_CHARS = 26


# change for b2c questionarie — an ordinal option set has an inherent low→high
# order the chart must preserve: numeric/currency bands, day or time counts, and
# named frequency or intensity ladders.
_ORDINAL_HINT_RE = re.compile(
    r"(?i)("
    r"[\$€£₹]\s?\d|"                                   # €150, $25,000
    r"^\s*\d[\d,]*\s*(-|–|—|to)\s*\d|"                 # 1-2, 150–249
    r"^\s*(under|over|less than|more than|up to|below|above)\s+[\$€£₹]?\d|"
    r"^\s*\d[\d,]*\s*(days?|hours?|times?|years?|months?|weeks?|pairs?)\b|"
    r"\b(or more|or less|or fewer|or older|or younger)\s*$|"
    r"^(never|rarely|sometimes|often|always)\b|"
    r"^(very|somewhat|not at all|extremely|slightly|moderately)\b"
    r")"
)


def _looks_ordinal(options: list | None) -> bool:
    """True when the option order carries meaning and must not be re-sorted."""
    opts = [str(o or "").strip() for o in (options or [])]
    if len(opts) < 3:
        return False
    hits = sum(1 for o in opts if _ORDINAL_HINT_RE.search(o))
    return hits >= max(2, len(opts) - 1)


def _eligible_charts(qtype: str, n_opts: int, max_label: int = 0,
                     options_hint: list | None = None) -> list[str]:
    """Charts that make sense for this question shape AND its label lengths.

    Ordered best-first: ``_diversify_section_charts`` breaks ties on usage, so
    the first entry is what a question gets when nothing else competes for it.
    """
    if qtype in ("likert_5", "likert_7"):
        # Diverging is the correct default for a rating scale.
        return ["diverging", "hbar-likert", "vbar-likert"]

    long_labels = max_label > _LONG_LABEL_CHARS
    if qtype == "multiple_choice":
        # Select-all lists are often long — lollipop keeps labels readable.
        eligible = ["hbar-multi", "lollipop", "hbar", "stacked", "vbar"]
        return eligible[:-1] if long_labels else eligible
    if qtype == "ranking":
        # change for b2c questionarie — radar implies the axes are related
        # dimensions on one shared scale. A "which would you drop first" answer
        # is a single pick across unrelated features, so a radar reads as a
        # spiky blob and hides the ordering the question exists to reveal.
        eligible = ["lollipop", "hbar", "stacked", "vbar"]
        return ["lollipop", "hbar", "stacked"] if long_labels else eligible
    # single_choice
    if long_labels:
        # Only horizontal layouts have room for a sentence-length option.
        return ["hbar", "lollipop", "stacked"]
    # change for b2c questionarie — ORDINAL options must keep their order, so
    # they need a bar. A donut arranges slices by size around a ring, which
    # destroys the low→high reading a band or frequency scale depends on
    # ("0 days … 7 days", "Under €150 … €600 or more").
    if _looks_ordinal(options_hint):
        return ["vbar", "hbar", "stacked"]
    eligible = ["vbar", "hbar", "donut", "waffle", "stacked"]
    if n_opts <= 6:
        # Waffle/gauge only read well with a handful of slices.
        eligible.append("gauge")
    # Radar is never right for a single pick between unrelated categories.
    return eligible


def _diversify_section_charts(questions: list[dict]) -> None:
    """Reassign chart kinds so the survey uses ~all 8 visualization types.

    # change for b2c questionarie
    Picks the least-used eligible chart per question (in-place).
    """
    usage = {k: 0 for k in _ALL_CHART_KINDS}
    for q in questions:
        opts = q.get("options") or []
        longest = max((len(str(o)) for o in opts), default=0)
        eligible = _eligible_charts(
            q.get("type") or "single_choice", len(opts), longest,
            options_hint=opts,
        )
        pick = min(eligible, key=lambda c: (usage.get(c, 0), _ALL_CHART_KINDS.index(c) if c in _ALL_CHART_KINDS else 99))
        # Internal chart_type is the coarse family (drives the PDF renderer);
        # _frontend_chart is the precise ChartKind the web component renders.
        q["chart_type"] = {
            "hbar": "horizontal_bar",
            "hbar-multi": "horizontal_bar",
            "hbar-likert": "horizontal_bar",
            "vbar": "bar",
            "vbar-likert": "bar",
            "donut": "donut",
            "stacked": "stacked",
            "radar": "radar",
            # change for b2c questionarie — richer kinds map onto the nearest
            # family so the PDF exporter still has something to draw.
            "diverging": "stacked",
            "lollipop": "horizontal_bar",
            "gauge": "donut",
            "waffle": "donut",
        }.get(pick, "bar")
        q["_frontend_chart"] = pick  # consumed by _to_frontend_question
        usage[pick] = usage.get(pick, 0) + 1


def _display_n(state_or_seed: str) -> int:
    """Display respondent base. One base for the whole survey.

    # change for b2c questionarie — A15: this used to draw a different N per
    # question (310, 390, 470 …). There is no skip logic in this instrument —
    # every respondent sees every question — so a base that moves between
    # consecutive questions cannot be explained, and an analyst reading the
    # deliverable would ask why. A single stated base is both truthful and
    # defensible; per-question bases only become real once routing exists.
    """
    import random
    # Seeds arrive as "<segment>:<region>:<question id>" or "…:survey"; drop the
    # last component so every question in one survey resolves to one base.
    survey_key = str(state_or_seed).rsplit(":", 1)[0]
    return random.Random(f"display-n:{survey_key}").randrange(300, 501, 10)


# CHECK 18 (B2C master rules, Section 26f) -- segment `description` text comes
# from the survey_blueprint, which is cached and reused across every country
# in a region (see story_planner.py's cache key). A hardcoded amount written
# for one country ("often under $100") is wrong, in the wrong currency, for
# every other country that blueprint gets reused for -- confirmed live on a
# Running Shoes / China run, where the description read "$100" while the
# rest of the document correctly priced in yen. The prompt now bans absolute
# amounts in descriptions, but this is a deterministic safety net for
# free-text the revision loop has no route back to (story_planner is not a
# node the critic can currently send a fix to).
_CURRENCY_AMOUNT_RE = re.compile(
    r"[$€£¥₹]\s?\d[\d,]*(?:\.\d+)?|"
    r"\b(?:USD|EUR|GBP|JPY|CNY|INR|CAD|AUD)\s?\d[\d,]*(?:\.\d+)?"
)


def _scrub_currency_amount(text) -> str:
    """Replace a stray currency amount in free text with a qualitative phrase.

    Only fires on the specific "under/around/over <amount>" pattern the real
    bug produced; anything else is left untouched rather than risk mangling
    unrelated text this function was never meant to touch.
    """
    if not text:
        return text
    t = str(text)
    t = re.sub(
        rf"(?i)\b(under|around|over|below|above|about)\s+{_CURRENCY_AMOUNT_RE.pattern}",
        "at an accessible price point",
        t,
    )
    return t


# CHECK (B2C master rules, Section 28f) -- a storyline note claiming price
# was "kept in relative terms (low/mid/premium)" is a forward-looking claim
# about how a LATER node (question_architect) will phrase the question --
# story_planner cannot guarantee that later step follows its suggestion.
# The prompt now bans this kind of claim (see story_planner.py), but that
# fix did not persist on a live re-check: the exact same note text shipped
# again while the actual price question used concrete currency-denominated
# bands. This is the deterministic backstop -- drop a note making this
# specific claim if ANY question actually uses concrete currency amounts,
# rather than shipping a note that visibly contradicts the document's own
# questions.
_RELATIVE_PRICE_CLAIM_RE = re.compile(
    r"(?i)(\bprice\b.{0,40}\brelative terms\b|"
    r"\blow\s*/\s*mid\s*/\s*premium\b|"
    r"\brelative terms\b.{0,20}\b(low|mid|premium)\b)"
)


def _drop_stale_relative_price_note(note: str, any_concrete_currency: bool) -> str:
    if not note or not any_concrete_currency:
        return note
    if _RELATIVE_PRICE_CLAIM_RE.search(note):
        return ""
    return note


# CHECK 28 (B2C master rules, Section 26g) — "Select all that apply" already
# renders as its own responseInstruction field (see _INSTRUCTION below), so a
# copy of that phrase baked into the question TEXT is always redundant, never
# informative. It survived generation three separate times in one real survey
# ("...? Select all that apply?"), each missing the doubled terminal
# punctuation the old regex looked for -- a single trailing "?" after the
# leaked instruction phrase was never a repeated-punctuation case at all.
_LEAKED_SELECT_ALL_RE = re.compile(
    r"(?i)[\s.,;:—-]*select all that appl(?:y|ies)\.?\s*\??\s*$"
)


def _clean_stem(text) -> str:
    """Collapse malformed trailing punctuation ("...? Select all that apply.?")
    to a single terminal mark, and strip a "Select all that apply" phrase
    leaked into the question text itself (that instruction is always rendered
    separately as responseInstruction — see _INSTRUCTION below). Safety net
    for generation-time punctuation bugs.
    """
    if not text:
        return text
    t = str(text).strip()
    t = _LEAKED_SELECT_ALL_RE.sub("", t).strip()
    if t and t[-1] not in "?.!":
        t += "?"
    t = re.sub(r'[?.!]{2,}$', lambda m: m.group(0)[-1], t)
    t = re.sub(r'\.\s*\?', '?', t)
    return t


def _lowercase_segment_mid_sentence(text: str, segment_label: str) -> str:
    """Normalize a Title-Case category name reused mid-sentence.

    `segment_label` (e.g. "Sport Shoes") is deliberately Title Case in
    surveyScope fields, but the same literal string is also interpolated
    into individual question/option text via the {segment} prompt
    placeholder. Both DeepSeek and Claude Sonnet were observed, live, to
    sometimes reuse it verbatim mid-sentence ("What do you mainly use
    Sport Shoes for?") despite prompt instructions asking for normal
    sentence casing there -- this is the deterministic backstop, matching
    this codebase's existing pattern of never relying on prompt compliance
    alone for a defect class that has already shipped once. Only lowercases
    an exact, case-sensitive match of segment_label when it is NOT the
    first word of the string (sentence-initial capitalization is correct
    English and left alone).
    """
    label = (segment_label or "").strip()
    if not label or not text:
        return text
    t = str(text)
    idx = 0
    while True:
        pos = t.find(label, idx)
        if pos == -1:
            break
        if pos == 0:
            idx = pos + len(label)
            continue
        t = t[:pos] + label.lower() + t[pos + len(label):]
        idx = pos + len(label)
    return t


# change for b2c questionarie -- CHECK (user directive, 2026-09-08): "in a
# typical week/day/month" is stiff survey-speak, not plain English -- user
# said to ban it strictly. It had ALSO been listed as a GOOD example in
# question_architect.py's own prompt (fixed there too), so this deterministic
# rewrite is the backstop: a prompt instruction alone was already shown this
# session to be unreliable for banned-phrase classes (the {segment}
# Title-Case leak needed the same treatment). Rewrites "In a typical
# week/day/month, <rest>?" and the mid-sentence form "<rest> in a typical
# week/day/month?" into the plain "<count> a week/day/month" phrasing a
# person would actually say.
_TYPICAL_PERIOD_LEAD_RE = re.compile(
    r"(?i)^\s*in\s+a\s+typical\s+(week|day|month|year),?\s*(.*)$"
)
_TYPICAL_PERIOD_TAIL_RE = re.compile(
    r"(?i)^(.*?)\s+in\s+a\s+typical\s+(week|day|month|year)\s*\?\s*$"
)
_PERIOD_NOUN = {"week": "week", "day": "day", "month": "month", "year": "year"}


def _delitteral_typical_period(text: str) -> str:
    """Rewrite "in a typical week/day/month/year" to plain "a week/day/...".

    Moves the period to the END of the sentence rather than dropping it --
    "In a typical week, how many days do you use X?" must keep meaning "out
    of a week", not become the now-ambiguous "how many days do you use X?".
    """
    t = (text or "").strip()
    if not t:
        return text
    m = _TYPICAL_PERIOD_LEAD_RE.match(t)
    if m:
        period, rest = m.group(1), m.group(2).strip()
        if not rest:
            return text
        rest = rest[0].upper() + rest[1:]
        noun = _PERIOD_NOUN[period.lower()]
        if rest.endswith("?"):
            return f"{rest[:-1].rstrip()} a {noun}?"
        return f"{rest} a {noun}"
    m = _TYPICAL_PERIOD_TAIL_RE.match(t)
    if m:
        head, period = m.group(1).rstrip(), m.group(2)
        return f"{head} a {_PERIOD_NOUN[period.lower()]}?"
    return text


# change for b2c questionarie -- CHECK (user directive, 2026-09-08): "What
# usually prompts you to buy X?" shipped live -- "prompts" is not a word a
# 6-year-old uses, even though the underlying prompt guidance already said
# "what specifically made them buy" in plain words. The model substituted a
# fancier synonym on its own; prompt wording alone was not strict enough to
# stop it, matching this session's repeated pattern (Title-Case segment
# leak, "in a typical week") of needing a deterministic backstop for a
# banned-word class rather than trusting compliance. Whole-word,
# case-preserving replacement of known stiff verbs with the plain
# equivalent a person would actually say.
# change for b2c questionarie -- CHECK (user directive, 2026-09-09): the
# CRITICAL LINGUISTIC RULES ban corporate jargon outright. Prompt rules alone
# have repeatedly proven unreliable for banned-word classes this session
# (Title-Case segment leak, "in a typical week", "prompts you to buy"), so
# every jargon word with a clean one-for-one plain swap is also corrected
# here. Deliberately conservative: only substitutions that cannot break
# grammar or change meaning. Words with no safe mechanical swap (catalyst,
# demographics, sentiment, proximity, mitigate) are left to the prompt ban
# and the validator's padding/jargon detectors -- a bad auto-rewrite of those
# would read worse than the original.
# Only substitutions that are grammatically safe in ANY position are listed.
# Verified by test: context-blind swapping of "purchased"->"bought" and
# "approximately"->"about" produced broken output ("Where did you bought...",
# "How much did you spend about?"), so those are excluded and left to the
# prompt ban instead. A wrong auto-rewrite reads worse than the original.
_STIFF_WORD_RE = re.compile(
    r"\b(utilize[sd]?|utilizing|indicate[sd]?|indicating|"
    r"acquire[sd]?|acquiring|commence[sd]?|initiate[sd]?|facilitate[sd]?|"
    r"additional|sufficient|tenure|proximity)\b",
    re.IGNORECASE,
)
_STIFF_WORD_MAP = {
    "utilize": "use", "utilizes": "use", "utilized": "used", "utilizing": "using",
    "indicate": "show", "indicates": "shows", "indicated": "showed", "indicating": "showing",
    "acquire": "get", "acquires": "gets", "acquired": "got", "acquiring": "getting",
    "commence": "start", "commenced": "started",
    "initiate": "start", "initiated": "started",
    "facilitate": "help", "facilitated": "helped",
    "additional": "extra",
    "sufficient": "enough",
    "tenure": "time",
    "proximity": "closeness",
}
# "prompt(s)/prompted you TO buy" needs the "to" dropped when swapped for
# "make(s)/made you buy" -- "make" doesn't take "to" the way "prompt" does
# ("makes you TO buy" is broken English) -- handled as its own pattern
# rather than the generic single-word swap above.
_PROMPTS_YOU_TO_RE = re.compile(
    r"\bprompt(s|ed|ing)?\s+you\s+to\b", re.IGNORECASE,
)
_PROMPTS_YOU_TO_MAP = {"s": "makes", "ed": "made", "ing": "making", "": "make"}


def _plain_word_substitute(text: str) -> str:
    """Swap a small set of known stiff/formal verbs for the plain equivalent."""
    if not text:
        return text

    def _sub_prompts(m: "re.Match") -> str:
        suffix = (m.group(1) or "").lower()
        return f"{_PROMPTS_YOU_TO_MAP[suffix]} you"

    text = _PROMPTS_YOU_TO_RE.sub(_sub_prompts, text)

    def _sub_word(m: "re.Match") -> str:
        word = m.group(0)
        plain = _STIFF_WORD_MAP.get(word.lower())
        if not plain:
            return word
        if word[0].isupper():
            plain = plain[0].upper() + plain[1:]
        return plain

    return _STIFF_WORD_RE.sub(_sub_word, text)


def _to_frontend_question(
    q: dict,
    *,
    primary_region_slug: str | None = None,
    display_n: int | None = None,
    n_seed: str | None = None,
    persona_names: list | None = None,
    segment_names: list | None = None,
) -> dict:
    """Convert one assembled question into the frontend SurveyQuestion shape."""
    answers = q.get("answers") or []
    # change for b2c questionarie — insight stays Top-response for frontend;
    # distributionNote carries the persona/% selection rationale.
    # change for b2c questionarie — A12: never let a panel-persona archetype
    # reach the client document.
    distribution_note = _scrub_persona_names(
        q.get("distribution_note") or "",
        persona_names or [],
        segment_names or [],
    )
    insight = ""
    if answers:
        top = max(answers, key=lambda a: a.get("percentage") or 0)
        pct = float(top.get("percentage") or 0)
        insight = f"Top response: {top.get('label')} ({pct:.2f}%)."

    chart = q.get("_frontend_chart") or _to_chart_kind(q.get("type"), q.get("chart_type"))
    # change for b2c questionarie — A12: a reader could not tell a multi-select
    # from a single pick, so a question summing to 118% looked like an error.
    # The type and its respondent instruction now travel with every question.
    # The pipeline's select-all type is ``multiple_choice`` — it is the only
    # type whose percentages do not sum to 100, so getting this name wrong
    # would mislabel exactly the questions the reader needs the note on.
    qtype = q.get("type") or ""
    _INSTRUCTION = {
        "multiple_choice": "Select all that apply.",
        "ranking": "Rank from most important to least important.",
        "likert_5": "Select one.",
        "likert_7": "Select one.",
        "single_choice": "Select one.",
    }
    out = {
        "id": q.get("id"),
        "qNum": int(str(q.get("id", "q0")).lstrip("q") or 0),
        # change for b2c questionarie -- a safety net for malformed punctuation
        # that survived generation ("...? Select all that apply.?"). Collapse
        # any run of terminal punctuation to a single mark.
        "question": _clean_stem(q.get("text")),
        "questionType": qtype,
        "responseInstruction": _INSTRUCTION.get(qtype, ""),
        # change for b2c questionarie — a ranking's per-item "value" is a mean
        # rank position or top-rank share, not a share of 100% respondents the
        # way single_choice/likert options are. Claiming it sums to 100% mislabels
        # exactly the questions where a reader most needs the correct frame.
        "sumsTo100": qtype not in ("multiple_choice", "ranking"),
        "options": q.get("options") or [],
        "chart": chart,
        "data": [
            {"label": a.get("label"), "value": round(float(a.get("percentage") or 0), 2)}
            for a in answers
        ],
        "insight": insight,
        "distributionNote": distribution_note,
    }
    # change for b2c questionarie — per-question display N (25k–30k)
    if display_n is not None:
        out["N"] = int(display_n)
    elif n_seed is not None:
        out["N"] = _display_n(n_seed)
    # change for b2c questionarie — optional; frontend ignores unknown fields
    if q.get("question_layer"):
        out["questionLayer"] = q.get("question_layer")
    if q.get("is_grounded"):
        out["isGrounded"] = True
    if q.get("funnel_position") is not None:
        out["funnelPosition"] = q.get("funnel_position")
    # change for b2c questionarie — Phase A8: publish the story position so the
    # UI can show the arc and Phase A6 can re-align beats across regions.
    if q.get("narrative_order") is not None:
        out["narrativeOrder"] = q.get("narrative_order")
    if q.get("beat_id"):
        out["beatId"] = q.get("beat_id")
    if q.get("beat_title"):
        out["beatTitle"] = q.get("beat_title")
    if q.get("beat_canonical"):
        out["beatCanonical"] = q.get("beat_canonical")

    llm_regions = q.get("data_by_region") or {}
    labels = [d["label"] for d in out["data"]]

    # change for b2c questionarie — region job: ONLY that region's % (no invented continents)
    if primary_region_slug and primary_region_slug != "global":
        region_vals = llm_regions.get(primary_region_slug)
        if region_vals and len(region_vals) == len(labels):
            out["data"] = [{"label": lb, "value": v} for lb, v in zip(labels, region_vals)]
            if out["data"]:
                out["insight"] = _insight_from_data_local(out["data"]) or out["insight"]
        out["dataByRegion"] = {primary_region_slug: out["data"]}
    else:
        # Global / multi-region jobs: keep real sims; fill gaps with deterministic approx.
        det = _regional_data(out["id"], q.get("type"), out["data"])
        by_region = {"global": out["data"]}
        for slug, det_vals in det.items():
            if slug == "global":
                continue
            vals = llm_regions.get(slug)
            if vals and len(vals) == len(labels):
                by_region[slug] = [{"label": lb, "value": v} for lb, v in zip(labels, vals)]
            elif slug in llm_regions:
                # Only include regions that were actually simulated.
                continue
            else:
                # Skip fabricating continents the job never ran.
                if llm_regions:
                    continue
                by_region[slug] = det_vals
        # Prefer only simulated + global when llm_regions is non-empty.
        if llm_regions:
            by_region = {"global": out["data"]}
            for slug, vals in llm_regions.items():
                if vals and len(vals) == len(labels):
                    by_region[slug] = [{"label": lb, "value": v} for lb, v in zip(labels, vals)]
        out["dataByRegion"] = by_region

    if q.get("type") == "multiple_choice":
        out["multiSelect"] = True
    if q.get("type") in _ORDERED_TYPES:
        out["ordered"] = True
    return out


def _insight_from_data_local(data: list) -> str:
    if not data:
        return ""
    top = max(data, key=lambda d: d.get("value") or 0)
    return f"Top response: {top.get('label')} ({float(top.get('value') or 0):.2f}%)."


def _build_frontend_sections(
    tabs_out: list,
    *,
    blueprint: dict | None = None,
    primary_region_slug: str | None = None,
    display_n: int | None = None,
    n_seed_prefix: str | None = None,
    persona_names: list | None = None,
) -> list:
    """Build the frontend's SurveySection[] from the assembled tabs."""
    # change for b2c questionarie — A12: only the study's own segments may be
    # named in a client-facing note.
    segment_names = [
        s.get("name") for s in narrative.segments_for(blueprint) if s.get("name")
    ]
    sections = []
    for tab_block in tabs_out:
        # change for b2c questionarie — a section may be named for the market
        # ("Streaming Habits"); fall back to the canonical mapping for the
        # standard four so existing payloads are unaffected.
        tab_id = tab_block["tab"]
        if tab_id in _FRONTEND_SECTIONS:
            # Keep the legacy frontend id so older payloads, A6 and the web
            # component are unaffected — but take the LABEL from the blueprint.
            # change for b2c questionarie — the hardcoded pair carried the old
            # generic wording ("Consumer Profile"), which silently overrode the
            # matched CMI framework's category name and left the PDF with no
            # usable section heading.
            sid = _FRONTEND_SECTIONS[tab_id][0]
            label = (
                narrative.section_label(blueprint, tab_id)
                or _FRONTEND_SECTIONS[tab_id][1]
            )
        else:
            # A market-specific section: its own id and its own client-facing name.
            sid = tab_id
            label = narrative.section_label(blueprint, tab_id)
        qs = []
        for q in tab_block["questions"]:
            seed = None
            if n_seed_prefix is not None:
                # change for b2c questionarie — unique N per question id
                seed = f"{n_seed_prefix}:{q.get('id') or q.get('text') or id(q)}"
            qs.append(
                _to_frontend_question(
                    q,
                    primary_region_slug=primary_region_slug,
                    display_n=display_n,
                    n_seed=seed,
                    persona_names=persona_names,
                    segment_names=segment_names,
                )
            )
        sections.append({
            "id": sid,
            "label": label,
            "questions": qs,
        })
    return sections


_GEO_RESIDENCE_RE = re.compile(
    r"(?i)\b("
    r"which\s+region\s+do\s+you\s+(live|reside)|"
    r"in\s+which\s+region\s+do\s+you\s+(currently\s+)?(live|reside)|"
    r"where\s+do\s+you\s+(currently\s+)?(live|reside)|"
    r"where\s+are\s+you\s+(based|located)|"
    r"select\s+your\s+(region|country)"
    r")\b"
)


def _is_geo_residence_question(text: str) -> bool:
    return bool(_GEO_RESIDENCE_RE.search(text or ""))


# change for b2c questionarie — mirror architect bans at assemble time
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


def _is_banned_demographic_or_brand(text: str) -> bool:
    return bool(_AGE_INCOME_RE.search(text or "") or _BRAND_COMPANY_RE.search(text or ""))


_SECTION_COVERS = {
    "consumer_profile": "how they relate to and use the category",
    "buying_behavior": "how they shop for and buy it",
    "preferences_expectations": "what they want from the product",
    "satisfaction_future_intent": "how they feel about it and what they will do next",
}


def _build_survey_definition(
    state: SurveyState,
    target_customer: dict,
    blueprint: dict,
    counts_by_tab: dict,
    total_questions: int,
    completeness_warnings: list | None = None,
) -> dict:
    """Plain-language definition of what this survey is and whose view it captures.

    # change for b2c questionarie — emitted FIRST in the JSON and rendered at the
    # top of the PDF, so a reader knows the context before seeing any question.
    """
    segment = (
        state.get("normalized_segment") or state.get("market_segment") or "the category"
    )
    # change for b2c questionarie — this block writes the title and every
    # "consumers in X" sentence a reader sees first, so it must name the
    # geography the file actually covers. Reading state["region"] directly
    # produced "consumer survey (Europe)" on top of the German instrument.
    region = (
        state.get("country") or state.get("geography_label")
        or state.get("region") or "Global"
    )
    ap = state.get("audience_profile") or {}
    if hasattr(ap, "model_dump"):
        ap = ap.model_dump()
    elif not isinstance(ap, dict):
        ap = {}

    qualifiers = [str(q) for q in (ap.get("qualifiers") or []) if q]
    exclusions = [str(x) for x in (ap.get("exclusions") or []) if x]
    behaviors = [str(b) for b in (ap.get("behaviors") or []) if b]

    who = f"Adults in {region} who buy or use {segment} for personal or household use."
    if qualifiers:
        who += " Respondents qualify if they: " + "; ".join(qualifiers[:5]) + "."
    if exclusions:
        who += " Excluded: " + "; ".join(exclusions[:4]) + "."

    hybrid_note = ""
    if state.get("is_hybrid"):
        hybrid_note = (
            f" {segment} is bought largely through businesses or institutions, so this "
            "survey interviews the consumer end-user (the patient, caregiver or "
            "household member who actually uses it) as a consumer proxy."
        )

    # change for b2c questionarie — the declared structure and the printed
    # section headers must come from ONE source. Writing them independently
    # produced a summary that named "Buying Behavior" while the body printed
    # "Purchase Journey & Decision Drivers", listed 25 questions against a
    # stated total of 29, and omitted the Profiling section entirely even
    # though it appears in the body with four real questions.
    sections = []
    for section in narrative.sections_for(blueprint):
        tab = section["section_id"]
        canon = section.get("canonical") or tab
        sid = _FRONTEND_SECTIONS[tab][0] if tab in _FRONTEND_SECTIONS else tab
        # Take the label the BODY will print, never the legacy generic name.
        label = (
            narrative.section_label(blueprint, tab)
            or (_FRONTEND_SECTIONS[tab][1] if tab in _FRONTEND_SECTIONS else tab)
        )
        sections.append({
            "id": sid,
            "label": label,
            "covers": (section.get("remit") or "").strip() or _SECTION_COVERS.get(canon, ""),
            "questions": counts_by_tab.get(tab, 0),
        })
    # Profiling is part of the instrument the respondent answers, so it belongs
    # in the count and in the structure summary.
    prof_id = standard_sections.PROFILING_SECTION_ID
    if counts_by_tab.get(prof_id):
        sections.append({
            "id": prof_id,
            "label": standard_sections.PROFILING_LABEL,
            "covers": ("Classification questions used only to cross-tabulate the "
                       "behavioural findings; asked last and never used to "
                       "qualify respondents."),
            "questions": counts_by_tab.get(prof_id, 0),
        })

    result = {
        "title": f"{segment} — consumer survey ({region})",
        "category": segment,
        "region": region,
        "audienceType": "B2C consumer" + (" (consumer proxy)" if state.get("is_hybrid") else ""),
        "studyPurpose": narrative.study_purpose(blueprint),
        "segments": [
            {"id": sg.get("segment_id"), "name": sg.get("name"),
             "description": _scrub_currency_amount(sg.get("description"))}
            for sg in narrative.segments_for(blueprint)
        ],
        # change for b2c questionarie — A13: "A business-to-consumer (B2C) market
        # research questionnaire about X" is internal classification language on
        # the first line a client reads. Say what the study does instead.
        "whatThisSurveyIs": (
            f"A segmentation study of {segment} buyers in {region}. It asks how "
            f"people use {segment}, how they choose it, and what would make them "
            "switch — then sorts respondents into the behavioural groups defined "
            "below. Questions stay at category level: no named brands, and "
            "nothing about adjacent categories." + hybrid_note
        ),
        "whoseContextThisIs": who,
        "whyTheseRespondents": (
            (state.get("classification_reason") or "").strip()
            or f"{segment} is bought and used by individual consumers."
        ),
        "respondentBehaviours": behaviors[:6],
        # change for b2c questionarie — A12: this list used to claim age and
        # income were not asked at all, while the Profiling section asked both.
        # A reader spots that contradiction immediately and stops trusting the
        # rest of the document. The rule the instrument actually follows is
        # narrower — and stating it accurately is stronger than the false claim.
        "whatIsDeliberatelyNotAsked": [
            "Age or household income as behavioural questions — no finding in this "
            "survey is driven by them. They are asked once, at the end, in "
            "Profiling, and used only to cross-tabulate the behavioural results.",
            "Brand, company or manufacturer names — the survey stays at category level "
            "so findings are not tied to one competitor set.",
            "Where the respondent lives — region is chosen when viewing the results, "
            "so asking it again would waste a question.",
        ],
        # change for b2c questionarie — describe what the numbers ARE, not the
        # mechanics of a panel that was never fielded. "Persona-weighted panel
        # of 2,500 synthetic respondents" both leaks an internal device and
        # reads as a claim about how the study was run.
        "howToReadTheNumbers": (
            "Percentages are modelled estimates built from published market "
            "evidence — they are not measurements from fielded interviews, and "
            "should be read as an expected shape of response rather than a "
            "result. Single-choice and rating-scale questions sum to "
            "100%. Multi-select questions do not — each option is an "
            "independent share of respondents choosing it. The N shown on each "
            "question is the display base for that item."
        ),
        "structure": {
            "totalQuestions": total_questions,
            "sections": sections,
            "storyline": (
                "Questions run in a planned narrative order: broad and easy first, "
                "specific and sensitive last."
            ),
            "storylineSummary": (blueprint.get("narrative_summary") or "").strip(),
        },
    }
    # change for b2c questionarie -- CHECK (Section 28a): a "revision cap
    # reached, shipped best-effort" warning used to live only in
    # survey.metadata, a backend block a reviewer would have to open raw
    # JSON to find. surveyDefinition is the document's own docstring-stated
    # purpose: "emitted FIRST in the JSON and rendered at the top of the
    # PDF, so a reader knows the context before seeing any question" -- a
    # completeness warning belongs exactly there, not buried in metadata a
    # reviewer never opens.
    if completeness_warnings:
        result["completenessWarnings"] = list(completeness_warnings)
    return result


def _scope_demographics(demographics, region: str) -> dict:
    """Force every geography-bearing value onto this file's region.

    # change for b2c questionarie — Issue 2. Any wording the model produced for
    # a different scope ("Global", another region) is replaced; the rest of the
    # block (age, gender, income descriptors) is passed through untouched.
    """
    out = dict(demographics) if isinstance(demographics, dict) else {}
    if not out:
        return out
    for key in list(out):
        if "geograph" in key.lower() or "region" in key.lower():
            out[key] = region
    return out


def _build_target_customer(state: SurveyState) -> dict:
    """Define who this survey interviews — emitted at the top of the JSON.

    # change for b2c questionarie
    """
    segment = state.get("normalized_segment") or state.get("market_segment") or "the category"
    # change for b2c questionarie — the target-customer block says "Adults in X
    # who buy…", so X must be the geography this file covers, not its parent.
    region = (
        state.get("country") or state.get("geography_label")
        or state.get("region") or "Global"
    )
    ap_raw = state.get("audience_profile") or {}
    if hasattr(ap_raw, "model_dump"):
        ap = ap_raw.model_dump()
    elif isinstance(ap_raw, dict):
        ap = ap_raw
    else:
        ap = {}

    demographics = ap.get("demographics") or {}
    behaviors = list(ap.get("behaviors") or [])
    qualifiers = list(ap.get("qualifiers") or [])
    exclusions = list(ap.get("exclusions") or [])
    rationale = (ap.get("rationale") or "").strip()
    assumptions = list(ap.get("assumptions") or [])

    demo_bits = []
    if isinstance(demographics, dict):
        for k, v in demographics.items():
            if v is None or v == "" or v == []:
                continue
            demo_bits.append(f"{k}: {v}" if not isinstance(v, (list, dict)) else f"{k}: {v}")

    summary_parts = [
        f"Adult consumers of {segment} in {region}",
    ]
    if qualifiers:
        summary_parts.append("who " + "; ".join(str(q) for q in qualifiers[:4]))
    summary = " — ".join(summary_parts) + "."
    if rationale:
        summary = f"{summary} {rationale}".strip()

    return {
        "summary": summary,
        "marketSegment": segment,
        "region": region,
        "inclusionCriteria": qualifiers,
        "behaviors": behaviors,
        "exclusions": exclusions,
        # change for b2c questionarie — the LLM writes this block and kept
        # returning "geography: Global, with emphasis on urban and suburban
        # areas" inside a file whose region is India. Two different geographies
        # stated in one document is a contradiction the reader cannot resolve,
        # so the geography key is overwritten from the file's own region rather
        # than trusted from the model.
        "contextDemographics": _scope_demographics(demographics, region),
        "assumptions": assumptions,
        "note": (
            "Respondents match this target customer. Age, income, and brand/"
            "company-name items are NOT asked in the questionnaire; they may "
            "appear here only as research context."
        ),
    }


# change for b2c questionarie — quota NEVER protects a duplicate. Every extra
# flagged by the validator is dropped, hard or soft, even if that takes a tab
# under target. A thin section is recoverable; a repeated question is not.
def _drop_duplicates(answered: list, duplicates: list, hard_duplicates: list | None = None) -> list:
    """Drop every duplicate-flagged extra, plus all banned screeners.

    The first id in each cluster is the representative and is kept; every other
    member is removed unconditionally. Geography-of-residence / age / income /
    brand-name items are always dropped too.
    """
    by_id = {q.get("id"): q for q in answered}
    drop = set()

    for q in answered:
        text = q.get("text") or ""
        # change for b2c questionarie — Screening/Profiling carry demographics by
        # design; never strip them here.
        if q.get("question_layer") in ("screening", "profiling"):
            continue
        if _is_geo_residence_question(text) or _is_banned_demographic_or_brand(text):
            drop.add(q.get("id"))

    for cluster in list(duplicates or []) + list(hard_duplicates or []):
        for dup in cluster[1:]:
            q = by_id.get(dup)
            if not q:
                continue
            # change for b2c questionarie — Screening and Profiling are a fixed
            # templated set. The three screeners deliberately share phrasing, so
            # the validator clusters them; dropping one breaks qualification.
            if q.get("question_layer") in ("screening", "profiling"):
                continue
            drop.add(dup)

    return [q for q in answered if q.get("id") not in drop]


# change for b2c questionarie — absolute uniqueness: nothing repeated, ever.
# Reworded-duplicate detection: near-identical option sets only count as a
# repeat when the wording also overlaps, so shared scales don't cause false
# positives that would silently delete a good question.
_SAME_OPTIONS_JACCARD = 0.85
_REWORD_TEXT_FLOOR = 0.45

# change for b2c questionarie — same-beat duplicates.
# Two questions tagged to the SAME storyline beat are covering the same
# narrative moment by construction, so they only earn their place if they probe
# different facets of it. Measured across 127 same-beat pairs in 20 published
# surveys the option-token overlap runs median 0.08 / p75 0.14. Every pair above
# 0.20 that also shared some wording turned out to be a genuine restatement on
# inspection ("what most often TRIGGERS you to buy" vs "what most often PROMPTS
# you to buy"; "on which OCCASIONS" vs "in which SITUATIONS") with no false
# positives, so the bar sits just above the p75 of ordinary same-beat pairs.
_SAME_BEAT_OPTIONS = 0.20
_SAME_BEAT_TEXT = 0.18


def _option_token_overlap(a_opts: list, b_opts: list, segment: str) -> float:
    """Jaccard over the WORDS used across two option sets.

    ``option_jaccard`` compares whole option strings, so "worn out or no longer
    comfortable" and "worn out or damaged" score as completely different. At
    token level the shared vocabulary shows up.
    """
    seg = _sim_tokens(segment)

    def bag(opts: list) -> set:
        out: set = set()
        for o in opts or []:
            out |= _sim_tokens(str(o))
        return out - seg

    ta, tb = bag(a_opts), bag(b_opts)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _enforce_absolute_uniqueness(
    answered: list, segment: str, blueprint: dict | None = None,
) -> tuple[list, list[str]]:
    """Final gate: no two questions in the published survey may overlap.

    Walks the survey in reading order and keeps the FIRST question of any
    similar group, dropping every later one — across tabs, not just within a
    tab. Quota never protects a repeat here: a short section is recoverable, a
    duplicated question is not.

    Returns ``(kept, notes)``.
    """
    section_order = narrative.section_ids(blueprint)
    tab_rank = {tab: i for i, tab in enumerate(section_order)}

    def _pos(q: dict) -> tuple:
        try:
            order = int(q.get("narrative_order") or q.get("funnel_position") or 999)
        except (TypeError, ValueError):
            order = 999
        return (tab_rank.get(q.get("tab"), 99), order, str(q.get("id") or ""))

    kept: list[dict] = []
    notes: list[str] = []
    seen_exact: set[str] = set()

    for q in sorted(answered, key=_pos):
        text = q.get("text") or ""
        options = q.get("options") or []
        key = norm_alnum(text)

        # change for b2c questionarie — Screening and Profiling are a fixed,
        # templated set. They are methodological, not findings: the three
        # screeners deliberately share phrasing ("do you / does anyone in your
        # household…") and dropping one breaks qualification. Never dedup them.
        if q.get("question_layer") in ("screening", "profiling"):
            kept.append(q)
            continue

        if key and key in seen_exact:
            notes.append(f"duplicate question removed: {text[:70]}")
            continue

        clash = None
        for prior in kept:
            p_text = prior.get("text") or ""
            p_opts = prior.get("options") or []
            if is_near_exact(text, p_text, options, p_opts, segment=segment):
                clash = prior
                break
            if token_set_jaccard(text, p_text, segment) >= LOOSE_JACCARD:
                clash = prior
                break
            # Same option set + same answer format + materially overlapping
            # wording = one question asked twice. All three conditions are
            # required: two genuinely different Likert items legitimately share
            # an identical scale, and two frequency items legitimately share
            # Weekly/Monthly/Rarely/Never.
            if (
                len(options) >= 3
                and prior.get("type") == q.get("type")
                and option_jaccard(options, p_opts) >= _SAME_OPTIONS_JACCARD
                and token_set_jaccard(text, p_text, segment) >= _REWORD_TEXT_FLOOR
            ):
                clash = prior
                break
            # change for b2c questionarie — A14: the same option list offered
            # twice is the same question twice, even when the stems are worded
            # differently and the answer format differs. "Which aspects most
            # affect your satisfaction" (select-all) beside "how much does each
            # factor affect your satisfaction" (single pick) shares its whole
            # option set — asking it both ways spends two slots on one reading.
            # A shared SCALE is exempt: two different likert items legitimately
            # offer the same five points. Restricted to questions filling the
            # SAME beat — two beats can legitimately reuse a generic option list
            # ("A/B/C/Other") while asking genuinely different things, and
            # dropping those would silently shorten the survey.
            if (
                len(options) >= 4
                and q.get("beat_id")
                and q.get("beat_id") == prior.get("beat_id")
                and q.get("type") not in ("likert_5", "likert_7")
                and prior.get("type") not in ("likert_5", "likert_7")
                and option_jaccard(options, p_opts) >= _SAME_OPTIONS_JACCARD
            ):
                clash = prior
                break
            # change for b2c questionarie — same storyline beat: the bar drops,
            # because a second question in one beat must add a NEW facet, and a
            # restatement of the same facet is a repeat however it is worded.
            beat = q.get("beat_id")
            if (
                beat
                and beat == prior.get("beat_id")
                and _option_token_overlap(options, p_opts, segment) >= _SAME_BEAT_OPTIONS
                and token_set_jaccard(text, p_text, segment) >= _SAME_BEAT_TEXT
            ):
                clash = prior
                break
        if clash is not None:
            notes.append(
                f"duplicate question removed ({q.get('id')} overlaps "
                f"{clash.get('id')}): {text[:70]}"
            )
            continue

        if key:
            seen_exact.add(key)
        kept.append(q)

    return kept, notes


_PERSONA_SUFFIX_WORDS = (
    "runners", "buyers", "wearers", "consumers", "users", "shoppers",
    "seekers", "adopters", "enthusiasts", "loyalists", "minimalists",
    "traditionalists", "explorers", "novices", "beginners", "professionals",
    "experts", "purists", "skeptics", "advocates", "switchers", "holdouts",
    "drivers", "owners", "customers", "spenders", "planners",
)
_PERSONA_LIKE_PHRASE_RE = re.compile(
    r"\b((?:[A-Z][a-zA-Z]*(?:-[A-Z][a-zA-Z]*)*\s+){1,3}"
    r"(?:" + "|".join(w.capitalize() for w in _PERSONA_SUFFIX_WORDS) + r")s?)\b"
)


def _fabricated_persona_issue(text: str, segment_names: list) -> str | None:
    """Flag a Title-Case, persona-suffixed phrase that names NONE of the
    survey's real segments and none of its known archetypes.

    CHECK (Section 30a): _scrub_persona_names below can only strip a name
    that matches a KNOWN archetype from persona_names -- it has no defence
    against a name the model fabricated on the spot that matches neither the
    real segments nor any known archetype. Confirmed live: a rendered
    takeaway named "Health-Focused Buyers", a group that does not exist
    anywhere in the survey's actual persona/segment data. This check runs
    independently of persona_names and catches ANY unrecognised persona-
    shaped phrase, not just previously-seen ones.
    """
    kept = {s.lower() for s in segment_names if s}
    for m in _PERSONA_LIKE_PHRASE_RE.finditer(text or ""):
        phrase = m.group(1).strip()
        if phrase.lower() not in kept:
            return phrase
    return None


def _scrub_persona_names(note: str, persona_names: list, segment_names: list) -> str:
    """Remove panel-persona archetypes from a client-facing distribution note.

    # change for b2c questionarie — A12

    The simulator reasons over a weighted persona mix ("Budget Binger",
    "Eco-Conscious"). Those are an internal device: the reader has never seen
    them and the deliverable never defines them, so a note naming them reads as
    boilerplate lifted from a generic persona library. The prompt now asks for
    the study's own segments; this makes sure a slip cannot reach the document.

    A note that is only about personas is dropped rather than half-rewritten —
    a stray fragment is worse than no note.
    """
    text = (note or "").strip()
    if not text:
        return text

    # CHECK (Section 30a) -- run the fabricated-name check BEFORE the
    # early-return-on-empty-persona_names path below, since a hallucinated
    # name needs no persona_names list to be dangerous.
    fabricated = _fabricated_persona_issue(text, segment_names)
    if fabricated:
        sentences = re.split(r"(?<=[.;])\s+", text)
        text = " ".join(
            s.strip() for s in sentences
            if fabricated.lower() not in s.lower()
        ).strip()
        text = re.sub(r"[;,]\s*$", ".", text)
        if text and (text[0].islower() or re.match(r"(?i)^(also|and|but|while|whereas)\b", text)):
            return ""
        if not text:
            return ""

    if not text or not persona_names:
        return text

    kept = {s.lower() for s in segment_names if s}
    segment_words = {w for s in segment_names for w in re.split(r"[^\w-]+", s.lower()) if w}

    # Match fragments as well as whole names. A persona called "Eco-Conscious
    # Sustainability Seeker" leaks as "Eco-conscious viewers", which a
    # full-name match never catches. Any distinctive multi-word or hyphenated
    # element of a persona name is treated as the persona.
    needles: set[str] = set()
    for p in persona_names:
        if not p or p.lower() in kept:
            continue
        needles.add(p)
        for part in re.split(r"\s+", p):
            # Hyphenated qualifiers ("Eco-Conscious", "Tech-Savvy") are the
            # give-away tokens; plain words like "Buyers" are not.
            if "-" in part and len(part) > 5 and part.lower() not in segment_words:
                needles.add(part)

    hits = [
        n for n in needles
        if re.search(rf"\b{re.escape(n)}s?\b", text, re.I)
    ]
    if not hits:
        return text
    # Drop any sentence that leans on a persona name.
    sentences = re.split(r"(?<=[.;])\s+", text)
    clean = [
        s for s in sentences
        if not any(re.search(rf"\b{re.escape(p)}s?\b", s, re.I) for p in hits)
    ]
    out = " ".join(x.strip() for x in clean if x.strip()).strip()
    # Removing the trailing clause of "A pulls X; B pulls Y" leaves a dangling
    # semicolon, which reads as a truncated sentence in the deliverable.
    out = re.sub(r"[;,]\s*$", ".", out)
    # change for b2c questionarie — A14: dropping the FIRST sentence can leave a
    # remainder that starts mid-thought ("also like simplified billing", "the
    # distribution is skewed toward..."). A note that opens lowercase or on a
    # connective reads as truncated text, which is worse than no note.
    if out and (out[0].islower() or re.match(r"(?i)^(also|and|but|while|whereas)\b", out)):
        return ""
    if out and not out.endswith((".", "!", "?")):
        out += "."
    return out


# change for b2c questionarie — A16: the padded-opener rule is stated in the
# prompt and enforced by the validator, but the revision loop is capped and the
# formula still reached publication on 7-8 questions in one survey. These are
# meaning-preserving rewrites, applied last, so the deliverable cannot ship the
# construction however the loop behaves.
_STEM_TIGHTEN = (
    # "Which of these best describes how you pay for X?" -> "How do you pay for X?"
    (re.compile(r"(?i)^which\s+(?:of\s+(?:these|the\s+following)\s+)?best\s+"
                r"describes\s+(how|where|when|why|who)\s+(.+?)\?*$"),
     # The promoted clause is declarative ("you pay for X"), so it needs the
     # auxiliary a direct question carries: "How DO you pay for X?".
     lambda m: (
         f"{m.group(1).capitalize()} "
         + (f"do {m.group(2)}" if re.match(r"(?i)^you\b", m.group(2))
            else m.group(2))
         + "?"
     )),
    # "Which of these best describes your household?" -> "How would you describe
    # your household?" — no wh-word to promote, so keep it grammatical.
    (re.compile(r"(?i)^which\s+(?:of\s+(?:these|the\s+following)\s+)?best\s+"
                r"describes\s+(.+?)\?*$"),
     lambda m: f"How would you describe {m.group(1)}?"),
    # "Which of these would you say ..." -> "Which ..."
    (re.compile(r"(?i)^which\s+of\s+these\s+would\s+you\s+say\s+(.+?)\?*$"),
     lambda m: f"Which {m.group(1)}?"),
    # "Which of these X" -> "What X" / "Which X". The options are printed
    # directly beneath the stem, so "of these" says nothing — it is pure
    # padding, and it opened seven questions in one survey.
    (re.compile(r"(?i)^which\s+of\s+these\s+(.+?)\?*$"),
     lambda m: (
         ("What " if re.match(
             r"(?i)^(would|will|is|are|was|were|do|does|did|has|have|had|can|"
             r"could|should|matters?|most|best|comes?|makes?|describes?)\b",
             m.group(1)) else "Which ")
         + m.group(1) + "?"
     )),
)


def _tighten_stem(text: str) -> str:
    """Strip a padded opener while preserving the question's meaning."""
    t = (text or "").strip()
    for pattern, rewrite in _STEM_TIGHTEN:
        m = pattern.match(t)
        if m:
            out = rewrite(m).strip()
            # "How do you pay" needs the auxiliary the original carried; if the
            # promoted clause already starts with one, leave it alone.
            out = re.sub(r"\s+", " ", out)
            return out[0].upper() + out[1:] if out else t
    return t


# CHECK (manager review, 2026-08-25 — China Sport Shoes) -- exact wording
# swap requested for 26 specific questions, moving from formal/survey-style
# phrasing to short conversational phrasing. Keyed on the OLD text with
# whitespace/case normalized, so this catches the exact phrasing whenever
# the generator reproduces it (segment-level, category-general questions
# recur across runs of similar categories) without depending on the LLM
# prompt alone remembering the new style -- same "detection backstop, not
# just a prompt ask" pattern used throughout this pipeline.
_MANAGER_WORDING_REWRITES = {
    "which activity do you use your sport shoes for most?":
        "What do you mainly use your sports shoes for?",
    "how attached do you feel to a particular brand for sport shoes?":
        "Do you usually stick to the same sports shoe brand?",
    "how often do you use your sport shoes for sport or exercise?":
        "How often do you wear sports shoes for exercise or sports?",
    "which best describes the fit you prefer in a sport shoe?":
        "What type of fit do you prefer in sports shoes?",
    "does the weather or season affect which sport shoes you wear?":
        "Does the weather affect which sports shoes you wear?",
    "how important is it that one pair works for more than one activity?":
        "Do you prefer sports shoes that can be used for different activities?",
    "what made you buy your most recent pair of sport shoes?":
        "Why did you buy your last pair of sports shoes?",
    "where did you buy your most recent pair?":
        "Where did you buy your sports shoes?",
    "where do you usually see sport shoe options before you buy?":
        "Where do you usually look for sports shoes before buying?",
    "how often do you replace your sport shoes?":
        "How often do you buy new sports shoes?",
    "how much did you pay for your most recent pair?":
        "How much did you spend on your last pair of sports shoes?",
    "which single factor has the biggest influence on your final choice?":
        "What matters most when choosing sports shoes?",
    "which of these must a sport shoe have for you to consider buying it?":
        "What features must sports shoes have for you to buy them?",
    "what makes you feel a pair of sport shoes is well made?":
        "What makes sports shoes feel high quality to you?",
    "how long do you think a good pair of sport shoes should last before "
    "wearing out?":
        "How long should a good pair of sports shoes last?",
    "which sustainability feature would matter most when choosing between "
    "similar shoes?":
        "Which eco-friendly feature matters most to you when choosing "
        "sports shoes?",
    "if you had to pay less, what would you give up first?":
        "What would you compromise on to pay a lower price?",
    "how well do your sport shoes usually perform compared to what you "
    "expected when you bought them?":
        "Do your sports shoes usually perform as well as you expected?",
    "overall, how satisfied are you with your current sport shoes?":
        "How satisfied are you with your current sports shoes?",
    "if you had a problem with your sport shoes, what would you most "
    "likely do?":
        "What would you do if you had a problem with your sports shoes?",
    "what would most likely make you switch to a different sport shoe?":
        "What would make you switch to different sports shoes?",
    "what would most encourage you to buy your next pair sooner rather "
    "than later?":
        "What would make you buy your next pair sooner?",
    "how interested would you be in a subscription that sends you a new "
    "pair every few months?":
        "Would you be interested in getting new sports shoes through a "
        "subscription?",
    "how likely are you to recommend your current sport shoes to a "
    "friend or family member?":
        "Would you recommend your current sports shoes to others?",
    "which age group are you in?":
        "What is your age group?",
    "who else lives in your household?":
        "Who do you live with?",
    "what was your total household income before tax last year?":
        "What is your household income range?",
}


# change for b2c questionarie -- two of the 27 exact-match entries above
# turned out to have LLM-generated near-variants on the very next run
# ("If you had to CUT ONE THING to pay less..." vs the mapped "If you had
# to pay less..."; the hardcoded NPS anchor's wording drifted before this
# fix, now fixed at its template source separately). Exact-match can never
# cover every paraphrase the model might produce -- these two narrow regex
# patterns catch the STIFF trade-off/recommend CONSTRUCTS regardless of
# minor wording variance, complementing (not replacing) the exact table.
# change for b2c questionarie -- widened after "If you had to give up one
# thing to pay less for X, what would it be?" slipped past a sequence-
# anchored pattern that required "pay less" to come immediately before
# "what would you give up" -- same lesson as fit_difficulty earlier this
# session: match the CONCEPTS anywhere in the sentence (lookahead-based),
# not a fixed clause order, so wording variance around the same construct
# doesn't defeat the match.
_TRADE_OFF_STIFF_RE = re.compile(
    r"(?i)^if you had to.*"
    r"(?=.*\bpay less\b)"
    r"(?=.*\b(give up|compromise|sacrifice)\b)"
)
_MANAGER_WORDING_PATTERNS = (
    (_TRADE_OFF_STIFF_RE, "What would you compromise on to pay a lower price?"),
    (re.compile(r"(?i)^how likely are you to recommend .{0,60}to a friend or "
                r"family member\??$"),
     lambda m: re.sub(
         r"(?i)^how likely are you to recommend (.{0,60}?) to a friend or "
         r"family member\??$",
         r"Would you recommend \1 to others?",
         m.string,
     )),
)


def _apply_manager_wording_rewrite(text: str) -> str:
    """Swap to the manager-approved conversational phrasing.

    Two layers: an exact-match table (the 27 confirmed pairs, matched
    ignoring case/whitespace) and a small set of regex patterns for
    constructs already observed to drift in wording between generations.
    """
    t = (text or "").strip()
    key = re.sub(r"\s+", " ", t).lower()
    if key in _MANAGER_WORDING_REWRITES:
        return _MANAGER_WORDING_REWRITES[key]
    for pattern, replacement in _MANAGER_WORDING_PATTERNS:
        m = pattern.match(t)
        if m:
            return replacement if isinstance(replacement, str) else replacement(m)
    return t


# CHECK (B2C master rules, Section 28c) -- validator_critic.py's redundancy
# detectors run DURING the revision loop, before the segment scoring model
# exists (segment_scorer runs only after the loop concludes, right before
# this assembler). So the nonDiscriminatingQuestions list -- itself only
# computable at this late stage -- can never be fed back INTO an earlier
# revision pass; the loop has already capped out by the time this runs.
# Confirmed live: two non-discriminating questions (a confidence scale and a
# frequency-of-trouble scale, both measuring fit-finding difficulty) survived
# every revision pass unmerged. Since a fully automatic merge here risks
# removing content a reviewer might actually want kept, this only FLAGS a
# high-confidence pair for review (matching 28a's completeness-warning
# mechanism) rather than silently dropping a question post-hoc.
_NON_DISCRIMINATING_PAIR_JACCARD = 0.30  # looser than LOOSE_JACCARD -- both
# questions in the pair have already passed every earlier redundancy check
# on their own, so a real duplicate here is likely worded further apart than
# a typical caught duplicate; this only needs to beat "clearly unrelated".


def _non_discriminating_duplicate_warnings(non_discriminating: list, segment: str) -> list:
    """Flag high-confidence duplicate pairs within the non-discriminating set.

    Both members of a flagged pair have already been shown (by
    nonDiscriminatingQuestions itself) to carry zero segmentation value, so
    there is nothing to lose by merging one into the other -- unlike a
    general redundancy call, which must weigh whether each question still
    supports a distinct business decision.
    """
    warnings = []
    for i in range(len(non_discriminating)):
        for j in range(i + 1, len(non_discriminating)):
            a, b = non_discriminating[i], non_discriminating[j]
            sim = token_set_jaccard(a.get("question") or "", b.get("question") or "", segment)
            if sim >= _NON_DISCRIMINATING_PAIR_JACCARD:
                warnings.append(
                    f"{a.get('id')} and {b.get('id')} both carry no "
                    "segmentation signal (see nonDiscriminatingQuestions) "
                    "and appear to measure a similar construct -- "
                    f'"{(a.get("question") or "")[:60]}" vs '
                    f'"{(b.get("question") or "")[:60]}". Review for merging '
                    "into one question."
                )
    return warnings


def _non_discriminating(frontend: dict, model: dict) -> list:
    """Questions no scoring weight touches — they separate nobody.

    # change for b2c questionarie — A16

    This is a segmentation study: a question earns its slot by pushing
    respondents toward one group rather than another. If the scoring model
    assigns no points to ANY of a question's options, it contributes nothing to
    the assignment and is context-setting at best. Profiling is excluded by
    design — it exists for cross-tabs, not for scoring.

    Reported rather than dropped: removing them would shorten the instrument
    below its target, and a couple of warm-up questions are legitimate. Naming
    them lets a reviewer judge, instead of guessing which questions are filler.
    """
    scored = {w.get("question_id") for w in (model.get("weights") or [])}
    if not scored:
        return []
    out = []
    for sec in frontend.get("sections") or []:
        for q in sec.get("questions") or []:
            if q.get("questionLayer") == "profiling":
                continue
            if q.get("id") not in scored:
                out.append({
                    "id": q.get("id"),
                    "qNum": q.get("qNum"),
                    "question": q.get("question"),
                    # change for b2c questionarie -- CHECK (Section 28h):
                    # tab id, so a later cross-check can tell whether an
                    # under-target section already has a non-discriminating
                    # question it could replace instead of only adding more.
                    "sectionId": sec.get("id"),
                })
    return out


def _remap_segment_model(model: dict, id_remap: dict) -> dict:
    """Point the scoring weights at the published question ids.

    # change for b2c questionarie — A11

    ``segment_scorer`` runs before questions are renumbered, so its weights key
    on the architect's working ids (``cp1``, ``bb3``). Published questions are
    ``q1..qN``. Without this translation every weight references an id the
    reader cannot find, and the scoring model is decorative.

    A weight whose question was dropped by dedup has no published id and is
    removed — it can no longer be scored.
    """
    if not model or not id_remap:
        return model or {}

    out = dict(model)
    kept, orphaned = [], 0
    for w in model.get("weights") or []:
        new_id = id_remap.get(str(w.get("question_id")))
        if not new_id:
            orphaned += 1
            continue
        kept.append({**w, "question_id": new_id})
    out["weights"] = kept
    if orphaned:
        out["notes"] = [
            *(out.get("notes") or []),
            f"{orphaned} scoring weight(s) removed: they scored questions that "
            "were dropped as duplicates before publication.",
        ]
    return out


# =============================================================================
# NEW DELIVERY FORMAT — B2C_MASTER_RULES_FINAL.txt / JSON_FORMAT_SPECIFICATION.txt
# Top level: surveyScope, behaviouralPersonas, structure, segments. No internal
# generation metadata (warnings, nonDiscriminatingQuestions, grounded_pct,
# revision_count, etc.) belongs in the delivered file -- those stay INTERNAL,
# computed and used upstream in build_survey_json (dedup, currency scrub,
# persona scrub, best-pass restore all still run exactly as before), never
# serialized. This function only shapes the final, already-cleaned data.
# =============================================================================

_DELIVERY_TYPE_MAP = {
    "single_choice": "single_select",
    "multiple_choice": "multi_select",
    # change for b2c questionarie -- likert_5/likert_7 are labeled N-point
    # scales (e.g. Very dissatisfied..Very satisfied), NOT a 0-10 numeric
    # rating -- rating_0_10 does not fit this shape. Mechanically this is
    # still a single-pick-one-of-N question (percentages sum to 100, order
    # preserved), so it maps to single_select with zero data loss.
    "likert_5": "single_select",
    "likert_7": "single_select",
}

# change for b2c questionarie -- the one genuine 0-10 numeric scale in this
# pipeline is the NPS recommend-question, currently typed single_choice with
# 11 options ("0 - Not at all likely" .. "10 - Extremely likely"). Detected
# by shape (11 options, first/last option starting with "0"/"10"), not by
# text pattern, so it is not tied to the exact question wording.
def _is_rating_0_10(qtype: str, options: list) -> bool:
    if qtype != "single_choice" or len(options) != 11:
        return False
    first, last = str(options[0] or ""), str(options[-1] or "")
    return first.strip().startswith("0") and last.strip().startswith("10")


def _delivery_question_type(qtype: str, options: list) -> str:
    if _is_rating_0_10(qtype, options):
        return "rating_0_10"
    return _DELIVERY_TYPE_MAP.get(qtype, "single_select")


_DELIVERY_STANDARD_SEGMENT_TITLES = (
    "Consumer Profile, Ownership & Usage Behaviour",
    "Purchase Journey & Decision Drivers",
    "Product Experience & Expectations",
    "Unmet Needs, Switching & Future Purchase Intent",
)


def _delivery_option_pct(a: dict) -> float:
    return round(float(a.get("percentage") or 0.0), 2)


def _build_delivery_json(
    *,
    state: SurveyState,
    blueprint: dict,
    tabs_out: list,
    region_name: str,
    segment_label: str,
    archetype_persona_names: list,
) -> dict:
    """Build the new-spec delivery JSON directly from cleaned, ordered tabs_out.

    tabs_out is already deduped, currency-scrubbed (upstream in
    build_survey_json), ordered, and id-renumbered -- this function only
    reshapes it into surveyScope/behaviouralPersonas/structure/segments.
    """
    segment_names = [
        sg.get("name") for sg in narrative.segments_for(blueprint) if sg.get("name")
    ]

    # ---- surveyScope -------------------------------------------------------
    definition_text = (
        f"This survey looks at how customers in {region_name} use, choose and "
        f"feel about {segment_label} — covering ownership and usage, how they "
        "shop and decide, what they expect from the product, and what would "
        "make them switch. Responses are grouped into four behavioural "
        "profiles, described below, to show how different types of buyers "
        "think and act differently within the same category."
    )
    ap = state.get("audience_profile") or {}
    if hasattr(ap, "model_dump"):
        ap = ap.model_dump()
    elif not isinstance(ap, dict):
        ap = {}
    qualifiers = [str(q) for q in (ap.get("qualifiers") or []) if q][:5]
    if not qualifiers:
        qualifiers = [
            f"Owns or has bought {segment_label} for personal use",
            f"Based in {region_name}",
            f"Makes their own {segment_label} purchase decisions",
            "Aged 18 or older",
        ]
    survey_scope = {
        "title": f"{segment_label} — Consumer Survey ({region_name})",
        "category": segment_label,
        "region": region_name,
        "audienceType": "B2C Consumer",
        "definition": definition_text,
        "customersWeSurveyed": qualifiers,
        "whatIsNotAsked": [
            "Age or household income as behavioural questions — asked once, "
            "at the end, in Profiling, and used only to cross-tabulate the "
            "behavioural results, never to drive a finding.",
            "Brand, company or manufacturer names — the survey stays at "
            "category level so findings are not tied to one competitor set.",
            f"Anything about adjacent categories outside {segment_label}.",
        ],
        "howToReadTheNumbers": (
            "Percentages are modelled example response shapes for "
            "survey-design review, not measurements from fielded "
            "interviews. Single-choice and rating-scale questions sum to "
            "100%. Multi-select questions do not — each option is an "
            "independent share of customers choosing it."
        ),
    }

    # ---- behaviouralPersonas ------------------------------------------------
    behavioural_personas = [
        {
            "name": sg.get("name"),
            "description": _scrub_currency_amount(sg.get("description") or ""),
        }
        for sg in narrative.segments_for(blueprint)
    ][:4]

    # ---- cap behavioural section question counts ---------------------------
    # change for b2c questionarie -- CHECK (live, Sport Shoes/China,
    # 2026-09-06): the 4 behavioural sections can land at different question
    # counts (e.g. 5/6/5/7) because each tab is generated independently and
    # "honest mode"/dedup can accept fewer than the target in one section
    # without affecting another.
    #
    # First attempt equalized every section DOWN to the smallest section's
    # count. Confirmed live on Wireless Earbuds/U.S. this is actively
    # harmful, not just cosmetic: one section generated only 3 strong
    # questions (validator score never cleared 0.43), so trimming dragged
    # sections that had 6-8 good, unique questions down to 3 as well --
    # deleting the survey's ONLY price/spend question in the process and
    # leaving a 16-question instrument with no purchase-price coverage at
    # all. Forcing equality is worse than leaving sections uneven.
    #
    # User's direction (2026-09-07): no forced equalization. Instead, cap
    # every behavioural section (never Profiling, a fixed 4-item demographic
    # block) at a MAXIMUM of 7 questions -- only trim a section that
    # exceeds the cap, never trim a section down to match a weaker sibling.
    # A section that only produced 3-5 good questions ships as-is; nothing
    # is deleted just to make sections look uniform. Drop from the END of
    # a capped section's existing narrative order (last-in-order first)
    # since sections are already ordered broadest/earliest to narrowest/
    # latest, so the earliest, most load-bearing questions are always kept.
    # change for b2c questionarie -- CHECK (user directive, 2026-09-09): the
    # cap is now PER SECTION (5 / 7 / 6 / 5), not one flat number. A flat cap
    # would trim Purchase Journey from its required 7 down to the smallest
    # section's size and silently delete required question types. Keyed by
    # canonical id so a renamed market section still gets the right cap.
    # change for b2c questionarie -- CHECK (audit, 2026-09-09): this was a
    # blind tail-trim. It is the LAST step before publication, so it could
    # silently undo the architect's slot guarantee -- a required question
    # added late in a section (slot top-ups are appended) was simply deleted
    # here, and the file then failed the very coverage gate the architect had
    # just satisfied. Trim slot-aware: keep one question per required theme
    # first, then fill the remaining places from the front.
    from src.nodes.validator_critic import REQUIRED_SLOTS as _REQ_SLOTS

    # change for b2c questionarie -- CHECK (live, UK vitamins 2026-09-09):
    # the architect's slot guarantee is not the last word. An advocacy/NPS
    # question was present when the guarantee ran, then removed downstream
    # (simulation/validation/best-pass restore), and the delivered file
    # shipped 4/5 in that section with the theme absent -- the guarantee had
    # already decided the section was full and never re-checked.
    #
    # Advocacy is the one required theme that is fully TEMPLATED (fixed
    # wording, fixed 0-10 scale), so unlike the others it can be restored
    # here deterministically, with no LLM call, at the last step before
    # publication. Any other missing theme still has to be reported rather
    # than invented -- see the coverage gate, which keeps such a pass dirty.
    for tb in tabs_out:
        if tb["tab"] == standard_sections.PROFILING_SECTION_ID:
            continue
        _canon_adv = narrative.section_canonical(blueprint, tb["tab"])
        if _canon_adv != "satisfaction_future_intent":
            continue
        _adv_pat = next(
            (p for k, _l, p in (_REQ_SLOTS.get(_canon_adv) or ())
             if k == "category_advocacy"), None,
        )
        if not _adv_pat:
            continue
        if any(re.search(_adv_pat, q.get("text") or "") for q in tb["questions"]):
            continue
        _n = int((tb["questions"][0] or {}).get("sample_size") or SAMPLE_SIZE)
        _labels = (["0 - Not at all likely"] + [str(i) for i in range(1, 10)]
                   + ["10 - Extremely likely"])
        # A plausible, mildly right-skewed NPS shape summing to 100.
        _pcts = [3.0, 2.0, 3.0, 4.0, 6.0, 12.0, 12.0, 17.0, 19.0, 12.0, 10.0]
        tb["questions"].append({
            "id": f"{tb['tab'][:2]}{len(tb['questions']) + 1}",
            "tab": tb["tab"],
            "text": f"Would you recommend {segment_label} to a friend?",
            "type": "single_choice",
            "chart_type": "bar",
            "options": _labels,
            "answers": [
                {"label": lab, "percentage": pct, "grounded_in": None,
                 "source_market": None, "grounding_label": "",
                 "confidence": "medium"}
                for lab, pct in zip(_labels, _pcts)
            ],
            "sample_size": _n,
            "distribution_note": (
                "Recommendation is an outcome measure and is not used to "
                "define the behavioural segments."
            ),
            "is_grounded": True,
            "question_confidence": "medium",
            "question_layer": "core",
            "funnel_position": len(tb["questions"]) + 1,
            "narrative_order": len(tb["questions"]) + 1,
            "evidence_aligned": False,
            "options_from_evidence": False,
        })

    for tb in tabs_out:
        if tb["tab"] == standard_sections.PROFILING_SECTION_ID:
            continue
        _canon = narrative.section_canonical(blueprint, tb["tab"])
        _cap = QUESTIONS_PER_SECTION.get(_canon)
        if not _cap or len(tb["questions"]) <= _cap:
            continue
        _slots = _REQ_SLOTS.get(_canon) or ()
        if not _slots:
            tb["questions"] = tb["questions"][:_cap]
            continue
        _kept, _claimed = [], set()
        for _k, _l, _p in _slots:
            for _q in tb["questions"]:
                if id(_q) in _claimed:
                    continue
                if re.search(_p, _q.get("text") or ""):
                    _kept.append(_q)
                    _claimed.add(id(_q))
                    break
        for _q in tb["questions"]:
            if len(_kept) >= _cap:
                break
            if id(_q) not in _claimed:
                _kept.append(_q)
                _claimed.add(id(_q))
        # Restore original narrative order among the survivors.
        _order = {id(q): i for i, q in enumerate(tb["questions"])}
        tb["questions"] = sorted(_kept[:_cap], key=lambda q: _order[id(q)])

    # ---- segments (4 behavioural + Profiling) -------------------------------
    segments_out = []
    for idx, tab_block in enumerate(tabs_out, start=1):
        tab_id = tab_block["tab"]
        if tab_id == standard_sections.PROFILING_SECTION_ID:
            title = "Profiling"
            focus = (
                "Classification questions used only to cross-tabulate the "
                "behavioural findings; asked last and never used to define "
                "the behavioural personas."
            )
        else:
            title = narrative.section_label(blueprint, tab_id) or tab_id
            sec_meta = next(
                (s for s in narrative.sections_for(blueprint) if s["section_id"] == tab_id),
                {},
            )
            focus = (sec_meta.get("remit") or "").strip() or _SECTION_COVERS.get(tab_id, "")

        questions_out = []
        for qn, q in enumerate(tab_block["questions"], start=1):
            options = q.get("options") or []
            qtype = _delivery_question_type(q.get("type") or "single_choice", options)
            answers = q.get("answers") or []
            by_label = {a.get("label"): a for a in answers}
            option_objs = [
                {
                    "label": _lowercase_segment_mid_sentence(opt, segment_label),
                    "pct": _delivery_option_pct(by_label.get(opt, {})),
                }
                for opt in options
            ]
            note = _scrub_persona_names(
                q.get("distribution_note") or "", archetype_persona_names, segment_names,
            )
            if tab_id == standard_sections.PROFILING_SECTION_ID and not note:
                note = "Profiling only; not used to define the behavioural personas."
            questions_out.append({
                "id": f"Q{qn}",
                "text": _lowercase_segment_mid_sentence(
                    _delitteral_typical_period(
                        _plain_word_substitute(
                            _apply_manager_wording_rewrite(
                                _clean_stem(_tighten_stem(q.get("text")))
                            )
                        )
                    ),
                    segment_label,
                ),
                "type": qtype,
                "options": option_objs,
                "key_insight": note,
            })

        segments_out.append({
            "id": idx,
            "title": title,
            "focus": focus,
            "questions": questions_out,
        })

    # ---- structure (derived, never hardcoded) -------------------------------
    structure = {
        "totalQuestions": sum(len(s["questions"]) for s in segments_out),
        "sections": [
            {
                "label": s["title"],
                "covers": s["focus"],
                "questions": len(s["questions"]),
            }
            for s in segments_out
        ],
    }

    return {
        "surveyScope": survey_scope,
        "behaviouralPersonas": behavioural_personas,
        "structure": structure,
        "segments": segments_out,
    }


def build_survey_json(state: SurveyState) -> dict:
    """Construct the canonical final_survey dict from validated state."""
    answered = state.get("answered_questions") or []
    critique = state.get("critique") or {}
    segment_label = (
        state.get("normalized_segment") or state.get("market_segment") or ""
    )

    # change for b2c questionarie — ship the BEST pass, not the last one.
    # The revision loop is not monotonic, so without this a survey that scored
    # 0.742 on its first pass can ship at 0.031 after two "improvements".
    #
    # CHECK 18 / Section 26f, CHECK / Section 27g: a pure score comparison
    # here can still pick a final pass with a zero-tolerance defect (wrong
    # currency, or a multi-select sum that was generated as a constrained
    # share instead of independent probabilities) over a pass that is clean
    # on both but scored lower on unrelated issues (validator_critic.py's
    # own best-pass snapshot already prefers clean-over-dirty regardless of
    # score — this must match that same preference, or the assembler can
    # undo it). Re-run the SAME detectors used during validation directly on
    # both candidate answer sets so the two call sites cannot drift apart.
    from src.nodes.validator_critic import _currency_issue as _val_currency_issue
    from src.nodes.validator_critic import (
        _core_layer_currency_issue as _val_core_currency_issue,
    )
    from src.nodes.validator_critic import (
        _suspicious_multiselect_sum_issue as _val_multiselect_sum_issue,
    )
    from src.nodes.validator_critic import find_construct_duplicates
    from src.nodes.validator_critic import (
        find_missing_required_slots as _val_find_missing_required_slots,
    )
    from src.nodes.validator_critic import (
        find_nps_shape_issue as _val_find_nps_shape_issue,
    )
    from src.nodes.validator_critic import (
        find_offtheme_questions as _val_find_offtheme_questions,
    )

    def _is_zero_tolerance_clean(qs: list) -> bool:
        expected = standard_sections.currency_for(
            state.get("country") or state.get("geography_label")
            or state.get("region") or ""
        )
        if any(_val_currency_issue(q.get("options") or [], expected) for q in qs):
            return False
        # change for b2c questionarie -- CHECK (Section 30, currency): a
        # core-layer question hardcoding ANY currency is wrong regardless
        # of what `expected` says for this specific job.
        if any(
            _val_core_currency_issue(q.get("options") or [])
            for q in qs if q.get("question_layer") == "core"
        ):
            return False
        sums = {
            q.get("id"): sum(float(a.get("percentage") or 0.0) for a in (q.get("answers") or []))
            for q in qs if q.get("type") == "multiple_choice"
        }
        if any(
            _val_multiselect_sum_issue(qid, total, sums)
            for qid, total in sums.items()
        ):
            return False
        # change for b2c questionarie -- CHECK (Section 30, escalated 28c):
        # match validator_critic.py's known_duplicate_fail exactly, using the
        # SAME shared find_construct_duplicates() so the two call sites
        # cannot disagree on what counts as a known duplicate.
        if find_construct_duplicates(qs):
            return False
        # change for b2c questionarie -- CHECK (user directive, 2026-09-08):
        # mirror validator_critic.py's required_slot_fail using the SAME
        # shared helpers, so a pass missing one of the 16 required question
        # types (or with a malformed NPS scale) can never be crowned "best"
        # at final restore either.
        _canon_to_market = {}
        for _s in ((state.get("survey_blueprint") or {}).get("sections") or []):
            _c = (_s.get("canonical") or "").strip()
            _sid = (_s.get("section_id") or "").strip()
            if _c and _sid:
                _canon_to_market[_c] = _sid
        if _val_find_missing_required_slots(qs, _canon_to_market):
            return False
        # change for b2c questionarie -- CHECK (user directive, 2026-09-09):
        # the slot list is closed, so an extra off-theme question makes a
        # pass dirty here too, not just during the revision loop.
        if _val_find_offtheme_questions(qs, _canon_to_market):
            return False
        if _val_find_nps_shape_issue(qs):
            return False
        return True

    restored_from = None
    best_score = state.get("best_score")
    final_score = critique.get("score")
    best_answered = state.get("best_answered")
    if best_score is not None and final_score is not None and best_answered:
        final_clean = _is_zero_tolerance_clean(answered)
        best_clean = _is_zero_tolerance_clean(best_answered)
        should_restore = (
            (best_clean and not final_clean)  # clean beats dirty, any score
            or (best_clean == final_clean and best_score > final_score)
        )
        if should_restore:
            restored_from = (final_score, best_score)
            answered = best_answered
            critique = state.get("best_critique") or critique
    answered = _drop_duplicates(
        answered,
        critique.get("duplicates"),
        critique.get("hard_duplicates"),
    )
    # change for b2c questionarie — last line of defence on uniqueness. Runs on
    # the final set, across all four tabs, after every other filter.
    blueprint = state.get("survey_blueprint") or narrative.default_blueprint(
        segment=segment_label, region=state.get("region") or "Global",
    )
    answered, dedup_notes = _enforce_absolute_uniqueness(
        answered, segment_label, blueprint,
    )

    # change for b2c questionarie — A12: last-resort guard on unfieldable items.
    # A stem promising a grid ("how satisfied with EACH of the following
    # aspects") whose options are only a rating scale has nothing to rate — a
    # respondent cannot answer it and an interviewer cannot field it. The
    # validator flags these, but the revision loop is capped, so one can survive
    # to publication. Losing a slot is better than shipping a question that
    # cannot be answered.
    from src.nodes.validator_critic import _is_broken_grid_question

    kept_q, broken = [], []
    for q in answered:
        if q.get("question_layer") in ("screening", "profiling"):
            kept_q.append(q)
            continue
        if _is_broken_grid_question(
            q.get("text") or "", q.get("options") or [], q.get("type") or "",
        ):
            broken.append(q)
            continue
        kept_q.append(q)
    if broken:
        answered = kept_q
        for q in broken:
            dedup_notes.append(
                f"unfieldable question removed ({q.get('id')}): grid stem with "
                f"no attributes to rate — {(q.get('text') or '')[:70]}"
            )

    # change for b2c questionarie — single-geo jobs publish only that region's %
    # change for b2c questionarie — when the survey is written for a country,
    # the deliverable must SAY that country. The parent region is still what
    # drives evidence and the shared core, but a file titled "Europe" that is
    # actually the German market misrepresents what the reader is holding.
    region_name = (
        state.get("country") or state.get("geography_label")
        or state.get("region") or "Global"
    )
    geo = match_geographic_region(region_name)
    primary_slug = region_to_slug(geo) if geo else (
        "global" if str(region_name).strip().lower() in ("global", "") else region_to_slug(region_name)
    )
    # change for b2c questionarie -- CHECK (Section 28g): `geo` only matches
    # one of the 5 GEOGRAPHIC_REGIONS names, so it was always None for a
    # country job ("China" is not itself a region) even though `region_name`
    # correctly held the country and `primary_slug` above already correctly
    # resolved to "china". `single_geo` and (further below) `frontend_slug`
    # both gated on `bool(geo)`, so a country job was silently treated as
    # multi-region/global downstream -- confirmed live: dataByRegion on a
    # single-country China survey carried only "global"/"asia-pacific" keys,
    # never "china", because this exact branch fell through to the
    # multi-region path. A country job is unambiguously single-geography
    # regardless of whether `geo` (the REGION match) is set.
    is_country_job = bool(state.get("country"))
    # Multi-region / Global jobs keep multi-slug dataByRegion.
    regions_sim = state.get("regions_to_simulate") or []
    single_geo = (
        is_country_job
        or (bool(geo) and len([r for r in regions_sim if r != "Global"]) <= 1)
    )
    if not single_geo and not geo:
        primary_slug = None  # signal: multi-region / global mode

    # Group by section (blueprint order); within a section sort by story order.
    # change for b2c questionarie
    section_order = [s["section_id"] for s in narrative.sections_for(blueprint, include_standard=True)]
    by_tab = {tab: [] for tab in section_order}
    for q in answered:
        by_tab.setdefault(q.get("tab"), []).append(q)

    tabs_out, counts_by_tab, counter = [], {}, 0
    # change for b2c questionarie — A11: architect id -> published id.
    id_remap: dict[str, str] = {}
    for tab in section_order:
        # change for b2c questionarie — Phase A8: storyline order wins. The beat
        # a question fills already encodes broad→specific and safe→sensitive, so
        # narrative_order is authoritative and funnel_position is the fallback
        # for anything generated before the storyline existed.
        beats = narrative.beats_for_tab(blueprint, tab)
        beat_rank = {b["beat_id"]: i for i, b in enumerate(beats)}

        def _order_key(q, _rank=beat_rank):
            def _as_int(value, default):
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return default

            order = _as_int(q.get("narrative_order"), 0)
            fallback = _as_int(q.get("funnel_position"), 999)
            return (
                order or fallback,
                _rank.get(q.get("beat_id"), 999),
                fallback,
                str(q.get("id") or ""),
            )

        ordered = sorted(by_tab.get(tab, []), key=_order_key)
        # change for b2c questionarie — spread chart types across the tab
        _diversify_section_charts(ordered)
        questions = []
        for slot, q in enumerate(ordered, start=1):
            counter += 1
            dbr = q.get("data_by_region") or {}
            # Region job: keep only this region's simulated % in the rich survey blob.
            if primary_slug and primary_slug != "global" and dbr:
                dbr = {primary_slug: dbr[primary_slug]} if primary_slug in dbr else dbr
            # change for b2c questionarie — re-densify story position after any
            # dedup drops, so published order is always 1..N with no gaps.
            q["narrative_order"] = slot
            q["funnel_position"] = slot
            # change for b2c questionarie — A11: the scoring model was built
            # against the architect's working ids (cp1, bb3). Questions are
            # renumbered to q1..qN here, so remember the mapping or every weight
            # points at an id the published instrument does not contain.
            if q.get("id"):
                id_remap[str(q["id"])] = f"q{counter}"
            questions.append({
                "id": f"q{counter}",
                # change for b2c questionarie — A16: last-pass stem tightening.
                "text": _tighten_stem(q.get("text")),
                "type": q.get("type"),
                "chart_type": q.get("chart_type"),
                "funnel_position": slot,
                # change for b2c questionarie — Phase A8 storyline provenance
                "narrative_order": slot,
                "beat_id": q.get("beat_id") or "",
                "beat_title": q.get("beat_title") or "",
                "beat_canonical": q.get("beat_canonical"),
                # change for b2c questionarie — A6 global selection needs layer tags
                "question_layer": q.get("question_layer") or "full",
                "sample_size": SAMPLE_SIZE,
                "options": q.get("options") or [],
                "answers": [{k: a.get(k) for k in _ANSWER_KEYS} for a in (q.get("answers") or [])],
                "is_grounded": q.get("is_grounded", False),
                "evidence_aligned": q.get("evidence_aligned", False),
                "options_from_evidence": q.get("options_from_evidence", False),
                "distribution_note": q.get("distribution_note", ""),
                "data_by_region": dbr,
                "_frontend_chart": q.get("_frontend_chart"),
            })
        counts_by_tab[tab] = len(questions)
        tabs_out.append({"tab": tab, "questions": questions})

    # Official metrics (prefer state-level from Node 6, fall back to critique).
    grounded_pct = state.get("grounded_pct", critique.get("grounded_pct", 0.0))
    direct_pct = state.get("direct_grounded_pct", critique.get("direct_grounded_pct", 0.0))
    adjacent_pct = state.get("adjacent_grounded_pct", critique.get("adjacent_grounded_pct", 0.0))
    revision_count = state.get("revision_count") or 0
    grounding_thin = bool(state.get("grounding_thin"))

    warnings = []
    # change for b2c questionarie — be explicit when a later revision was discarded
    if restored_from:
        warnings.append(
            f"Restored the best revision (score {restored_from[1]}) — the final "
            f"pass scored {restored_from[0]} and was discarded."
        )
    if grounding_thin:
        warnings.append("Grounding is thin — most answers are inferred, not data-backed.")
    # change for b2c questionarie — surface anything the uniqueness gate removed
    if dedup_notes:
        warnings.append(
            f"{len(dedup_notes)} duplicate/overlapping question(s) removed at assembly."
        )
        warnings.extend(dedup_notes[:10])
    # change for b2c questionarie — under-target vs QUESTIONS_PER_TAB_TARGET via CATEGORY_PLAN
    # change for b2c questionarie -- CHECK (Section 28h): keep the canonical
    # tab ids as structured data (not just the formatted warning string), so
    # they can be cross-referenced against nonDiscriminatingQuestions below
    # without re-parsing a tab name back out of prose.
    under_target_tabs = []
    for tab, target in narrative.plan_for(blueprint, QUESTIONS_PER_TAB_TARGET):
        if counts_by_tab.get(tab, 0) < target:
            warnings.append(f"Tab '{tab}' under target: {counts_by_tab.get(tab, 0)}/{target}.")
            under_target_tabs.append(tab)
    if revision_count >= 3:
        warnings.append(f"Revision cap reached ({revision_count}) — shipped best-effort with remaining issues.")
    if adjacent_pct >= _HIGH_ADJACENT_PCT:
        warnings.append(f"{adjacent_pct}% of grounded options use data borrowed from related markets (see grounding_label).")

    ctx = current_date_context()

    # change for b2c questionarie — personas are NOT shipped. They are an
    # internal device for weighting the simulated percentages: the reader has
    # never seen them, the deliverable never defines them, and their names are
    # already scrubbed out of every visible string (_scrub_persona_names).
    # Leaving the catalog in the payload put ~20 archetypes with invented age,
    # income and education bands into a file whose only client-facing
    # classification is the four CMI survey categories.
    frontend_slug = primary_slug if (single_geo or geo) else None
    # change for b2c questionarie — per-question display N (25k–30k); sample_size stays 2500
    seg_key = state.get("normalized_segment") or state.get("market_segment") or "survey"
    n_seed_prefix = f"{seg_key}:{region_name}"
    survey_display_n = _display_n(f"{n_seed_prefix}:survey")
    # change for b2c questionarie -- CHECK (Section 28e): named so it can be
    # reused for storyline_notes below, not just the frontend distribution
    # notes this was originally built for.
    archetype_persona_names = [
        p.get("name")
        for p in ((state.get("persona_catalog") or {}).get("personas") or [])
        if p.get("name")
    ]
    frontend = {
        "sections": _build_frontend_sections(
            tabs_out,
            blueprint=blueprint,
            primary_region_slug=frontend_slug,
            n_seed_prefix=n_seed_prefix,
            # change for b2c questionarie — A12: the panel archetypes, so a
            # distribution note that names one can be caught and stripped.
            persona_names=archetype_persona_names,
        ),
    }
    # change for b2c questionarie -- CHECK (Section 28f): whether ANY
    # question actually uses a concrete currency-denominated option, so a
    # storyline note claiming price was "kept in relative terms" can be
    # checked against what the document actually shipped.
    any_concrete_currency = any(
        _CURRENCY_AMOUNT_RE.search(str(o))
        for sec in frontend["sections"]
        for q in sec.get("questions") or []
        for o in q.get("options") or []
    )
    # Strip assembler-only helper keys from the rich survey blob.
    for t in tabs_out:
        for q in t["questions"]:
            q.pop("_frontend_chart", None)

    target_customer = _build_target_customer(state)
    # change for b2c questionarie -- CHECK (Section 28a): surface only the
    # "shipped incomplete" class of warning (revision cap reached, a
    # section under its target count) on the client-facing document --
    # not every backend note (grounding-thin, adjacent-data-borrowed are
    # informational quality context, not an incompleteness signal).
    completeness_warnings = [
        w for w in warnings
        if w.startswith("Revision cap reached") or "under target" in w
    ]
    # change for b2c questionarie -- CHECK (Section 28c): non-discriminating
    # duplicate check computed here (frontend + id_remap already exist) so
    # it can be folded into the SAME completeness-warning banner as 28a
    # rather than only living in nested nonDiscriminatingQuestions metadata.
    non_discriminating = _non_discriminating(
        frontend, _remap_segment_model(state.get("segment_model") or {}, id_remap),
    )
    completeness_warnings.extend(
        _non_discriminating_duplicate_warnings(non_discriminating, segment_label)
    )
    # change for b2c questionarie -- CHECK (Section 28h): a soft note, not a
    # hard violation (some non-discriminating questions are legitimate
    # context/market-sizing content worth keeping regardless) -- if a
    # section is ALREADY under its target count AND already contains a
    # non-discriminating question, a future regeneration should prioritize
    # replacing that question with one that both fits the section and
    # contributes segmentation signal, rather than only adding a new
    # question alongside it. under_target_tabs holds canonical ids
    # ("consumer_profile"); nonDiscriminatingQuestions' sectionId holds the
    # short display id ("profile") -- translate through the same
    # _FRONTEND_SECTIONS mapping the frontend itself was built with, so the
    # two do not silently fail to match on a namespace difference.
    under_target_display_ids = {
        _FRONTEND_SECTIONS.get(tab, (tab, ""))[0] for tab in under_target_tabs
    }
    for nd in non_discriminating:
        if nd.get("sectionId") in under_target_display_ids:
            completeness_warnings.append(
                f"{nd.get('id')} carries no segmentation signal and sits in "
                "a section already under its target question count -- "
                "prioritize replacing it with a question that both fits "
                "the section and contributes segmentation signal, rather "
                "than only adding a new question alongside it."
            )
    # change for b2c questionarie -- CHECK (B2C_MASTER_RULES_FINAL /
    # JSON_FORMAT_SPECIFICATION): the delivered file is now the new-spec
    # shape (surveyScope/behaviouralPersonas/structure/segments) built
    # directly from tabs_out, which has already been through every upstream
    # cleaning pass (dedup, currency scrub, persona scrub, best-pass
    # restore, id renumbering). All the internal-only bookkeeping this
    # function computed above (warnings, non_discriminating,
    # completeness_warnings, survey_definition, target_customer, frontend,
    # segmentModel, storyline metadata, etc.) stays used for INTERNAL
    # decision-making (e.g. is_zero_tolerance_clean) but is deliberately not
    # serialized -- "nothing else at the top level" per the spec.
    return _build_delivery_json(
        state=state,
        blueprint=blueprint,
        tabs_out=tabs_out,
        region_name=region_name,
        segment_label=segment_label,
        archetype_persona_names=archetype_persona_names,
    )


def to_json_string(survey_dict: dict) -> str:
    """Serialize the final survey dict to canonical pretty JSON."""
    return json.dumps(survey_dict, indent=2, ensure_ascii=False)


def assembler(state: SurveyState) -> dict:
    """Assemble the final survey object (Node 7, terminal)."""
    return {"final_survey": build_survey_json(state)}


if __name__ == "__main__":
    state = {
        "normalized_segment": "Organic Milk", "segment_type": "b2c", "region": "Global",
        "audience_profile": {"demographics": {"age": "25-55"}},
        "classification_reason": "Bought by individual shoppers for household use.",
        "grounding_thin": False, "revision_count": 1,
        "grounded_pct": 38.0, "direct_grounded_pct": 22.0, "adjacent_grounded_pct": 16.0,
        "counts_by_tab": {"consumer_profile": 1, "buying_behavior": 1,
                          "preferences_expectations": 0, "satisfaction_future_intent": 0},
        "critique": {"duplicates": [["bb1", "cp9"]]},
        "answered_questions": [
            {"id": "cp1", "tab": "consumer_profile", "text": "In which region do you live?",
             "type": "single_choice", "chart_type": "donut",
             "options": ["North America", "Europe", "Asia Pacific", "Rest of World"],
             "evidence_aligned": True, "is_grounded": True, "distribution_note": "Regions grounded.",
             "answers": [
                 {"label": "North America", "percentage": 41.0, "grounded_in": "https://example.gov/dairy",
                  "source_market": "Organic Milk", "grounding_label": "", "confidence": "high"},
                 {"label": "Europe", "percentage": 33.0, "grounded_in": "https://example.gov/dairy",
                  "source_market": "Organic Milk", "grounding_label": "", "confidence": "high"},
                 {"label": "Asia Pacific", "percentage": 18.0, "grounded_in": "https://example.gov/dairy",
                  "source_market": "Organic Milk", "grounding_label": "", "confidence": "high"},
                 {"label": "Rest of World", "percentage": 8.0, "grounded_in": "https://example.gov/dairy",
                  "source_market": "Organic Milk", "grounding_label": "", "confidence": "high"},
             ]},
            {"id": "bb3", "tab": "buying_behavior", "text": "How often do you buy organic milk?",
             "type": "single_choice", "chart_type": "bar",
             "options": ["Weekly", "Monthly", "Rarely", "Never"],
             "evidence_aligned": False, "is_grounded": False, "distribution_note": "Inferred.",
             "answers": [
                 {"label": "Weekly", "percentage": 45.0, "grounded_in": None, "source_market": None,
                  "grounding_label": "", "confidence": "low"},
                 {"label": "Monthly", "percentage": 35.0, "grounded_in": None, "source_market": None,
                  "grounding_label": "", "confidence": "low"},
                 {"label": "Rarely", "percentage": 15.0, "grounded_in": None, "source_market": None,
                  "grounding_label": "", "confidence": "low"},
                 {"label": "Never", "percentage": 5.0, "grounded_in": None, "source_market": None,
                  "grounding_label": "", "confidence": "low"},
             ]},
        ],
    }
    survey = build_survey_json(state)
    print(to_json_string(survey))
