"""Screening and Profiling — the two sections every consultancy study carries.

# change for b2c questionarie — Phase A9 (segmentation study format)

A real segmentation questionnaire is bookended by two sections the narrative
planner should not invent, because their content is methodological rather than
category-driven:

    SCREENING   qualify the respondent, confirm they decide, exclude industry
                insiders. Asked first; a wrong answer terminates the interview.

    PROFILING   age, household composition, income, NPS, overall satisfaction.
                Asked LAST, and used only for cross-tabs — never as segmentation
                input, and never as a behavioural finding.

Both are templated here so they are consistent across every market and cannot
drift. Only the category wording is substituted in.

On the age/income ban
---------------------
The instrument bans age and income everywhere EXCEPT this Profiling section.
The ban exists so behavioural sections cannot fall back on demographics; it was
never meant to stop a study cutting its results by age. Quarantining them here
keeps both properties: no demographic question can appear in a behavioural
section, and the cross-tabs a client expects still exist.
"""

from __future__ import annotations

SCREENING_SECTION_ID = "screening"
PROFILING_SECTION_ID = "profiling"

SCREENING_LABEL = "Screening"
PROFILING_LABEL = "Profiling & Classification"

# Beats give the two sections the same storyline machinery as the rest.
SCREENING_BEATS = [
    {"beat_id": "category_qualification", "title": "Category qualification",
     "intent": "Confirm the respondent actually uses or buys the category.",
     "canonical": None, "order": 1},
    {"beat_id": "decision_authority", "title": "Decision authority",
     "intent": "Confirm they decide, or share the decision, for the household.",
     "canonical": None, "order": 2},
    {"beat_id": "industry_exclusion", "title": "Industry exclusion",
     "intent": "Screen out market research, advertising and category insiders.",
     "canonical": None, "order": 3},
]

PROFILING_BEATS = [
    {"beat_id": "age_band", "title": "Age", "intent": "Age band for cross-tabs.",
     "canonical": None, "order": 1},
    {"beat_id": "gender", "title": "Gender",
     "intent": "Gender for cross-tabs.", "canonical": None, "order": 2},
    {"beat_id": "household_composition", "title": "Household composition",
     "intent": "Who else is in the household.", "canonical": None, "order": 3},
    {"beat_id": "income_band", "title": "Household income",
     "intent": "Income band for cross-tabs.", "canonical": None, "order": 4},
    {"beat_id": "advocacy_nps", "title": "Recommendation likelihood",
     "intent": "Standard 0-10 NPS item.", "canonical": None, "order": 5},
    {"beat_id": "overall_satisfaction_profile", "title": "Overall satisfaction",
     "intent": "Single headline satisfaction reading for cross-tabs.",
     "canonical": None, "order": 6},
]


def screening_section() -> dict:
    return {
        "section_id": SCREENING_SECTION_ID,
        "label": SCREENING_LABEL,
        "remit": (
            "Qualify the respondent for the study: category usage, decision "
            "authority, and industry exclusion. A disqualifying answer ends the "
            "interview."
        ),
        "canonical": None,
        "order": 0,
        "kind": "screening",
    }


def profiling_section() -> dict:
    return {
        "section_id": PROFILING_SECTION_ID,
        "label": PROFILING_LABEL,
        "remit": (
            "Demographics and headline metrics used for CROSS-TABS only, never "
            "as segmentation input."
        ),
        "canonical": None,
        "order": 99,
        "kind": "profiling",
    }


def screening_questions(segment: str, examples: str = "") -> list[dict]:
    """The three standard screeners, worded for this category.

    ``examples`` is an illustrative brand list for the qualification item. Brand
    names are banned in the instrument proper, but a screener has to name the
    category concretely or respondents mis-qualify — that is standard practice
    and the exception is deliberate.
    """
    eg = f" (e.g. {examples})" if examples else ""
    return [
        {
            "id": "s1", "beat_id": "category_qualification",
            "text": f"Do you currently buy or use {segment}{eg}?",
            "type": "single_choice",
            "options": [
                "Yes, regularly",
                "Yes, occasionally",
                "No, but I have in the past 12 months",
                "No, never",
            ],
            "terminate_on": ["No, never"],
            "screening": True,
        },
        {
            "id": "s2", "beat_id": "decision_authority",
            "text": (
                f"Who decides which {segment} your household buys?"
            ),
            "type": "single_choice",
            "options": [
                "I decide on my own",
                "I share the decision with someone else",
                "Someone else decides, I have some input",
                "Someone else decides entirely",
            ],
            "terminate_on": ["Someone else decides entirely"],
            "screening": True,
        },
        {
            "id": "s3", "beat_id": "industry_exclusion",
            "text": (
                "Do you or does anyone in your household work in market "
                "research, advertising, or the industry that makes or sells "
                f"{segment}?"
            ),
            "type": "single_choice",
            "options": ["Yes", "No"],
            "terminate_on": ["Yes"],
            "screening": True,
        },
    ]


# change for b2c questionarie — currency per geography, so the income bands
# match the money questions elsewhere in the same instrument.
_CURRENCY_BY_GEO = {
    "u.s.": "$", "united states": "$", "usa": "$", "canada": "C$",
    "north america": "$",
    "germany": "€", "france": "€", "italy": "€", "spain": "€",
    "benelux": "€", "netherlands": "€", "europe": "€",
    "u.k.": "£", "uk": "£", "united kingdom": "£",
    "denmark": "kr", "norway": "kr", "sweden": "kr",
    "india": "₹", "japan": "¥", "china": "¥",
    "australia": "A$", "brazil": "R$", "mexico": "MX$",
    "saudi arabia": "SAR", "united arab emirates": "AED",
}


def currency_for(geography: str) -> str:
    """Currency symbol for a country or region; falls back to '$'."""
    return _CURRENCY_BY_GEO.get((geography or "").strip().lower(), "$")


# change for b2c questionarie -- CHECK (live, Sport Shoes/China, 2026-09-06):
# income bands used to be ONE fixed set of thresholds (25/50/75/100/150,000)
# applied to WHATEVER currency symbol the geography mapped to. That produces
# a plausible-looking "$25,000-$150,000" ladder for the US, but the IDENTICAL
# digits with a Yen or Yuan symbol pasted on ("Y25,000") describe a household
# living in poverty, not a realistic middle-class range for that market --
# same defect the architect prompt already warns against for LLM-written
# questions ("getting the symbol right but the amounts wrong is still
# wrong"), just living in hardcoded Python instead. Real annual household
# income scale is genuinely different per market (order-of-magnitude
# differences are normal — JPY and INR routinely quote incomes in millions/
# lakhs where USD/EUR quote in tens of thousands) so thresholds must be
# keyed per geography, not shared. Approximate, broadly realistic
# middle-of-market ladders -- not official statistics, but the right ORDER
# OF MAGNITUDE for each currency, which is what matters for a respondent
# reading the options and recognising their own bracket.
_INCOME_STEPS_BY_GEO = {
    "u.s.": (25000, 50000, 75000, 100000, 150000),
    "united states": (25000, 50000, 75000, 100000, 150000),
    "usa": (25000, 50000, 75000, 100000, 150000),
    "north america": (25000, 50000, 75000, 100000, 150000),
    "canada": (30000, 60000, 90000, 120000, 180000),
    "germany": (20000, 40000, 60000, 80000, 120000),
    "france": (20000, 40000, 60000, 80000, 120000),
    "italy": (15000, 30000, 45000, 60000, 90000),
    "spain": (15000, 30000, 45000, 60000, 90000),
    "benelux": (20000, 40000, 60000, 80000, 120000),
    "netherlands": (20000, 40000, 60000, 80000, 120000),
    "europe": (20000, 40000, 60000, 80000, 120000),
    "u.k.": (18000, 35000, 55000, 75000, 110000),
    "uk": (18000, 35000, 55000, 75000, 110000),
    "united kingdom": (18000, 35000, 55000, 75000, 110000),
    "denmark": (200000, 400000, 600000, 800000, 1200000),
    "norway": (300000, 500000, 700000, 900000, 1300000),
    "sweden": (200000, 400000, 600000, 800000, 1200000),
    "india": (300000, 600000, 1000000, 1500000, 2500000),
    "japan": (3000000, 5000000, 7000000, 9000000, 12000000),
    "china": (100000, 200000, 350000, 500000, 800000),
    "australia": (40000, 70000, 100000, 140000, 200000),
    "brazil": (30000, 60000, 100000, 150000, 250000),
    "mexico": (150000, 300000, 500000, 750000, 1200000),
    "saudi arabia": (60000, 120000, 180000, 250000, 400000),
    "united arab emirates": (80000, 150000, 250000, 350000, 550000),
}
_DEFAULT_INCOME_STEPS = (25000, 50000, 75000, 100000, 150000)


def _income_steps_for(geography: str) -> tuple:
    return _INCOME_STEPS_BY_GEO.get(
        (geography or "").strip().lower(), _DEFAULT_INCOME_STEPS
    )


def _fmt_amount(n: int) -> str:
    return f"{n:,}"


def _income_bands(currency: str = "$", geography: str = "") -> list[str]:
    """Household-income bands rendered in one currency, at that market's own
    realistic annual-income scale (see _INCOME_STEPS_BY_GEO)."""
    c = currency
    steps = _income_steps_for(geography)
    bands = [f"Under {c}{_fmt_amount(steps[0])}"]
    for lo, hi in zip(steps, steps[1:]):
        bands.append(f"{c}{_fmt_amount(lo)} - {c}{_fmt_amount(hi - 1)}")
    bands.append(f"{c}{_fmt_amount(steps[-1])} or more")
    bands.append("Prefer not to say")
    return bands


def profiling_questions(segment: str, currency: str = "$", geography: str = "") -> list[dict]:
    """Standard cross-tab items. Asked last; never used for segmentation.

    # change for b2c questionarie -- CHECK (user directive, 2026-09-08):
    # capped to the 3 most-used cross-tab cuts (age, gender, income) as part
    # of the short-survey format. household_composition ("who else lives in
    # your household") dropped -- least frequently analysed of the four.
    # change for b2c questionarie -- CHECK (user directive, 2026-09-09): the
    # 23-question framework specifies Profiling = age group + gender ONLY,
    # so household income is dropped too. `currency`/`geography` are kept in
    # the signature (callers still pass them, and _income_bands stays for
    # the money questions the architect writes inside Section 2).
    """
    return [
        {
            "id": "d1", "beat_id": "age_band",
            "text": "Which age group are you in?",
            "type": "single_choice",
            "options": ["18-24", "25-34", "35-44", "45-54", "55-64", "65 or older"],
            "profiling": True,
        },
        {
            # change for b2c questionarie — A13: gender is a standard cross-tab
            # cut and was missing. Like age it lives only here, never as a
            # behavioural question.
            "id": "d2", "beat_id": "gender",
            "text": "What is your gender?",
            "type": "single_choice",
            "options": ["Woman", "Man", "Non-binary", "Prefer to self-describe",
                        "Prefer not to say"],
            "profiling": True,
        },
    ]


# change for b2c questionarie — satisfaction and NPS used to sit in Profiling,
# after age/gender/household/income. That broke the reading flow twice over: the
# Satisfaction section had no headline satisfaction reading of its own (its lead
# slot got filled by an intent or switching question instead), and the Profiling
# block jumped from household income back to how you feel about the product.
# Both are attitude questions about the category, so they belong at the head of
# the satisfaction section — which is also where professional instruments put
# them, with demographics kept last and separate.
def satisfaction_anchor_questions(segment: str) -> list[dict]:
    """Overall satisfaction + NPS — the headline reads for the satisfaction tab.

    Asked before the generated satisfaction questions so the section opens on
    the broadest judgement and narrows from there, and so demographics stay a
    clean, uninterrupted final block.
    """
    return [
        {
            "id": "s1", "beat_id": "overall_satisfaction",
            # change for b2c questionarie — simplified per manager review,
            # 2026-08-25 (China Sport Shoes): "Overall, how satisfied are
            # you with X you currently use?" -> "How satisfied are you with
            # your current X?" -- drops the throat-clearing "Overall," lead
            # the same way _UNNECESSARY_TIMEFRAME_LEAD_RE already strips
            # other filler leads, just for this hardcoded template string.
            # change for b2c questionarie -- CHECK (user directive,
            # 2026-09-09): plain-language scale, matching the labels now
            # mandated in question_architect.py's prompt. This function is
            # currently disabled (_SHORT_SURVEY_NO_ANCHORS) but is kept
            # aligned so re-enabling it cannot reintroduce formal wording.
            "text": f"How satisfied are you with your current {segment}?",
            "type": "likert_5",
            "options": [
                "Very unsatisfied", "Unsatisfied", "Neutral",
                "Satisfied", "Very satisfied",
            ],
            "standard": True,
        },
        {
            "id": "s2", "beat_id": "advocacy_nps",
            # change for b2c questionarie — this is a hardcoded template
            # question, never touched by the architect or the revision loop
            # or the manager-wording exact-match table (whose keys must match
            # byte-for-byte, and this template is assembled with an f-string,
            # so slight generation-time variance around it never lines up).
            # Simplified per manager review, 2026-08-25 (China Sport Shoes):
            # "How likely are you to recommend X to a friend or family
            # member?" -> "Would you recommend X to others?" -- shorter,
            # conversational phrasing on the SAME 0-10 NPS construct/scale.
            "text": f"Would you recommend the {segment} you currently use to others?",
            "type": "single_choice",
            "options": [
                "0 - Not at all likely", "1", "2", "3", "4", "5",
                "6", "7", "8", "9", "10 - Extremely likely",
            ],
            "nps": True,
            "standard": True,
        },
    ]


def fielding_notes(segment: str, n_segments: int, sample_size: int = 2500) -> dict:
    """Methodology notes a client expects on the back page."""
    per_segment = max(150, sample_size // max(1, n_segments) // 2)
    return {
        "recommendedSample": (
            f"n=800-1,200 for stable segment-level sub-groups "
            f"(roughly n>={per_segment} per segment after weighting). At "
            "n=1,000 the margin of error on a whole-sample percentage is "
            "about ±3.1pp at 95% confidence; on a segment of n=200 it widens "
            "to about ±6.9pp, so segment-level differences under ~10pp should "
            "not be reported as real."
        ),
        "quotaControls": (
            "Apply quotas on the category-usage and frequency screeners so heavy "
            "users — who are more willing survey-takers — are not over-sampled."
        ),
        "estimatedLength": "8-10 minutes.",
        # change for b2c questionarie — A15: the mechanics a scripter needs.
        # Without these the document is a question list, not a fieldable
        # instrument.
        "scriptingNotes": [
            "Rotate the order of answer options on every non-ordinal question "
            "to remove primacy bias. Do NOT rotate ordinal scales (frequency "
            "bands, price bands, satisfaction) — their order carries meaning.",
            "Anchor 'Other', 'None of these' and 'Prefer not to say' to the "
            "bottom of the list, exempt from rotation.",
            "Every 'Other' option takes an open-text capture, coded back into "
            "the frame at analysis if any single verbatim exceeds 3%.",
            "No routing or skip logic: every respondent sees every question, so "
            "the base is constant across the instrument.",
            "Include one attention check in the middle third — a standard "
            "instructed-response item. Terminate on failure and replace.",
        ],
        "caveats": [
            # change for b2c questionarie — say what the numbers are without
            # describing panel mechanics that were never run.
            "Percentages in this document are modelled from published market "
            "evidence, not measured from fielded interviews. Replace them with "
            "primary data once the study is run.",
            # change for b2c questionarie — CHECK 13/14 (B2C master rules,
            # Section 26h): this backend has no genuine ranking calculation,
            # so the instrument no longer contains ranking items — trade-off
            # questions ask which ONE factor matters most instead. Run a
            # MaxDiff exercise separately if true rank-order data is needed.
            "This instrument uses single-choice trade-off questions rather "
            "than ranking exercises. Run a MaxDiff exercise separately if "
            "you need true rank-order data on driver importance.",
            "Profiling questions are for cross-tabs only and are excluded from "
            "the segmentation scoring model.",
        ],
    }
