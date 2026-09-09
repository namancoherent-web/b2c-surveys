"""Survey shape configuration: tabs, per-tab counts, sample size, and mode.

# change for b2c questionarie

The survey is organized into four fixed tabs. Each tab holds a slice of the
total, and each tab maps to a distinct class of question:

    consumer_profile          — who the respondent is: demographics, firmographics,
                                role, household/company context, lifestyle, values.
    buying_behavior           — how they buy: triggers, frequency, channels, budget,
                                decision drivers, information sources, purchase journey.
    preferences_expectations  — what they want: feature priorities, brand/format
                                preferences, expectations, willingness to pay, dealbreakers.
    satisfaction_future_intent — how they feel and what's next: satisfaction, loyalty,
                                NPS-style sentiment, switching intent, future plans, trends.

GENERATION_MODE controls how the answer_estimator / validator behave:
    "quota"  — always emit the full per-tab count, inferring distributions where
               evidence is thin (volume-first).
    "honest" — only emit questions/options that can be reasonably grounded,
               potentially producing fewer questions (grounding-first).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# change for b2c questionarie — ensure RUN_REGIONS / counts load before parse
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Geographic granularity the survey may use — mirrors the frontend's region set
# (Global + five continents). Country-level questions/data are forbidden;
# country evidence generalizes to its parent region.
REGIONS = [
    "Global",
    "North America",
    "Europe",
    "Asia Pacific",
    "Latin America",
    "Middle East & Africa",
]

# Continental regions only (excludes Global) — used for regional variants.
GEOGRAPHIC_REGIONS = [r for r in REGIONS if r != "Global"]

# change for b2c questionarie — optional subset for e2e / cost-limited runs
# e.g. RUN_REGIONS=Europe,North America  → only those + A6 global blend
def _parse_run_regions() -> list[str]:
    raw = os.getenv("RUN_REGIONS", "").strip()
    if not raw:
        return list(GEOGRAPHIC_REGIONS)
    out: list[str] = []
    for part in raw.split(","):
        # Late import avoided — match inline against GEOGRAPHIC_REGIONS
        key = part.strip().lower()
        if not key:
            continue
        for r in GEOGRAPHIC_REGIONS:
            if r.lower() == key or r.lower().replace(" & ", "-").replace(" ", "-") == key:
                if r not in out:
                    out.append(r)
                break
    return out or list(GEOGRAPHIC_REGIONS)


RUN_GEOGRAPHIC_REGIONS = _parse_run_regions()

# Region name -> frontend geography slug (dataByRegion keys / publish filenames).
# change for b2c questionarie — MUST match cmi-platform-ai geographies.ts RegionSlug
REGION_SLUGS = {
    "Global": "global",
    "North America": "north-america",
    "Europe": "europe",
    "Asia Pacific": "asia-pacific",
    "Latin America": "latin-america",
    "Middle East & Africa": "middle-east-africa",
}

# Ordered publish list (index.json + <slug>/<region>.json). Includes global.
PUBLISH_REGION_SLUGS = [
    REGION_SLUGS["Global"],
    REGION_SLUGS["North America"],
    REGION_SLUGS["Europe"],
    REGION_SLUGS["Asia Pacific"],
    REGION_SLUGS["Latin America"],
    REGION_SLUGS["Middle East & Africa"],
]

# change for b2c questionarie — Phase A6: approx. consumer-market blend weights
# Keys MUST be PUBLISH_REGION_SLUGS entries (excludes global).
REGION_BLEND_WEIGHTS = {
    REGION_SLUGS["North America"]: 0.22,
    REGION_SLUGS["Europe"]: 0.22,
    REGION_SLUGS["Asia Pacific"]: 0.32,
    REGION_SLUGS["Latin America"]: 0.12,
    REGION_SLUGS["Middle East & Africa"]: 0.12,
}

# change for b2c questionarie — Phase A5: core + regional module (default)
REGION_QUESTION_MODE = os.getenv("REGION_QUESTION_MODE", "core_plus_module").strip().lower()
if REGION_QUESTION_MODE not in ("core_plus_module", "fully_distinct"):
    REGION_QUESTION_MODE = "core_plus_module"

# change for b2c questionarie — instrument length matched to published practice.
#
# Was 5–10/section (target 8 → ~32 + 6 profiling = 38 screens). That is roughly
# 15-16 min desktop / 20 min mobile at the industry ~2 questions-per-minute rate,
# which is past the point where real instruments stop.
#
# Benchmarks the target below is set against:
#   - Pew ATP W127 "Data Privacy" — a full topical study — is 30 stems, of which
#     24 are ASK ALL; filters and a split ballot mean one respondent answers
#     ~22-26. Pew's broadband tracker is 8.
#   - Qualtrics: drop-off climbs sharply past 12 min desktop / 9 min mobile.
#   - SurveyMonkey (100k surveys): steepest incremental drop-off is in the first
#     15 questions, and per-question dwell falls 30s → 19s by Q26 — the tail of a
#     long instrument is answered with measurably less deliberation.
#   - Sawtooth: segmentation separation DEGRADES as basis variables grow
#     (avg F-stat 67 at 20 vars → 50 at 40 → 32 at 80). Fewer, better-chosen
#     basis questions produce cleaner clusters, so cutting length does not cost
#     segment quality here — it helps it.
#
# Per the brief: at least 6 questions in any one section, and no more than 40
# across the whole instrument. 4 sections x 8 = 32 generated + 2 satisfaction
# anchors + 4 profiling = 38 screens, inside the 40 ceiling with room for a
# section that runs long. Profiling is excluded from the segment model
# (see standard_sections).
# change for b2c questionarie -- CHECK (user directive, 2026-09-09): the
# framework moved from a flat 4-per-section (16 total) to PER-SECTION counts
# 5 / 7 / 6 / 5 = 23 behavioural questions, plus 2 profiling (age, gender).
# Deliberately NOT env-configurable, per explicit instruction.
#
# A hard target with no slack risks the model padding with near-duplicates to
# hit the count. That is mitigated by REQUIRED_SLOTS in validator_critic.py,
# which names the EXACT question type each slot must carry and gates the run
# until every one exists, plus the construct-level duplicate detector in
# question_architect.py's generation loop as a backstop.
QUESTIONS_PER_SECTION = {
    "consumer_profile": 5,
    "buying_behavior": 7,
    "preferences_expectations": 6,
    "satisfaction_future_intent": 5,
}
# Flat fallbacks kept for the handful of call sites that still want a single
# number (global_selector's per-tab pick target, validator's under-target
# note). MIN is the smallest section so a legitimately short section is not
# reported as under floor; MAX/TARGET track the largest/most common.
QUESTIONS_PER_TAB_MIN = min(QUESTIONS_PER_SECTION.values())
QUESTIONS_PER_TAB_MAX = max(QUESTIONS_PER_SECTION.values())
QUESTIONS_PER_TAB_TARGET = QUESTIONS_PER_TAB_MAX

# Hard ceiling on the whole instrument, including the standard sections.
TOTAL_QUESTION_CAP = int(os.getenv("TOTAL_QUESTION_CAP", "45"))
# Floor — a survey shorter than this is short-changing the brief.
TOTAL_QUESTION_FLOOR = int(os.getenv("TOTAL_QUESTION_FLOOR", "35"))

# Clamp so misconfigured env cannot invert the band.
if QUESTIONS_PER_TAB_MIN < 1:
    QUESTIONS_PER_TAB_MIN = 1
if QUESTIONS_PER_TAB_MAX < QUESTIONS_PER_TAB_MIN:
    QUESTIONS_PER_TAB_MAX = QUESTIONS_PER_TAB_MIN
if not (QUESTIONS_PER_TAB_MIN <= QUESTIONS_PER_TAB_TARGET <= QUESTIONS_PER_TAB_MAX):
    QUESTIONS_PER_TAB_TARGET = max(
        QUESTIONS_PER_TAB_MIN,
        min(QUESTIONS_PER_TAB_MAX, QUESTIONS_PER_TAB_TARGET),
    )

# Core vs module split (target 4 → core 3 + module 1).
MODULE_PER_TAB = max(1, min(3, QUESTIONS_PER_TAB_TARGET // 3))
CORE_PER_TAB = max(QUESTIONS_PER_TAB_MIN - MODULE_PER_TAB, QUESTIONS_PER_TAB_TARGET - MODULE_PER_TAB)
if CORE_PER_TAB + MODULE_PER_TAB != QUESTIONS_PER_TAB_TARGET:
    CORE_PER_TAB = QUESTIONS_PER_TAB_TARGET - MODULE_PER_TAB

_TAB_NAMES = (
    "consumer_profile",
    "buying_behavior",
    "preferences_expectations",
    "satisfaction_future_intent",
)

# (tab_name, question_count) — order here defines tab order in the final survey.
# change for b2c questionarie -- CHECK (user directive, 2026-09-09): per-section
# counts, not a flat TARGET (5 / 7 / 6 / 5 = 23 behavioural).
CATEGORY_PLAN = [
    (name, QUESTIONS_PER_SECTION.get(name, QUESTIONS_PER_TAB_TARGET))
    for name in _TAB_NAMES
]

# Total questions targeted across all tabs (23 under the current framework).
TOTAL_TARGET = sum(count for _, count in CATEGORY_PLAN)

# Hard floor for quota mode (allow small shortfall vs target).
MIN_TOTAL_QUESTIONS = QUESTIONS_PER_TAB_MIN * len(_TAB_NAMES)

# Cache-hit requires at least this many questions (was hardcoded 90).
QUESTIONNAIRE_CACHE_MIN_QUESTIONS = MIN_TOTAL_QUESTIONS

# Synthetic respondent base used when rendering answer-percentage distributions.
SAMPLE_SIZE = 2500

# "quota" | "honest" — see module docstring.
GENERATION_MODE = "quota"


def match_geographic_region(name: str) -> str | None:
    """Return canonical GEOGRAPHIC_REGIONS name or None."""
    if not name:
        return None
    key = str(name).strip().lower()
    for r in GEOGRAPHIC_REGIONS:
        if r.lower() == key:
            return r
    # Also accept frontend slugs.
    for label, slug in REGION_SLUGS.items():
        if slug == key and label != "Global":
            return label
    return None


def region_to_slug(region: str) -> str:
    """Map display region name to publish/frontend slug."""
    if region in REGION_SLUGS:
        return REGION_SLUGS[region]
    matched = match_geographic_region(region)
    if matched:
        return REGION_SLUGS[matched]
    return (region or "global").lower().replace(" ", "-").replace("&", "")
