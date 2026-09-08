"""CMI regions and their countries.

# change for b2c questionarie

The canonical CMI geography: five regions, each with the country list the
platform reports on. Supplied by the user (2026-08-11) and authoritative — the
`cmi_countries` table is not present in the Neon DB, so this file is the source
of truth for country-level work.

`TOP_COUNTRIES` names the two countries a survey is generated for in each
region. "Top" here means the largest consumer market in that region by
population-weighted spending power, which is stable across most consumer
categories; a category with a genuinely different centre of gravity can
override it per run.

Rest-of-* entries are reporting buckets, never survey targets: you cannot field
a consumer survey in "Rest of Europe".
"""
from __future__ import annotations

import os

REGION_COUNTRIES: dict[str, list[str]] = {
    "North America": ["U.S.", "Canada"],
    "Europe": [
        "Germany", "U.K.", "Spain", "France", "Italy", "Benelux", "Denmark",
        "Norway", "Sweden", "Russia", "Rest of Europe",
    ],
    "Asia Pacific": [
        "China", "Taiwan", "India", "Japan", "South Korea", "Indonesia",
        "Malaysia", "Philippines", "Singapore", "Australia",
        "Rest of Asia Pacific",
    ],
    "Latin America": [
        "Brazil", "Argentina", "Mexico", "Rest of Latin America",
    ],
    "Middle East & Africa": [
        "Bahrain", "Kuwait", "Oman", "Qatar", "Saudi Arabia",
        "United Arab Emirates", "Israel", "South Africa", "North Africa",
        "Central Africa",
    ],
}

# The two countries surveyed per region by default.
TOP_COUNTRIES: dict[str, list[str]] = {
    "North America": ["U.S.", "Canada"],
    "Europe": ["Germany", "U.K."],
    "Asia Pacific": ["China", "India"],
    "Latin America": ["Brazil", "Mexico"],
    "Middle East & Africa": ["Saudi Arabia", "United Arab Emirates"],
}

# Buckets that aggregate whatever is not named separately — never survey these.
_AGGREGATE_PREFIXES = ("rest of",)


def is_surveyable(country: str) -> bool:
    """False for reporting buckets like 'Rest of Europe'."""
    return not (country or "").strip().lower().startswith(_AGGREGATE_PREFIXES)


def countries_for(region: str, top_only: bool = True) -> list[str]:
    """Countries for a region — the default two, or every surveyable one.

    ``RUN_COUNTRIES`` narrows the fan-out for a test run, the same way
    ``RUN_REGIONS`` does: ``RUN_COUNTRIES=Germany`` generates one file instead
    of the region's usual two. Names not in this region are ignored, so one
    setting works across a multi-region run.
    """
    if top_only:
        picked = list(TOP_COUNTRIES.get(region) or [])
    else:
        picked = [c for c in REGION_COUNTRIES.get(region) or [] if is_surveyable(c)]

    raw = os.getenv("RUN_COUNTRIES", "").strip()
    if not raw:
        return picked
    wanted = {p.strip().lower() for p in raw.split(",") if p.strip()}
    # Match against every country in the region, not just the top two, so an
    # explicit request for a non-default country still runs.
    pool = [c for c in REGION_COUNTRIES.get(region) or [] if is_surveyable(c)]
    return [c for c in pool if c.lower() in wanted or slug(c) in wanted]


def region_for(country: str) -> str | None:
    """Which region a country belongs to."""
    needle = (country or "").strip().lower()
    for region, countries in REGION_COUNTRIES.items():
        if any(c.lower() == needle for c in countries):
            return region
    return None


def slug(name: str) -> str:
    """'U.K.' -> 'uk', 'Saudi Arabia' -> 'saudi-arabia'."""
    out = (name or "").lower().replace(".", "").replace("&", "and")
    return "-".join(out.split())


ALL_REGIONS = list(REGION_COUNTRIES.keys())
ALL_TOP_COUNTRIES = [c for cs in TOP_COUNTRIES.values() for c in cs]
