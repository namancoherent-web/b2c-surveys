"""Shared helpers for building, tagging, and de-duplicating evidence facts.

Used by both the database side (``src/db.py``) and the web side
(``src/nodes/evidence_harvester.py``) so every fact — wherever it comes from —
has the same shape and the same numeric-extraction rules.

Unified evidence fact shape
---------------------------
::

    {
      "fact": str,            # the cleaned statement/snippet (truncated to ~2000 chars)
      "value": str | None,    # extracted numeric/percentage/currency, if any
      "is_numeric": bool,     # True if it carries a usable figure
      "topic": str,           # short topic label ("regional share", "pricing", ...)
      "tab_hint": str | None, # which of the 4 survey tabs it likely serves
      "origin": str,          # "cmi_database" | "web" | "stat" | "reddit" | "exa" | "youtube" | "twitter"
      "authority": str,       # "high" | "medium" | "low"
      "source_ref": str,      # "report:<id>" / report title+section, or URL
      "title": str,           # report title+section, or page title
      "retrieved_at": str     # ISO-8601 UTC timestamp
    }
"""

import re
from datetime import datetime, timezone
from typing import List, Optional
from urllib.parse import urlparse

# Authoritative / statistical hosts → origin="stat" when matched.
STAT_DOMAINS = (
    ".gov", "census", "eurostat", "ec.europa.eu", "statista", "oecd",
    "who.int", "worldbank", "imf.org", "un.org", "bls.gov", "statcan",
    "ons.gov", "pewresearch", "nielsen", "gartner", "mckinsey", "data.gov",
)

_ORIGIN_AUTHORITY = {
    "stat": "high",
    "cmi_database": "high",
    "exa": "medium",
    "web": "medium",
    "reddit": "low",
    # change for b2c questionarie — social voice sources (language only)
    "youtube": "low",
    "twitter": "low",
}

# Origins that may NEVER ground numeric answer percentages.
VOICE_ONLY_ORIGINS = frozenset({"reddit", "youtube", "twitter"})

# Maximum characters kept per fact (the indexer ranks/trims further downstream).
MAX_FACT_CHARS = 2000

# Default ceiling on total facts so Node 3 isn't overwhelmed; numeric facts win.
DEFAULT_FACT_CEILING = 120

# --- Numeric / figure extraction --------------------------------------------
# Match the FIRST figure that looks like real evidence: currency amounts,
# percentages, CAGR, or magnitude numbers (million/billion/etc.).
_NUMERIC_RE = re.compile(
    r"""
    (?:US\$|\$|€|£|¥|INR|Rs\.?)\s?\d[\d,]*\.?\d*\s?
        (?:trillion|billion|million|thousand|bn|mn|k)?   # currency amount
    | \d[\d,]*\.?\d*\s?%                                  # percentage
    | CAGR\s+of\s+\d[\d.]*\s?%?                           # CAGR phrase
    | \d[\d,]*\.?\d*\s?(?:trillion|billion|million|thousand|bn|mn)  # magnitude
    """,
    re.IGNORECASE | re.VERBOSE,
)


def extract_value(text: str) -> Optional[str]:
    """Return the first numeric/percentage/currency substring, or None."""
    if not text:
        return None
    match = _NUMERIC_RE.search(text)
    return match.group(0).strip() if match else None


# --- Topic / tab inference ---------------------------------------------------
# Keyword → which of the 4 fixed tabs a fact most likely serves.
_TAB_KEYWORDS = {
    "consumer_profile": (
        "demographic", "age", "income", "gender", "household", "population",
        "who buys", "education", "urban", "rural", "generation", "millennial",
    ),
    "buying_behavior": (
        "purchase", "buy", "buying", "frequency", "channel", "spend", "budget",
        "decision", "online", "retail", "e-commerce", "distribution", "basket",
    ),
    "preferences_expectations": (
        "prefer", "feature", "expectation", "willing to pay", "priority",
        "brand", "attribute", "flavour", "flavor", "packaging", "premium",
    ),
    "satisfaction_future_intent": (
        "satisfaction", "loyalty", "nps", "switch", "future", "intent",
        "forecast", "cagr", "growth", "trend", "outlook", "repurchase",
    ),
}

_TOPIC_KEYWORDS = (
    ("regional share", ("region", "regional", "geograph", "north america", "asia", "europe")),
    ("market size", ("market size", "market value", "valued at", "worth", "revenue")),
    ("growth/CAGR", ("cagr", "growth rate", "forecast", "outlook")),
    ("segment share", ("segment", "share", "split", "by type", "by product")),
    ("pricing", ("price", "pricing", "cost", "premium", "tier")),
    ("purchase frequency", ("frequency", "how often", "per week", "per month")),
    ("demographics", ("demographic", "age", "income", "gender", "household")),
    ("preferences", ("prefer", "preference", "feature", "attribute")),
)


def infer_tab_hint(text: str) -> Optional[str]:
    """Best-guess which of the 4 tabs a fact serves, or None if unclear."""
    if not text:
        return None
    low = text.lower()
    best, best_score = None, 0
    for tab, kws in _TAB_KEYWORDS.items():
        score = sum(1 for kw in kws if kw in low)
        if score > best_score:
            best, best_score = tab, score
    return best


def is_stat_domain(url: str) -> bool:
    """True when the URL host matches a known statistical/government domain."""
    try:
        host = urlparse(url).netloc.lower()
    except Exception:  # noqa: BLE001
        return False
    return any(d in host for d in STAT_DOMAINS)


def infer_topic(text: str, fallback: str = "general") -> str:
    """Short topic label for a fact based on heading/snippet keywords."""
    if not text:
        return fallback
    low = text.lower()
    for label, kws in _TOPIC_KEYWORDS:
        if any(kw in low for kw in kws):
            return label
    return fallback


# --- Fact construction / dedup / capping ------------------------------------
def build_fact(
    *,
    fact: str,
    origin: str,
    source_ref: str,
    title: str,
    topic: Optional[str] = None,
    tab_hint: Optional[str] = None,
    value: Optional[str] = None,
    authority: Optional[str] = None,
    region: Optional[str] = None,
) -> Optional[dict]:
    """Assemble one unified evidence-fact dict.

    Truncates ``fact`` to ``MAX_FACT_CHARS``, extracts a numeric value if one
    wasn't supplied, and infers topic/tab_hint when not given. Returns None for
    empty input so callers can ``filter`` it out.
    """
    if not fact or not fact.strip():
        return None

    fact = fact.strip()[:MAX_FACT_CHARS]
    value = value if value is not None else extract_value(fact)
    topic = topic or infer_topic(title or fact)
    tab_hint = tab_hint if tab_hint is not None else infer_tab_hint(f"{title} {fact}")

    if authority is None:
        authority = _ORIGIN_AUTHORITY.get(origin, "low")

    out = {
        "fact": fact,
        "value": value,
        "is_numeric": value is not None,
        "topic": topic,
        "tab_hint": tab_hint,
        "origin": origin,
        "authority": authority,
        "source_ref": source_ref,
        "title": title,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
    }
    # change for b2c questionarie — region tag for A4b harvest
    if region:
        out["region"] = region
    return out


def build_voice_fact(
    *,
    fact: str,
    origin: str,
    source_ref: str,
    title: str,
    topic: Optional[str] = None,
    tab_hint: Optional[str] = None,
) -> Optional[dict]:
    """Build a consumer-language fact that can NEVER carry a numeric value.

    # change for b2c questionarie — social sources are language/pain only
    """
    f = build_fact(
        fact=fact,
        origin=origin,
        source_ref=source_ref,
        title=title,
        topic=topic,
        tab_hint=tab_hint,
        value=None,
        authority="low",
    )
    if not f:
        return None
    f["value"] = None
    f["is_numeric"] = False
    return f


def dedupe_facts(facts: List[dict]) -> List[dict]:
    """Drop duplicates keyed on (source_ref, hash of first 120 fact chars)."""
    seen = set()
    out = []
    for f in facts:
        key = (f.get("source_ref"), hash((f.get("fact") or "")[:120]))
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
    return out


def cap_facts(facts: List[dict], ceiling: int = DEFAULT_FACT_CEILING) -> List[dict]:
    """Cap total facts at ``ceiling``, preferring numeric (figure-bearing) ones."""
    if len(facts) <= ceiling:
        return facts
    numeric = [f for f in facts if f.get("is_numeric")]
    non_numeric = [f for f in facts if not f.get("is_numeric")]
    return (numeric + non_numeric)[:ceiling]
