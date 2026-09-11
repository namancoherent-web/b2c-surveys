"""Per-country brand allowlist, keyed on (country, product_category).

# change for b2c questionarie -- CHECK (spec: b2c_survey_question_structure
# _v3.txt Part C + survey_wording_and_brand_policy_update.txt Part B,
# 2026-09-11)

Brand names were previously banned outright in this generator. That ban is
lifted, but a brand is a FACTUAL CLAIM ABOUT A MARKET: if a brand appears in
a survey for a country where it is not sold, the respondent could not have
bought it, so any response distribution attached to it is fabricated and the
whole survey is invalid.

The spec is explicit that the model is not the source of truth --
"Every brand written into a survey must be traceable to a verification step,
not to the model's memory" and "Never guess. Unverified means excluded."

So this module works in two stages:

    1. SEARCH   -- the existing DDG pool retrieves real pages about brands
                   sold in that country for that category. This is evidence,
                   not opinion.
    2. JUDGE    -- an LLM is shown ONLY those search results and asked to
                   apply the five region gates to each candidate. It may
                   only pass a brand it can point to in the evidence, and
                   each verdict carries the source URL.

If either stage yields nothing usable the lookup returns EMPTY, and an empty
result is a hard instruction to the rest of the pipeline: generate brand-free
for this survey and set ``brand_verified = false``. There is deliberately no
code path that invents a brand, and none that repairs a failed brand check by
substituting a different brand (the spec forbids both).

Results are cached per (country, category) so a country's list is built once
and reused, and so a run is reproducible.
"""

from __future__ import annotations

import logging
import re

from src import cache_store
from src.date_utils import current_date_context

logger = logging.getLogger(__name__)

# Cache namespace + schema version. Bump the version to invalidate every
# previously-cached allowlist (e.g. after changing the gate prompt).
_NS = "brand_allowlist"
_SCHEMA_V = "brands_v1"

# The spec caps a brand question at 4-6 options INCLUDING the catch-all, so
# at most 5 real brands are ever needed.
MAX_BRANDS = 5

# Catch-all option labels (spec B4 / v3 Part C). The first is mandatory on
# every brand question; the second is for consideration-set slots only.
CATCHALL = "Another brand not listed here"
CATCHALL_NO_RECALL = "Do not remember"

# Slots permitted to carry brands (spec B1). Everything else is brand-free.
# Keyed by REQUIRED_SLOTS slot_key so the validator can gate on it directly.
BRAND_SLOTS: frozenset = frozenset({
    "item_variant",              # "Brand and variant owned" in v3 Part B
    "acquisition_channel",       # retailer / channel name
    "alternative_consideration", # consideration set
})

# Spec B1: "Cap: no more than 3 brand-bearing questions per survey."
MAX_BRAND_QUESTIONS = 3


_VERIFY_PROMPT = """\
You are verifying which brands may legally and factually appear in a consumer
survey about {category} written for {country}. Today is {today}.

You are NOT being asked what brands you know. You are being asked to judge the
SEARCH RESULTS BELOW. A brand you believe exists but cannot point to in these
results must be failed. Inventing a brand, or a local sub-brand, invalidates
the entire survey.

SEARCH RESULTS:
{evidence}

For each brand that plausibly appears in the results, apply all FIVE gates for
{country} specifically:

  1 AVAILABILITY   Sold through normal retail or distribution in {country}.
                   Grey imports, personal imports and resale listings do NOT
                   count.
  2 CATEGORY MATCH Sells {category} in {country}. Presence in {country} for a
                   DIFFERENT category does not qualify.
  3 ACTIVE         Has not exited, been withdrawn, been banned, or dropped
                   this category in {country}.
  4 LOCAL NAME     Written exactly as it is marketed in {country}. Some brands
                   trade under different names in different markets - use the
                   local trading name, not the global one.
  5 REAL PRESENCE  Meaningful distribution, not a token listing. A brand with
                   negligible presence must not be given an option slot that
                   implies a measurable share.

Set a gate false whenever the evidence does not positively support it. "Probably
available" is false. "It is a global brand so it must be everywhere" is false.
A region is not a market - evidence about a neighbouring country in the same
region does not establish presence in {country}.

Also give presence_rank: 1 for the clear market leader in {country}, then 2, 3
and so on by plausible market presence as shown in the evidence.

Put the URL you relied on in evidence_url. A verdict with every gate true but
no evidence_url will be discarded.
"""


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


def _dedupe_same_company(brands: list[dict]) -> list[dict]:
    """Drop brands that are the same company written two ways (spec B4).

    Conservative on purpose: only collapses when one name contains the other
    as a whole word (e.g. "Oral-B" vs "Oral B Pro"), never on fuzzy overlap.
    """
    kept: list[dict] = []
    for b in brands:
        name = _norm(b.get("brand"))
        if not name:
            continue
        clash = False
        for k in kept:
            other = _norm(k.get("brand"))
            a, z = sorted((name, other), key=len)
            if a and re.search(rf"\b{re.escape(a)}\b", z):
                clash = True
                break
        if not clash:
            kept.append(b)
    return kept


def _search_evidence(country: str, category: str, max_results: int = 6) -> list[dict]:
    """Real search hits about brands sold in this country for this category."""
    try:
        from src.searchers.ddg_pool import DDGWorkerPool
    except Exception as exc:  # noqa: BLE001
        logger.warning("brand allowlist: search unavailable (%s)", exc)
        return []

    queries = [
        f"best selling {category} brands in {country}",
        f"top {category} brands market share {country}",
        f"buy {category} {country} brands available",
    ]
    pool = DDGWorkerPool(workers=3)
    facts: list[dict] = []
    for q in queries:
        try:
            facts.extend(
                pool.search_to_findings(
                    q, max_results=max_results, segment=category,
                    serves_goal=f"brands sold in {country}", angle="brand availability",
                )
            )
        except Exception as exc:  # noqa: BLE001 — one bad query must not sink the lookup
            logger.warning("brand allowlist: query failed (%s): %s", q, exc)
    return facts


def _evidence_block(facts: list[dict], limit: int = 40) -> str:
    lines = []
    for f in facts[:limit]:
        txt = (f.get("fact") or "")[:400]
        url = f.get("source_ref") or ""
        if txt:
            lines.append(f"- {txt}\n  source: {url}")
    return "\n".join(lines) or "(no results)"


def lookup(country: str, product_category: str, *, allow_search: bool = True) -> dict:
    """Verified brands for one (country, category) pair.

    Returns::

        {"brands": [{"brand", "presence_rank", "evidence_url"}, ...],
         "verified_on": "<iso date>" | None,
         "country": ..., "category": ...}

    An EMPTY ``brands`` list is a hard instruction to generate brand-free and
    set ``brand_verified = false``. Callers must never fill that gap
    themselves.
    """
    country = (country or "").strip()
    product_category = (product_category or "").strip()
    empty = {
        "brands": [], "verified_on": None,
        "country": country, "category": product_category,
    }
    if not country or not product_category:
        return empty

    key = cache_store.make_key(_SCHEMA_V, _norm(country), _norm(product_category))
    cached = cache_store.get_json(_NS, key)
    if cached is not None:
        return cached
    if not allow_search:
        return empty

    facts = _search_evidence(country, product_category)
    if not facts:
        logger.info(
            "brand allowlist: no evidence for (%s, %s) -> brand-free",
            country, product_category,
        )
        cache_store.set_json(_NS, key, empty)
        return empty

    try:
        from src.llm import get_structured_llm
        from src.models import BrandVerification

        llm = get_structured_llm(BrandVerification, temperature=0.0, max_tokens=3000)
        result: BrandVerification = llm.invoke(
            _VERIFY_PROMPT.format(
                country=country, category=product_category,
                today=current_date_context()["iso"],
                evidence=_evidence_block(facts),
            )
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("brand allowlist: verification failed (%s) -> brand-free", exc)
        cache_store.set_json(_NS, key, empty)
        return empty

    passed = [
        {
            "brand": v.brand.strip(),
            "presence_rank": int(v.presence_rank or 99),
            "evidence_url": (v.evidence_url or "").strip(),
            "reason": (v.reason or "").strip(),
        }
        for v in (result.verdicts or ())
        # ALL five gates, plus a real source URL. A brand that passes every
        # gate but cites nothing is exactly the model-memory case the spec
        # forbids, so it is dropped here rather than trusted.
        if v.brand and v.available and v.category_match and v.active
        and v.local_name_correct and v.meaningful_presence
        and (v.evidence_url or "").strip().startswith("http")
    ]
    passed = _dedupe_same_company(passed)
    passed.sort(key=lambda b: b["presence_rank"])
    passed = passed[:MAX_BRANDS]

    out = {
        "brands": passed,
        "verified_on": current_date_context()["iso"] if passed else None,
        "country": country,
        "category": product_category,
    }
    cache_store.set_json(_NS, key, out)
    logger.info(
        "brand allowlist: (%s, %s) -> %d verified brand(s)",
        country, product_category, len(passed),
    )
    return out


def brand_names(country: str, product_category: str, **kw) -> list[str]:
    """Just the ordered brand names; empty list means generate brand-free."""
    return [b["brand"] for b in lookup(country, product_category, **kw).get("brands", [])]


def is_brand_slot(slot_key: str) -> bool:
    """True when this REQUIRED_SLOTS slot may carry brand names (spec B1)."""
    return slot_key in BRAND_SLOTS
