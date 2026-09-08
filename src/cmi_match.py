"""Match a market segment to its CMI domain framework.

# change for b2c questionarie

The RAR defines the four survey categories per DOMAIN, not one generic set: a
food market opens on consumption occasions, an electronics market on device
ownership, a services market on service usage. Picking the right framework is
therefore the first thing the pipeline must decide, and it drives the section
names for the whole study.

Matching is deliberately cheap-first and never fatal:

    1. token overlap against the 201 B2C sub-sector rows  (no LLM, no network)
    2. token overlap against domain names                 (no LLM)
    3. Consumer Goods                                     (safe default)

An LLM tiebreak is available via ``classify_domain_llm`` for the ambiguous
cases, but the caller decides whether to pay for it — a wrong framework still
produces a valid survey, just with a less specific section vocabulary, so
blocking a run on taxonomy would be the wrong trade.
"""
from __future__ import annotations

import re
from functools import lru_cache

from src.cmi_frameworks import B2C_FRAMEWORKS, B2C_SUBSECTORS, CANONICAL_IDS

DEFAULT_DOMAIN = "Consumer Goods"

# Words that carry no signal about which domain a market belongs to.
_STOP = {
    "global", "market", "markets", "the", "a", "an", "of", "and", "for", "in",
    "by", "size", "share", "report", "analysis", "industry", "products",
    "product", "services", "service", "system", "systems", "solutions",
}


# The CMI taxonomy names categories formally ("Footwear", "Non-Alcoholic
# Beverages"); real market names use everyday words ("running shoes", "bottled
# water"). Without a bridge almost everything falls through to the default, so
# map the common consumer vocabulary onto the taxonomy's own terms.
_SYNONYMS = {
    "shoe": "footwear", "shoes": "footwear", "sneaker": "footwear",
    "sneakers": "footwear", "boot": "footwear", "trainers": "footwear",
    "yoghurt": "yogurt", "soda": "carbonated", "cola": "carbonated",
    "beer": "alcoholic", "wine": "alcoholic", "spirits": "alcoholic",
    "bottled": "beverages", "drink": "beverages", "drinks": "beverages",
    "smartwatch": "wearable", "smartwatches": "wearable",
    "watch": "wearable", "wearables": "wearable",
    "phone": "smartphones", "smartphone": "smartphones",
    "laptop": "computing", "tablet": "computing", "pc": "computing",
    "tv": "video", "television": "video", "streaming": "video",
    "car": "vehicles", "cars": "vehicles", "vehicle": "vehicles",
    "ev": "vehicles", "automobile": "vehicles",
    "serum": "skin", "serums": "skin", "skincare": "skin",
    "shampoo": "hair", "toothpaste": "oral",
    "mattress": "bedding", "mattresses": "bedding",
    "furniture": "furnishings", "cookware": "kitchenware",
    "gaming": "gaming", "game": "gaming", "games": "gaming",
    "ridehailing": "mobility", "taxi": "mobility", "rideshare": "mobility",
    "supplement": "supplements", "vitamin": "supplements",
    "snack": "snacks", "petfood": "pet",
    "fryer": "kitchenware", "blender": "kitchenware", "cooker": "kitchenware",
    "kettle": "kitchenware", "microwave": "kitchenware",
    "mattress": "bedding", "pillow": "bedding", "duvet": "bedding",
    "hailing": "mobility", "hail": "mobility",
    "meal": "food", "kit": "food", "grocery": "food", "groceries": "food",
    "shoe": "footwear", "footwear": "footwear",
}

# Words so generic that a match on them alone means nothing. They still count
# toward a two-word match, but never carry a single-word one.
_WEAK = {
    "air", "home", "care", "delivery", "digital", "smart", "mobile", "online",
    "personal", "consumer", "device", "devices", "app", "apps", "electric",
}


def _tokens(text: str) -> set[str]:
    """Lowercase content words, singularised crudely, plus taxonomy synonyms."""
    words = re.findall(r"[a-z0-9]+", (text or "").lower())
    out = set()
    for w in words:
        if w in _STOP or len(w) < 3:
            continue
        out.add(w)
        # Cheap singular so "shoes" matches "shoe", "batteries" -> "batterie"
        # is harmless because both sides get the same treatment.
        if w.endswith("s") and len(w) > 3:
            out.add(w[:-1])
        syn = _SYNONYMS.get(w)
        if syn:
            out.add(syn)
    return out


@lru_cache(maxsize=1)
def _subsector_tokens() -> list[tuple[set[str], dict]]:
    # Index the parent market too — "Apparel, Footwear & Fashion" carries the
    # word a shopper would use, while the sub-sector name may not.
    return [
        (_tokens(r["sub_sector"]) | _tokens(r.get("parent_market", "")), r)
        for r in B2C_SUBSECTORS
    ]


def _domain_key(name: str) -> str:
    """Normalise a domain name for lookup.

    Framework domains come from workbook FILENAMES ("Food Beverages") while
    sub-sector rows carry the display name from the sheet ("Food & Beverages").
    Without this the two never join and every food market fell back to the
    Consumer Goods default.
    """
    return re.sub(r"[^a-z0-9]+", "", (name or "").lower())


@lru_cache(maxsize=1)
def _framework_by_domain() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for f in B2C_FRAMEWORKS:
        out.setdefault(_domain_key(f["domain"]), []).append(f)
    return out


def _framework_for(domain: str, hint: str = "") -> dict | None:
    """Pick a framework for a domain, disambiguating multi-variant domains.

    Automotive carries four B2C variants (EV, aftermarket, mobility services,
    public transport). Score each variant's name against the market text and
    take the best; fall back to the first, which is the domain's primary.
    """
    variants = _framework_by_domain().get(_domain_key(domain)) or []
    if not variants:
        return None
    if len(variants) == 1:
        return variants[0]
    hint_tokens = _tokens(hint)
    best, best_score = variants[0], 0
    for v in variants:
        score = len(hint_tokens & _tokens(v["classification"]))
        if score > best_score:
            best, best_score = v, score
    return best


def match_domain(segment: str) -> dict:
    """Return the best framework for this market.

    Always returns a usable framework — never raises, never returns None.
    The ``confidence`` field tells the caller how much to trust it:
      * ``subsector`` — matched a named CMI sub-sector (strongest)
      * ``domain``    — matched a domain name only
      * ``default``   — nothing matched, using Consumer Goods
    """
    seg_tokens = _tokens(segment)

    # 1. Strongest signal: a named sub-sector in the CMI taxonomy. Score on the
    # share of the SEGMENT's own words that were recognised — the sub-sector
    # side varies in length (it now includes the parent market), so scoring
    # against it would penalise correct matches on broad parents.
    best_row, best_score = None, 0
    for toks, row in _subsector_tokens():
        if not toks:
            continue
        overlap = seg_tokens & toks
        if not overlap:
            continue
        # A single shared generic word is noise, not a match: "air fryer" hit
        # "Air Care & Home Fresheners" on "air", and "meal kit delivery" hit
        # "Drug Delivery Systems" on "delivery". Require either two shared
        # words, or one shared word that appears in the sub-sector's OWN name
        # (not merely its parent market).
        own = _tokens(row["sub_sector"])
        meaningful = overlap - _WEAK
        if not meaningful:
            continue  # matched only on filler like "air", "smart", "delivery"
        strong = len(overlap) >= 2 or bool(meaningful & own)
        if not strong:
            continue
        score = len(overlap) * 100 // max(len(seg_tokens), 1)
        # Break ties toward the row whose own name (not just its parent)
        # carried the match — that is the more specific classification.
        score += 5 * len(overlap & own)
        if score > best_score:
            best_row, best_score = row, score

    if best_row and best_score >= 30:
        domain = best_row["domain"].strip()
        fw = _framework_for(domain, segment)
        if fw:
            return {
                "domain": domain,
                "classification": fw["classification"],
                "sections": dict(fw["sections"]),
                "sub_sector": best_row["sub_sector"],
                "attributes": dict(best_row.get("attributes") or {}),
                "confidence": "subsector",
            }

    # 2. Weaker: the market text names a domain outright ("consumer services").
    for variants in _framework_by_domain().values():
        domain = variants[0]["domain"]
        if _tokens(domain) & seg_tokens:
            fw = _framework_for(domain, segment)
            if fw:
                return {
                    "domain": domain,
                    "classification": fw["classification"],
                    "sections": dict(fw["sections"]),
                    "sub_sector": "",
                    "attributes": {},
                    "confidence": "domain",
                }

    # 3. Default. Consumer Goods carries the generic four categories, which are
    # correct for most consumer markets even when the domain is unrecognised.
    fw = _framework_for(DEFAULT_DOMAIN, segment) or {}
    return {
        "domain": DEFAULT_DOMAIN,
        "classification": fw.get("classification", "B2C"),
        "sections": dict(fw.get("sections") or {}),
        "sub_sector": "",
        "attributes": {},
        "confidence": "default",
    }


def section_labels(match: dict) -> list[tuple[str, str]]:
    """[(canonical_id, label), ...] in arc order for a matched framework."""
    sections = match.get("sections") or {}
    return [(cid, sections[cid]) for cid in CANONICAL_IDS if cid in sections]


def must_cover_topics(match: dict) -> dict[str, list[str]]:
    """Per-section seed topics from the sub-sector's Top-5 attributes."""
    return {k: list(v) for k, v in (match.get("attributes") or {}).items() if v}


# One-line remit per canonical section, appended to the CMI label so the
# planner knows what the section is FOR when it writes beats.
_REMITS = {
    "consumer_profile": (
        "who uses the product and how — usage context, occasions, frequency, "
        "who is involved"
    ),
    "buying_behavior": (
        "how they shop for and buy it — trigger, research, channel, spend, "
        "decision drivers"
    ),
    "preferences_expectations": (
        "what they need from the product — must-haves, quality cues, fit, "
        "claims, trade-offs"
    ),
    "satisfaction_future_intent": (
        "outcomes — satisfaction, problems, switching, advocacy, future "
        "purchase intent"
    ),
}


def blueprint_sections(match: dict) -> list[dict]:
    """Framework sections in the shape ``narrative.normalize_sections`` wants.

    Passing this as ``sections_override`` pins the CMI category names for the
    whole study, so the planner only chooses beats INSIDE fixed sections rather
    than inventing section names per run.
    """
    out: list[dict] = []
    for order, (cid, label) in enumerate(section_labels(match), start=1):
        out.append({
            "section_id": cid,
            "label": label,
            "canonical": cid,
            "order": order,
            "remit": _REMITS.get(cid, ""),
        })
    return out
