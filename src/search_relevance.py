"""Keep web hits on-topic for the market segment.

Fixes cases like ``air fryer`` matching aviation (Air India) because DDG/Exa
split on ``air``. Two levers:

1. ``product_phrase_for_query`` — quote the product words in search strings.
2. ``is_hit_relevant`` — drop hits that don't share enough product tokens with
   the segment (before expensive Jina fetches).
"""

from __future__ import annotations

import re

# Same spirit as db._SEGMENT_STOPWORDS — not product identity.
_STOP = frozenset({
    "global", "worldwide", "international", "market", "markets", "industry",
    "sector", "report", "size", "share", "growth", "trends", "trend",
    "analysis", "forecast", "outlook", "the", "and", "for", "of", "by",
    "product", "products", "service", "services", "solution", "solutions",
    "type", "types", "consumer", "customers", "customer", "buyers", "buyer",
})

# Tokens that often collide with unrelated domains when matched alone.
_WEAK = frozenset({
    "air", "ice", "pack", "smart", "watch", "home", "light", "high", "energy",
    "laser", "water", "filter", "gel", "electric", "vehicle", "power", "soft",
    "hard", "pro", "max", "plus", "mini", "ultra", "super", "best", "top",
    "new", "free", "online", "india", "china", "europe", "asia",
})


def product_tokens(segment: str) -> list[str]:
    """Significant product tokens from a segment label (order preserved)."""
    words = re.findall(r"[a-z0-9]+", (segment or "").lower())
    return [w for w in words if len(w) > 2 and w not in _STOP]


def product_phrase_for_query(segment: str) -> str:
    """Quoted multi-word product for keyword search; else bare token/segment."""
    toks = product_tokens(segment)
    if len(toks) >= 2:
        return '"' + " ".join(toks) + '"'
    if len(toks) == 1:
        return toks[0]
    return (segment or "").strip() or "market"


def is_hit_relevant(
    title: str = "",
    url: str = "",
    snippet: str = "",
    segment: str = "",
) -> bool:
    """True if a search hit looks related to the segment product.

    Multi-token products (e.g. air fryer): require the full phrase OR at least
    two product tokens in title/url/snippet. A lone weak token (``air``) is
    never enough — that is the Air India failure mode.
    """
    toks = product_tokens(segment)
    if not toks:
        return True

    blob = f"{title} {url} {snippet}".lower()
    # URL path often uses hyphens
    blob_norm = re.sub(r"[^a-z0-9]+", " ", blob)

    if len(toks) == 1:
        return toks[0] in blob_norm

    phrase = " ".join(toks)
    if phrase in blob_norm or "-".join(toks) in blob.replace(" ", ""):
        return True

    present = [t for t in toks if t in blob_norm]
    if len(present) >= 2:
        return True

    # Exactly one token hit: allow only if that token is distinctive (not weak)
    # and the other product tokens are not all weak either — still prefer 2+.
    if len(present) == 1 and present[0] not in _WEAK:
        # Single strong token (e.g. "sneakers") can pass; "air" cannot.
        return True

    return False
