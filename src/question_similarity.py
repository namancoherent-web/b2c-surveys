"""Question-text similarity helpers for dedup (architect + validator).

# change for b2c questionarie — Phase A3
"""

from __future__ import annotations

import re

# change for b2c questionarie — stricter uniqueness (similar Qs must never ship).
# Near-identical → always drop (uncapped). Loose → also treated as hard fail.
NEAR_EXACT_JACCARD = 0.78
LOOSE_JACCARD = 0.58
_MIN_SHARED_TOKENS = 2

_STOP = {
    "how", "what", "which", "who", "when", "where", "why", "do", "does", "you",
    "your", "the", "of", "for", "and", "are", "is", "to", "in", "on", "with",
    "most", "often", "please", "select", "all", "that", "apply", "following",
    "best", "describes", "would", "have", "any", "this", "these", "from", "buy",
}


def norm_alnum(text: str) -> str:
    return "".join(ch for ch in (text or "").lower() if ch.isalnum())


def tokens(text: str) -> set[str]:
    return {
        w for w in re.findall(r"[a-z]+", (text or "").lower())
        if len(w) > 2 and w not in _STOP
    }


def token_set_jaccard(a: str, b: str, segment: str = "") -> float:
    """Distinctive-token Jaccard (segment tokens stripped).

    # change for b2c questionarie — short, wholly on-topic questions ("How often
    # do you buy organic milk?") lose every token to stopword + segment
    # stripping. Returning 0.0 there declared obvious paraphrases *maximally
    # dissimilar* and let them ship. Fall back to unstripped tokens instead.
    """
    seg = tokens(segment)
    ta = tokens(a) - seg
    tb = tokens(b) - seg
    if not ta or not tb:
        ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    shared = len(ta & tb)
    if shared < _MIN_SHARED_TOKENS and shared < min(len(ta), len(tb)):
        # Allow near-exact short questions when almost the whole set overlaps.
        if shared / len(ta | tb) < NEAR_EXACT_JACCARD:
            return shared / len(ta | tb)
    return shared / len(ta | tb)


def option_jaccard(opts_a: list, opts_b: list) -> float:
    sa = {norm_alnum(str(o)) for o in (opts_a or []) if str(o or "").strip()}
    sb = {norm_alnum(str(o)) for o in (opts_b or []) if str(o or "").strip()}
    sa.discard("")
    sb.discard("")
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def is_near_exact(
    text_a: str,
    text_b: str,
    opts_a: list | None = None,
    opts_b: list | None = None,
    segment: str = "",
) -> bool:
    """True when two questions are near-identical (must always be deduped)."""
    if norm_alnum(text_a) and norm_alnum(text_a) == norm_alnum(text_b):
        return True
    sim = token_set_jaccard(text_a, text_b, segment)
    if sim >= NEAR_EXACT_JACCARD:
        return True
    # High text overlap + high option overlap → near-exact paraphrase.
    if sim >= 0.70 and option_jaccard(opts_a or [], opts_b or []) >= 0.60:
        return True
    return False


def is_loose_similar(text_a: str, text_b: str, segment: str = "") -> bool:
    sim = token_set_jaccard(text_a, text_b, segment)
    return sim >= LOOSE_JACCARD and sim < NEAR_EXACT_JACCARD


def too_similar_to_any(
    text: str,
    prior_texts: list[str],
    segment: str = "",
    threshold: float = NEAR_EXACT_JACCARD,
) -> bool:
    """True if text is near-duplicate of any prior (used during generation)."""
    key = norm_alnum(text)
    if not key:
        return True
    for prior in prior_texts:
        if norm_alnum(prior) == key:
            return True
        if token_set_jaccard(text, prior, segment) >= threshold:
            return True
    return False
