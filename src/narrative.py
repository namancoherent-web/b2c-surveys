"""Survey storyline — the narrative arc a questionnaire must follow.

# change for b2c questionarie — Phase A8 (story order)

A questionnaire is not a bag of questions: it is a sequence. This module owns
that sequence.

Two levels:

1. **Sections** (the four tabs) are FIXED and ordered — profile → behavior →
   preferences → satisfaction. That macro arc is already a standard consulting
   funnel and the frontend hardcodes the four section ids.
2. **Beats** are the ordered narrative slots INSIDE a section (e.g.
   ``category_entry → usage_occasion → usage_frequency``). Beats are generated
   per segment *and per region* by ``nodes/story_planner.py`` so the story fits
   the actual category and local shopping reality.

Every generated beat also carries a ``canonical`` tag drawn from
``CANONICAL_BEATS`` below. Region-local beats may be renamed, reordered, added
or dropped, but the canonical tag is what lets Phase A6 line the same narrative
moment up across five independently-planned regions. Without it, a per-region
storyline would make the global blend incoherent.

Ordering contract
-----------------
Question order within a tab is ALWAYS ``(beat order, hint within beat,
generation order)`` — never the raw model output and never list position. See
``order_questions``.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

# The CANONICAL sections. These are no longer the only possible sections — a
# segment's blueprint may rename, reorder, add or drop them — but they remain
# the shared vocabulary, the cross-region/cross-market anchor, and the fallback
# when the planner is unavailable. Every generated section maps onto one of
# these via its ``canonical`` tag, exactly as beats do.
TAB_ORDER = (
    "consumer_profile",
    "buying_behavior",
    "preferences_expectations",
    "satisfaction_future_intent",
)

TAB_SECTION_IDS = {
    "consumer_profile": "profile",
    "buying_behavior": "behavior",
    "preferences_expectations": "preferences",
    "satisfaction_future_intent": "satisfaction",
}

# (canonical_id, default label, what it must cover)
CANONICAL_SECTIONS: tuple[tuple[str, str, str], ...] = (
    ("consumer_profile", "Consumer Profile",
     "how the respondent relates to and uses the category — usage context, "
     "occasions, frequency, who is involved, need state"),
    ("buying_behavior", "Buying Behavior",
     "how they shop for and buy it — triggers, discovery, channel, cycle, "
     "spend, decision drivers"),
    ("preferences_expectations", "Preferences & Expectations",
     "what they want from the product — must-haves, quality cues, format, "
     "claims, trade-offs, value"),
    ("satisfaction_future_intent", "Satisfaction & Future Intent",
     "outcomes — satisfaction, friction, switching, advocacy, future use"),
)

MIN_SECTIONS = 3
MAX_SECTIONS = 6

# Respondent segments the study must be able to tell apart.
# change for b2c questionarie — exactly 4 behavioural segments per study, so
# the production site always renders the same number of groups. The RAR defines
# the four SURVEY CATEGORIES (which become the sections); it says nothing about
# respondent segments, so those stay behavioural and market-specific — a new
# set of names for every market — just fixed in count.
MIN_SEGMENTS = 4
MAX_SEGMENTS = 4


def canonical_section_ids() -> tuple[str, ...]:
    return tuple(s[0] for s in CANONICAL_SECTIONS)


def canonical_section_rank() -> dict[str, int]:
    return {s[0]: i for i, s in enumerate(CANONICAL_SECTIONS)}


def default_sections() -> list[dict]:
    """The canonical four, as blueprint section dicts."""
    return [
        {
            "section_id": sid,
            "label": label,
            "remit": remit,
            "canonical": sid,
            "order": i + 1,
        }
        for i, (sid, label, remit) in enumerate(CANONICAL_SECTIONS)
    ]


def sections_for(blueprint: dict | None, *, include_standard: bool = False) -> list[dict]:
    """Ordered sections for this blueprint, falling back to the canonical four.

    ``include_standard`` appends Profiling as the closing section. Screening is
    not part of the questionnaire — qualification is a fielding concern, so the
    survey opens on its first substantive question.
    """
    secs = (blueprint or {}).get("sections") or []
    if not secs:
        secs = default_sections()
    secs = sorted(secs, key=lambda s: int(s.get("order") or 999))
    if not include_standard:
        return secs
    from src import standard_sections as _std

    return [*secs, _std.profiling_section()]


def section_ids(blueprint: dict | None) -> list[str]:
    return [s["section_id"] for s in sections_for(blueprint)]


def section_label(blueprint: dict | None, section_id: str) -> str:
    for s in sections_for(blueprint):
        if s["section_id"] == section_id:
            return s.get("label") or section_id.replace("_", " ").title()
    return section_id.replace("_", " ").title()


def section_canonical(blueprint: dict | None, section_id: str) -> str:
    """Canonical section a generated section maps onto (for A6 + fallbacks)."""
    for s in sections_for(blueprint):
        if s["section_id"] == section_id:
            return s.get("canonical") or section_id
    return section_id if section_id in canonical_section_rank() else "consumer_profile"

MIN_BEATS_PER_TAB = 3
MAX_BEATS_PER_TAB = 8

# Sort key for a question whose beat_id matches no beat in the blueprint —
# unknown beats sink to the end of the tab but keep their relative order.
_UNKNOWN_BEAT_ORDER = 10_000


# ---------------------------------------------------------------------------
# Canonical arc — the shared vocabulary, the cross-region anchor, and the
# fallback storyline when the planner LLM is unavailable.
# ---------------------------------------------------------------------------
CANONICAL_BEATS: dict[str, tuple[tuple[str, str, str], ...]] = {
    "consumer_profile": (
        # change for b2c questionarie -- CHECK (Section 27a): "the category"
        # is ambiguous between the PRODUCT (running shoes) and the activity
        # or occasion it serves (running) -- confirmed a Running Shoes
        # survey's opening beat filled with running-frequency/purpose
        # questions that never mentioned shoes at all, so nothing anchored
        # the survey to its actual subject until question 5. The product
        # itself, not its surrounding activity, must open the survey.
        ("category_entry", "Category relationship",
         "Whether and how the respondent uses/owns/buys the PRODUCT ITSELF "
         "(the exact category being surveyed) — not the broader activity, "
         "occasion or lifestyle around it. For a product used to DO "
         "something (running shoes, a fitness tracker), ask about the "
         "product (ownership, which type, how many), not the activity "
         "(how often you run, why you exercise) — activity-context beats "
         "come later. This is the widest, safest opening question in the "
         "whole survey, and it must name the product explicitly enough "
         "that no respondent could mistake it for a survey about something "
         "else the same activity happens to involve."),
        ("usage_occasion", "Usage occasions",
         "When, where and why the product is used; the situations that bring "
         "them into the category."),
        ("usage_frequency", "Usage frequency and intensity",
         "How often and how much they use it, in a concrete recent timeframe."),
        ("household_role", "Who is involved",
         "Who in the household uses it and who decides to buy it."),
        ("need_state", "Need state and motivation",
         "The underlying need, lifestyle or life stage driving category use."),
    ),
    "buying_behavior": (
        ("purchase_trigger", "What starts a purchase",
         "The trigger that turns a latent need into an actual purchase occasion."),
        ("discovery", "Discovery and information",
         "Where they learn about and compare options before buying."),
        ("channel", "Where they buy",
         "Retail formats and channels used to buy this category."),
        ("purchase_cycle", "Purchase frequency and cycle",
         "How often they buy, restock or replace."),
        ("spend", "Category spend",
         "Typical spend band on this category — never household income."),
        ("decision_drivers", "Decision drivers",
         "What actually decides the choice at the shelf or checkout."),
        ("repurchase", "Repeat and routine",
         "Whether buying is habitual, planned, or reconsidered every time."),
    ),
    "preferences_expectations": (
        ("must_haves", "Must-have attributes",
         "Non-negotiable product requirements."),
        ("quality_cues", "Quality cues",
         "Sensory or functional signals used to judge quality."),
        ("format_packaging", "Format and packaging",
         "Preferred sizes, formats and pack types."),
        ("claims_credentials", "Claims and credentials",
         "Certifications, origin and sustainability claims that carry weight."),
        ("tradeoffs", "Trade-offs",
         "What they give up when they cannot have everything at once."),
        ("value_expectation", "Value and willingness to pay",
         "Price and value expectations — the most sensitive item in this tab, "
         "so it belongs late."),
    ),
    "satisfaction_future_intent": (
        ("overall_satisfaction", "Overall satisfaction",
         "The headline satisfaction reading for the category."),
        ("satisfaction_drivers", "What drives satisfaction",
         "Which specific aspects satisfy or disappoint."),
        ("friction", "Friction and unmet needs",
         "Problems, frustrations and gaps in the current experience."),
        ("switching", "Switching triggers",
         "What would make them change what they buy, at category level."),
        ("advocacy", "Recommend intent",
         "Likelihood to recommend the category or their current choice."),
        ("future_use", "Future category use",
         "Where their use of the category is heading."),
    ),
}


def _clip(text: Any, limit: int) -> str:
    """Trim to a word boundary with an ellipsis — never mid-word.

    # change for b2c questionarie — a hard slice produced "It closes with satis".
    """
    s = str(text or "").strip()
    if len(s) <= limit:
        return s
    cut = s[:limit]
    space = cut.rfind(" ")
    if space > limit * 0.6:
        cut = cut[:space]
    return cut.rstrip(" ,;:-") + "…"


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")
    return s[:48]


def canonical_ids(tab: str) -> tuple[str, ...]:
    """Canonical beat ids for a tab, in canonical narrative order."""
    return tuple(b[0] for b in CANONICAL_BEATS.get(tab, ()))


def canonical_order(tab: str) -> dict[str, int]:
    """Canonical beat id -> its position in the canonical arc."""
    return {bid: i for i, bid in enumerate(canonical_ids(tab))}


def canonical_beats(tab: str) -> list[dict]:
    """Canonical arc for one tab as beat dicts."""
    return [
        {
            "beat_id": bid,
            "title": title,
            "intent": intent,
            "canonical": bid,
            "order": i + 1,
        }
        for i, (bid, title, intent) in enumerate(CANONICAL_BEATS.get(tab, ()))
    ]


def default_storyline() -> dict[str, list[dict]]:
    """The canonical arc as a full blueprint — planner fallback."""
    return {tab: canonical_beats(tab) for tab in TAB_ORDER}


def default_blueprint(segment: str = "", region: str = "") -> dict:
    """A complete blueprint object using the canonical arc."""
    return {
        "segment": segment,
        "region": region,
        "study_purpose": "",
        "segments": [],
        "sections": default_sections(),
        "storyline": default_storyline(),
        "narrative_summary": (
            "Standard consumer research arc: establish category relationship, "
            "then how they buy, then what they want, then how they feel and "
            "what they will do next."
        ),
        "adaptation_notes": ["Canonical arc — no segment adaptation applied."],
        "source": "default",
    }


# ---------------------------------------------------------------------------
# Blueprint normalization
# ---------------------------------------------------------------------------
def normalize_sections(raw: Any) -> list[dict]:
    """Sanitize a planner's section list.

    Guarantees unique slugged ids, a valid ``canonical`` on every section, a
    count inside [MIN_SECTIONS, MAX_SECTIONS], and dense 1..N ordering. Two
    sections may not claim the same canonical — that would make cross-region and
    cross-market alignment ambiguous.
    """
    valid_canon = set(canonical_section_ids())
    out: list[dict] = []
    seen_ids: set[str] = set()
    seen_canon: set[str] = set()

    for item in (raw or []):
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip()
        sid = _slug(item.get("section_id") or label)
        if not sid or not label or sid in seen_ids:
            continue
        canon = _slug(item.get("canonical") or "")
        if canon not in valid_canon or canon in seen_canon:
            # Fall back to the next unclaimed canonical, keeping the arc's order.
            canon = next(
                (c for c in canonical_section_ids() if c not in seen_canon), None,
            )
            if canon is None:
                continue
        out.append({
            "section_id": sid,
            "label": label[:60],
            "remit": str(item.get("remit") or "").strip()[:300],
            "canonical": canon,
            "order": len(out) + 1,
        })
        seen_ids.add(sid)
        seen_canon.add(canon)
        if len(out) >= MAX_SECTIONS:
            break

    # Top up from the canonical arc so a thin plan still covers the story.
    if len(out) < MIN_SECTIONS:
        for sec in default_sections():
            if len(out) >= MIN_SECTIONS:
                break
            if sec["canonical"] in seen_canon or sec["section_id"] in seen_ids:
                continue
            out.append({**sec, "order": len(out) + 1})
            seen_ids.add(sec["section_id"])
            seen_canon.add(sec["canonical"])

    if not out:
        return default_sections()
    for i, s in enumerate(out, start=1):
        s["order"] = i
    return out


# change for b2c questionarie — A10: segment names must read as a research
# classification, not a persona nickname. Every name has to end in an ordinary
# noun for the people themselves.
_SEGMENT_HEAD_NOUNS = {
    "consumers", "consumer", "buyers", "buyer", "shoppers", "shopper",
    "users", "user", "owners", "owner", "households", "household",
    "subscribers", "subscriber", "members", "member", "viewers", "viewer",
    "customers", "customer", "respondents", "respondent", "drinkers",
    "drinker", "runners", "runner", "wearers", "wearer", "families",
    "family", "purchasers", "purchaser", "adopters", "adopter",
    "beginners", "beginner", "parents", "parent", "athletes", "athlete",
    "adventurers", "adventurer", "cooks", "cook", "gamers", "gamer",
    "drivers", "driver", "riders", "rider", "travellers", "traveller",
    "travelers", "traveler", "collectors", "collector",
    "advocates", "advocate",
}

# change for b2c questionarie — A17: an enumerated list will always miss a word,
# and a miss produces a DOUBLED noun ("Culinary Adventurers Consumers"). English
# forms person-nouns predictably, so recognise the shape rather than the word.
# Checked only AFTER the slang map below, so "Rotators" is still rewritten.
_PERSON_NOUN_SUFFIXES = ("ers", "ors", "ists", "ians", "eers", "yers")


def _looks_like_person_noun(word: str) -> bool:
    """True for a plural noun that names people by what they do.

    Runners, Owners, Adventurers, Collectors, Enthusiasts, Technicians — all
    formed the same way. Recognising the pattern stops the head-noun list from
    having to be exhaustive.
    """
    w = (word or "").lower()
    return len(w) > 5 and w.endswith(_PERSON_NOUN_SUFFIXES)

# Coined or slangy nouns a marketing deck would use. Mapped to the plain word.
_SEGMENT_NOUN_FIXES = {
    "rotators": "Subscribers", "trialists": "Users", "samplers": "Users",
    "bingers": "Viewers", "hunters": "Buyers", "seekers": "Buyers",
    "fans": "Customers", "junkies": "Users", "enthusiasts": "Users",
    "loyalists": "Customers", "switchers": "Customers", "optimizers": "Buyers",
    "optimisers": "Buyers", "balancers": "Buyers", "completists": "Viewers",
    "joiners": "Members", "returnees": "Customers", "socialisers": "Users",
    "socializers": "Users", "planners": "Households", "monitors": "Users",
    "trackers": "Users", "choosers": "Buyers", "commuters": "Buyers",
    "sneakerheads": "Buyers", "foodies": "Buyers", "bingers": "Viewers",
    # change for b2c questionarie — A17: these END in a person-noun shape, so
    # the morphological test would wave them through. Slang is checked first.
    "fashionistas": "Buyers", "influencers": "Users", "gurus": "Users",
    "warriors": "Users", "addicts": "Users", "buffs": "Users",
    "aficionados": "Buyers", "devotees": "Customers", "diehards": "Customers",
    # change for b2c questionarie — A17: adjectives nominalised into a plural
    # ("Occasionals", "Casuals"). They read as a noun to the eye but name no
    # one, and they produced "Price-Sensitive Occasionals Users". Mapping them
    # here makes the doubled-noun pass recognise and collapse the phrase.
    "occasionals": "Users", "casuals": "Users", "regulars": "Customers",
    "heavies": "Users", "lights": "Users", "loyals": "Customers",
}

# Sensible default head noun per category keyword, for a name that dangles.
_SEGMENT_DEFAULT_NOUN = (
    ("subscription", "Subscribers"), ("streaming", "Subscribers"),
    ("milk", "Households"), ("grocer", "Households"), ("food", "Households"),
    ("shoe", "Buyers"), ("apparel", "Buyers"), ("watch", "Owners"),
    ("vehicle", "Owners"), ("car", "Owners"), ("app", "Users"),
)


def _default_head_noun(market: str) -> str:
    low = (market or "").lower()
    for key, noun in _SEGMENT_DEFAULT_NOUN:
        if key in low:
            return noun
    return "Consumers"


def formalize_segment_name(name: str, market: str = "") -> str:
    """Force a segment name into ``[qualifier] + [plain noun for the people]``.

    # change for b2c questionarie — A10

    Swaps a coined noun for the ordinary one ("Price-Driven Rotators" ->
    "Price-Driven Subscribers") and appends a head noun when the name dangles
    ("Category-Committed" -> "Category-Committed Consumers"). A name that
    already follows the construction is returned untouched.
    """
    raw = re.sub(r"\s+", " ", str(name or "").strip())
    if not raw:
        return raw
    words = raw.split(" ")

    # Undo a doubled head noun ("Household Co-Viewers Subscribers"). The name
    # already ends in a noun for the people, so the appended one is redundant.
    def _person_noun(w: str) -> bool:
        w = re.sub(r"[^a-z]", "", w.lower().rsplit("-", 1)[-1])
        return (
            w in _SEGMENT_HEAD_NOUNS
            or w in _SEGMENT_NOUN_FIXES
            # change for b2c questionarie — A17: catches "Adventurers" and any
            # other person-noun the list does not happen to enumerate, which is
            # what produced "Culinary Adventurers Consumers".
            or _looks_like_person_noun(w)
        )

    while len(words) > 2 and _person_noun(words[-1]) and _person_noun(words[-2]):
        words = words[:-1]
    raw = " ".join(words)
    # A hyphenated compound is headed by its last element — "Co-Viewers" is
    # already a noun naming the people, so it must not collect another one.
    last = re.sub(r"[^a-z]", "", words[-1].lower().rsplit("-", 1)[-1])

    # change for b2c questionarie — A17: slang is resolved BEFORE the shape
    # test. "Rotators" and "Fashionistas" both look like person-nouns, but they
    # are marketing register and must still be rewritten.
    fix = _SEGMENT_NOUN_FIXES.get(last)
    if fix:
        return " ".join([*words[:-1], fix]).strip()
    if last in _SEGMENT_HEAD_NOUNS or _looks_like_person_noun(last):
        return raw
    # No noun naming the people — the phrase dangles, so supply one.
    return f"{raw} {_default_head_noun(market)}"


def normalize_blueprint(
    raw: Any,
    *,
    segment: str = "",
    region: str = "",
    source: str = "planner",
    sections_override: Any = None,
) -> dict:
    """Sanitize planner output into a usable blueprint.

    Sections may be segment-specific (see ``normalize_sections``); the storyline
    is keyed by whatever section ids survive that pass. Within each section every
    beat gets a unique non-empty ``beat_id`` and ``title``, ``canonical`` is
    either a valid canonical beat id for that section's canonical or ``None``,
    beats are capped at ``MAX_BEATS_PER_TAB``, thin sections are topped up from
    the canonical arc, and ``order`` is a dense 1..N.

    ``sections_override`` pins the section list (used so every region of one
    market shares the same tabs).
    """
    sections = normalize_sections(
        sections_override
        if sections_override is not None
        else ((raw or {}).get("sections") if isinstance(raw, dict) else None)
    )
    by_id = {s["section_id"]: s for s in sections}

    storyline_raw: dict = {}
    if isinstance(raw, dict):
        # Accept {"storyline": {...}} or a bare {section: [...]} mapping.
        inner = raw.get("storyline") if "storyline" in raw else raw
        if isinstance(inner, list):
            # Pydantic shape: [{"tab": ..., "beats": [...]}, ...]
            for entry in inner:
                if not isinstance(entry, dict):
                    continue
                key = _slug(entry.get("tab") or entry.get("section_id") or "")
                if key in by_id:
                    storyline_raw[key] = entry.get("beats") or []
        elif isinstance(inner, dict):
            for key, beats in inner.items():
                key = _slug(key)
                if key in by_id:
                    storyline_raw[key] = beats or []

    out: dict[str, list[dict]] = {}
    for section in sections:
        tab = section["section_id"]
        # Beat vocabulary comes from the section's CANONICAL, so a renamed
        # section still has a sensible fallback arc and anchor set.
        valid_canon = set(canonical_ids(section["canonical"]))
        beats: list[dict] = []
        seen: set[str] = set()
        for item in storyline_raw.get(tab) or []:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            bid = _slug(item.get("beat_id") or title)
            if not bid or not title or bid in seen:
                continue
            canon = item.get("canonical")
            canon = _slug(canon) if canon else ""
            beats.append({
                "beat_id": bid,
                "title": title[:80],
                "intent": str(item.get("intent") or "").strip()[:400],
                "canonical": canon if canon in valid_canon else None,
                "order": len(beats) + 1,
            })
            seen.add(bid)
            if len(beats) >= MAX_BEATS_PER_TAB:
                break

        # Top up thin sections from the canonical arc, skipping covered beats.
        if len(beats) < MIN_BEATS_PER_TAB:
            covered = {b["canonical"] for b in beats if b["canonical"]}
            for cb in canonical_beats(section["canonical"]):
                if len(beats) >= MIN_BEATS_PER_TAB:
                    break
                if cb["beat_id"] in seen or cb["beat_id"] in covered:
                    continue
                beats.append({**cb, "order": len(beats) + 1})
                seen.add(cb["beat_id"])

        for i, b in enumerate(beats, start=1):
            b["order"] = i
        out[tab] = beats

    # change for b2c questionarie — A10: repair segment names the model wrote in
    # a marketing register. The planner prompt states the construction, but the
    # deliverable cannot carry "Premium Bingers" or a dangling "Value-Led", so
    # the rule is enforced here as well as asked for there.
    raw_segments = (raw or {}).get("segments") if isinstance(raw, dict) else None
    seen_seg: set[str] = set()
    segments: list[dict] = []
    for item in (raw_segments or []):
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        sid = _slug(item.get("segment_id") or name)
        if not sid or not name or sid in seen_seg:
            continue
        segments.append({
            "segment_id": sid,
            "name": formalize_segment_name(name, segment)[:48],
            "description": str(item.get("description") or "").strip()[:400],
            "hypothesis": str(item.get("hypothesis") or "").strip()[:300],
        })
        seen_seg.add(sid)
        if len(segments) >= MAX_SEGMENTS:
            break

    # change for b2c questionarie — MIN_SEGMENTS was declared but never
    # enforced, unlike sections which top up from the canonical arc. A planner
    # that returned one segment produced a one-segment study, and the site
    # expects a consistent four. Top up with generic behavioural groups so the
    # count is always right; the planner's own names are kept first.
    _FALLBACK = (
        ("frequent-users", "Frequent Users",
         "Buy or use the category more often than most."),
        ("occasional-users", "Occasional Users",
         "Buy or use the category infrequently."),
        ("price-sensitive-buyers", "Price-Sensitive Buyers",
         "Choose primarily on price and value."),
        ("quality-led-buyers", "Quality-Led Buyers",
         "Choose primarily on quality and performance."),
    )
    for sid, name, desc in _FALLBACK:
        if len(segments) >= MIN_SEGMENTS:
            break
        if sid in seen_seg:
            continue
        segments.append({
            "segment_id": sid, "name": name,
            "description": desc, "hypothesis": "",
        })
        seen_seg.add(sid)

    return {
        "segment": segment,
        "region": region,
        "study_purpose": str(
            (raw or {}).get("study_purpose") or ""
        ).strip()[:400] if isinstance(raw, dict) else "",
        # change for b2c questionarie — A11: the stated axis the segments cut on.
        "segmentation_lens": str(
            (raw or {}).get("segmentation_lens") or ""
        ).strip()[:300] if isinstance(raw, dict) else "",
        # change for b2c questionarie — A12: the category's non-negotiable topics.
        "must_cover_topics": [
            str(t).strip()[:60]
            for t in ((raw or {}).get("must_cover_topics") or [])[:6]
            if str(t or "").strip()
        ] if isinstance(raw, dict) else [],
        "segments": segments,
        "sections": sections,
        "storyline": out,
        "narrative_summary": _clip(
            (raw or {}).get("narrative_summary") if isinstance(raw, dict) else "", 600,
        ),
        "adaptation_notes": [
            str(n)[:240]
            for n in ((raw or {}).get("adaptation_notes") or [])[:10]
        ] if isinstance(raw, dict) else [],
        "source": source,
    }


def beats_for_tab(blueprint: dict | None, tab: str) -> list[dict]:
    """Ordered beats for one section, falling back to its canonical arc.

    ``tab`` is a section id, which may be segment-specific. When the blueprint
    carries no beats for it, the canonical section it maps onto supplies them.
    """
    storyline = (blueprint or {}).get("storyline") or {}
    beats = storyline.get(tab) or []
    if not beats:
        return canonical_beats(section_canonical(blueprint, tab))
    return sorted(beats, key=lambda b: int(b.get("order") or 999))


# Sections are the unit now; keep the old name as the obvious alias.
beats_for_section = beats_for_tab


def beat_orders(blueprint: dict | None) -> dict[str, dict[str, int]]:
    """{section_id: {beat_id: zero-based order}} for fast lookup."""
    return {
        sid: {
            b["beat_id"]: i
            for i, b in enumerate(beats_for_tab(blueprint, sid))
        }
        for sid in section_ids(blueprint)
    }


def blueprint_digest(blueprint: dict | None) -> str:
    """Compact signature of a blueprint — used in cache keys."""
    parts = []
    for sid in section_ids(blueprint):
        ids = "+".join(b.get("beat_id", "") for b in beats_for_tab(blueprint, sid))
        parts.append(f"{sid}:{ids}")
    return "|".join(parts)[:400]


def plan_for(blueprint: dict | None, per_section: int) -> list[tuple[str, int]]:
    """[(section_id, question target)] — the runtime replacement for CATEGORY_PLAN."""
    return [(sid, per_section) for sid in section_ids(blueprint)]


# ---------------------------------------------------------------------------
# Question allocation across beats
# ---------------------------------------------------------------------------
def allocate_questions(beats: list[dict], count: int) -> list[tuple[dict, int]]:
    """Spread ``count`` questions across ordered ``beats``.

    Earlier beats win the remainder, so a short tab still opens properly rather
    than starting halfway through the story.
    """
    if not beats or count <= 0:
        return []
    n = len(beats)
    if count <= n:
        return [(b, 1) for b in beats[:count]]
    base, extra = divmod(count, n)
    return [(b, base + (1 if i < extra else 0)) for i, b in enumerate(beats)]


def chunk_allocations(
    allocations: list[tuple[dict, int]], max_per_chunk: int
) -> list[list[tuple[dict, int]]]:
    """Group consecutive beat allocations into generation chunks.

    Chunks never straddle the storyline out of order, so each LLM call sees a
    contiguous slice of the narrative.
    """
    chunks: list[list[tuple[dict, int]]] = []
    current: list[tuple[dict, int]] = []
    running = 0
    for beat, n in allocations:
        if current and running + n > max_per_chunk:
            chunks.append(current)
            current, running = [], 0
        current.append((beat, n))
        running += n
    if current:
        chunks.append(current)
    return chunks


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------
def order_questions(questions: list[dict], beats: list[dict]) -> list[dict]:
    """Sort one tab's questions into narrative order and stamp positions.

    Sort key is ``(beat order, model's within-beat hint, generation order)``.
    Stamps ``narrative_order`` and ``funnel_position`` as a dense 1..N, and
    backfills ``beat_title`` / ``beat_canonical`` from the blueprint so
    downstream consumers never have to re-resolve the beat.
    """
    order_map = {b.get("beat_id"): i for i, b in enumerate(beats)}
    by_id = {b.get("beat_id"): b for b in beats}

    def _hint(q: dict) -> int:
        try:
            return max(1, int(q.get("funnel_position") or 1))
        except (TypeError, ValueError):
            return 1

    decorated = [
        (
            order_map.get(q.get("beat_id"), _UNKNOWN_BEAT_ORDER),
            _hint(q),
            i,
            q,
        )
        for i, q in enumerate(questions)
    ]
    decorated.sort(key=lambda t: (t[0], t[1], t[2]))

    ordered = []
    for n, (_, _, _, q) in enumerate(decorated, start=1):
        beat = by_id.get(q.get("beat_id"))
        q["narrative_order"] = n
        q["funnel_position"] = n
        if beat:
            q["beat_title"] = beat.get("title") or ""
            q["beat_canonical"] = beat.get("canonical")
        else:
            q.setdefault("beat_title", "")
            q.setdefault("beat_canonical", None)
        ordered.append(q)
    return ordered


def _token_closest(text: str, beats: list[dict]) -> str | None:
    """Best beat match by token overlap against beat id + title."""
    want = {t for t in re.findall(r"[a-z]+", (text or "").lower()) if len(t) > 2}
    if not want:
        return None
    best, best_hits = None, 0
    for b in beats:
        cand = f"{b.get('beat_id', '')} {b.get('title', '')}".lower()
        hits = sum(1 for t in want if t in cand)
        if hits > best_hits:
            best, best_hits = b.get("beat_id"), hits
    return best


def remap_questions_to_beats(
    questions: list[dict], beats: list[dict], tab: str
) -> list[dict]:
    """Rewrite ``beat_id`` onto a different blueprint's beats, in place.

    The shared CORE instrument is written against the canonical arc so it stays
    identical across regions, but each region plans its own storyline. This maps
    a core question onto the equivalent beat of the region actually publishing
    it, using the ``canonical`` anchor first, then title tokens, then nearest
    canonical rank. A question is always placed — never left to sink to the end.
    """
    if not beats:
        return questions
    valid = {b["beat_id"] for b in beats}
    by_canon: dict[str, str] = {}
    for b in beats:
        c = b.get("canonical")
        if c and c not in by_canon:
            by_canon[c] = b["beat_id"]
    rank = canonical_order(tab)

    for q in questions:
        bid = q.get("beat_id") or ""
        if bid in valid:
            continue
        # Core questions carry canonical ids directly; generated ones carry the
        # canonical tag stamped by order_questions.
        canon = q.get("beat_canonical") or (bid if bid in rank else None)

        target = by_canon.get(canon) if canon else None
        if not target:
            target = _token_closest(f"{bid} {q.get('beat_title') or ''}", beats)
        if not target and canon in rank:
            want = rank[canon]
            target = min(
                beats,
                key=lambda b: abs(rank.get(b.get("canonical"), 999) - want),
            ).get("beat_id")
        q["beat_id"] = target or beats[0]["beat_id"]
    return questions


def coverage(questions: Iterable[dict], beats: list[dict]) -> dict:
    """How much of the storyline a tab's questions actually cover."""
    planned = [b.get("beat_id") for b in beats]
    used = {q.get("beat_id") for q in questions if q.get("beat_id")}
    missing = [b for b in planned if b not in used]
    return {
        "planned": len(planned),
        "covered": len([b for b in planned if b in used]),
        "missing": missing,
        "untagged": sum(1 for q in questions if not q.get("beat_id")),
    }


def storyline_for_metadata(blueprint: dict | None) -> dict:
    """Compact storyline keyed by FRONTEND section id, for published metadata.

    Phase A6 reads this back off the regional files to order and align the
    global questionnaire, so it must survive publication.
    """
    out: dict[str, list[dict]] = {}
    for section in sections_for(blueprint):
        sid = frontend_section_id(section["section_id"])
        out[sid] = [
            {
                "beatId": b.get("beat_id"),
                "title": b.get("title"),
                "canonical": b.get("canonical"),
                "order": b.get("order"),
            }
            for b in beats_for_tab(blueprint, sid)
        ]
    return out


def frontend_section_id(section_id: str) -> str:
    """Id this section is published under.

    The canonical four keep their legacy frontend ids (``profile`` …) so older
    payloads, Phase A6 and the web component are unaffected. A market-specific
    section is published under its own id.
    """
    return TAB_SECTION_IDS.get(section_id, section_id)


def sections_for_metadata(blueprint: dict | None) -> list[dict]:
    """Published section list — what the UI tabs are and what they map onto."""
    return [
        {
            "id": frontend_section_id(s["section_id"]),
            "label": s.get("label"),
            "canonical": s.get("canonical"),
            "order": s.get("order"),
            "remit": s.get("remit", ""),
        }
        for s in sections_for(blueprint)
    ]


def canonical_rank(section_id: str, canonical_beat: str | None) -> int:
    """Position of a canonical beat within its section's canonical arc.

    Phase A6 uses this to order questions drawn from independently planned
    regional storylines. ``section_id`` may be a generated section id or a
    canonical one. Unmapped beats sort last.
    """
    if not canonical_beat:
        return _UNKNOWN_BEAT_ORDER
    canon = section_id if section_id in canonical_section_rank() else None
    if canon is None:
        # Frontend ids used by older payloads.
        for tab, sid in TAB_SECTION_IDS.items():
            if sid == section_id:
                canon = tab
                break
    if canon is None:
        # A generated section id — search every canonical arc for the beat.
        for tab in TAB_ORDER:
            rank = canonical_order(tab)
            if canonical_beat in rank:
                return rank[canonical_beat]
        return _UNKNOWN_BEAT_ORDER
    return canonical_order(canon).get(canonical_beat, _UNKNOWN_BEAT_ORDER)


def segments_for(blueprint: dict | None) -> list[dict]:
    """Behavioural segments this study must separate."""
    return (blueprint or {}).get("segments") or []


def study_purpose(blueprint: dict | None) -> str:
    return ((blueprint or {}).get("study_purpose") or "").strip()


def segmentation_lens(blueprint: dict | None) -> str:
    """The axis these segments are cut on, and why. # change for b2c questionarie"""
    return ((blueprint or {}).get("segmentation_lens") or "").strip()


def must_cover_topics(blueprint: dict | None) -> list:
    """Topics this category cannot omit. # change for b2c questionarie — A12"""
    return list((blueprint or {}).get("must_cover_topics") or [])
