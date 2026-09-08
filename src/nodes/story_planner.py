"""Node 4b — story_planner.

# change for b2c questionarie — Phase A8

Plans the survey's NARRATIVE ARC before any question is written.

The four sections are fixed (profile → behavior → preferences → satisfaction).
This node decides the ordered *beats* inside each one, adapted to the actual
category and to this region's shopping reality — a consumable, a durable, a
subscription and a service each tell a different story, and the questionnaire
should follow the right one.

The storyline is planned PER REGION, so local buying reality can reshape the
arc. To keep five independently-planned regions comparable in the Phase A6
global blend, every beat must also declare the ``canonical`` beat it maps onto
from ``narrative.CANONICAL_BEATS``. That tag is the cross-region anchor.

This node writes no questions and no percentages — only the shape of the story.
"""

from __future__ import annotations

import json

from src import cache_store, cmi_match, narrative
from src.date_utils import current_date_context
from src.llm import get_structured_llm
from src.models import SurveyBlueprint
from src.state import SurveyState

# change for b2c questionarie — v3: section labels must read as a research
# firm's report headings, not consumer-magazine copy ("Content Wishlist").
# v2 entries carry the informal names and must not be reused.
# v4: the blueprint now carries study_purpose + behavioural segments.
# v5: institutional register for BOTH section headings and segment
# names — v4 produced "Streaming Habits" / "Premium Bingers".
# v6: segment names must read as a research classification —
# [behavioural qualifier] + [plain noun naming the people].
# v7: the planner must name its segmentation_lens before cutting segments,
# and scope the study when the business sells to consumers AND firms.
# v8: the planner now declares must_cover_topics — the issues its category
# cannot omit — which the validator then enforces against the instrument.
# v9: beat titles must be written in the category's own words. v8 produced
# generic labels ("Category relationship", "Usage occasions") that print under
# every question and are the most visible sign of a reused template.
# v10: cache key is now geo_label-aware (was parent-region-only, a collision
# risk between countries in the same region) and no longer includes volatile
# per-run evidence text (was causing a fresh, uncontrolled 4th-segment
# concept on every separate regeneration of the same category+geography).
# v11: "Advocates" recognised as a valid segment head-noun, fixing a doubled
# noun ("Clean-Beauty Advocates Owners") that shipped from a stale cache.
_SCHEMA_V = "storyline_v11"

_TAB_REMIT = {
    "consumer_profile": (
        "category relationship and usage context — who uses it, on what "
        "occasions, how often, driven by what need. Never age, income or "
        "where they live."
    ),
    "buying_behavior": (
        "the purchase journey — what triggers it, how they discover options, "
        "where they buy, how often, what they spend on the category, what "
        "decides the choice. Never brand or company names."
    ),
    "preferences_expectations": (
        "what they want from the product — must-haves, quality cues, format, "
        "claims, trade-offs, and value expectations. Never named competitors."
    ),
    "satisfaction_future_intent": (
        "outcomes — satisfaction, what drives it, friction, switching "
        "triggers, recommend intent, and where their use is heading."
    ),
}


PROMPT = """\
You are a senior questionnaire designer at a top consumer-insights firm,
planning the NARRATIVE ARC of a B2C survey on "{segment}" for the region
"{region}". Today is {today}.

You are NOT writing questions yet. You are designing the STUDY: what it is for,
who it must tell apart, and the order it asks in.

STEP 0 — DEFINE THE STUDY AND ITS SEGMENTS.
This is a SEGMENTATION study. Its job is to place every respondent into one of
a small number of behavioural groups, so the client knows who they are selling
to and how those groups differ.

Write `study_purpose` as one sentence naming what the study identifies and the
evidence it uses — e.g. "Identify which of 4 behavioural segments each
respondent belongs to, based on viewing habits, spend behaviour and content
motivations."

CHOOSE THE LENS FIRST. Before naming any segment, decide which axis actually
separates buyers in THIS category, and write it into `segmentation_lens` with
one sentence of justification. Pick one, or a stated combination of two:

- USAGE / BEHAVIOURAL — how often, how much, how deeply engaged. Best where
  frequency genuinely changes the economics (streaming, groceries, fuel).
- JOBS-TO-BE-DONE — what the purchase is hired to do. Best where the same
  product serves unrelated needs (a smartwatch as a fitness tool vs a watch).
- PRICE SENSITIVITY / VALUE ORIENTATION — willingness to trade price against
  quality. Best in commoditised or heavily promoted categories.
- PSYCHOGRAPHIC — attitude, identity, lifestyle. Best where the purchase is
  expressive (fashion, organic food, EVs) — weakest on its own, because
  attitudes predict behaviour poorly. Pair it with a behavioural axis.

Do not default to usage frequency because it is the easiest to ask about.
Choose the axis on which this category's buyers genuinely differ.
{dual_motion_block}
Then define EXACTLY 4 `segments`. Each needs an id, a `name` that is a research
classification of the defining behaviour (see SEGMENT NAMING below), a
`description` of how that group behaves in THIS category, and a `hypothesis`
the data should confirm or kill. Every segment must be separated by the lens you
just chose — if two segments differ on some other axis, you picked the wrong lens.

NEVER PUT AN ABSOLUTE CURRENCY AMOUNT IN A `description` (CHECK 18/26f). This
blueprint is cached and reused across every country in the region — the same
description text will be read by an India file, a China file, and any other
country in this region. A hardcoded amount ("often under $100") is only ever
correct for one of them and wrong, in the wrong currency, for the rest.
  BAD:  "look for durable, comfortable shoes that last, often under $100"
  GOOD: "look for durable, comfortable shoes at an accessible price point"
Describe price sensitivity, income band, or spend level in RELATIVE terms
(low-cost, mid-range, premium, budget-conscious) — never a specific number or
currency symbol. The same rule applies to every other numeric/monetary detail
in a description: keep it qualitative so it stays true in every country this
blueprint is reused for.

NAME WHAT THIS CATEGORY CANNOT OMIT. Write 3-6 `must_cover_topics`: the issues
a specialist in this category would immediately notice were missing from the
questionnaire. These are the category's real commercial questions, not generic
research themes.
  Streaming: account/password sharing · ad-supported tier · monthly spend ·
             devices used · where they go after cancelling
  Running shoes: replacement cycle · injury or comfort problems · fit ·
             price paid · where they run
  Grocery staple: pack size · where they shop · price per unit · who else
             in the household eats it
Each topic is 1-4 plain words that will literally appear in a question about it,
so its coverage can be checked. A survey missing its category's defining issue
reads as generic no matter how well the questions are written.

BEAT TITLES MUST NAME THIS CATEGORY. A beat title is printed under every
question in the deliverable, so a generic one is the most visible sign that a
template was reused.
  BAD (generic, could be any category): "Category relationship" ·
      "Usage occasions" · "Usage frequency and intensity" ·
      "What starts a purchase" · "Must-have attributes"
  GOOD (organic milk): "How much milk the household drinks" ·
      "Meals and occasions milk is used for" · "What triggers a top-up shop" ·
      "Fat content, packaging and provenance"
  GOOD (streaming): "Which services they subscribe to" ·
      "How viewing fits the week" · "What triggers a new subscription"
Write every beat title in the category's own words.

RULES FOR SEGMENTS (never violate):
- BEHAVIOURAL AND ATTITUDINAL ONLY. Segments are defined by what people DO and
  what drives them — frequency, spend pattern, switching, motivation, who they
  buy for. NEVER by age, income, gender or location.
- MUTUALLY EXCLUSIVE and collectively covering the market. A respondent should
  fall clearly into one.
- REACHABLE BY CLOSED QUESTIONS. Every segment must be identifiable from answers
  to the questions this survey will ask. If you cannot picture the question that
  separates it, it is the wrong segment.
- Include at least one low-engagement / low-commitment group and at least one
  heavy or high-value group — real markets have both.

SEGMENT NAMING — a segment name is a research classification, written the way a
professional services firm writes one. It is NOT a persona nickname.

EVERY segment name uses exactly this construction:

    [behavioural qualifier] + [plain noun naming the people]

The name MUST end in an ordinary English noun for the people themselves —
Consumers, Buyers, Shoppers, Users, Owners, Households, Subscribers, Members,
Viewers, Customers, Respondents. Pick the one that fits the category.

- The qualifier states the behaviour that defines the group: High-Frequency,
  Price-Sensitive, Value-Driven, Quality-Led, Occasional, Infrequent, Lapsed,
  Multi-Service, Single-Service, Convenience-Led, Performance-Focused,
  Routine, Committed.
- Two to four words, Title Case, no slang, no invented nouns, no wordplay.
- The name must be a COMPLETE noun phrase. An adjective on its own is not a
  segment name.

BAD — invented or slangy nouns (these read as marketing personas):
  "Premium Bingers" · "Content Hunters" · "Deal Hunters" · "Super Fans" ·
  "Price-Driven Rotators" · "Occasional Trialists" · "Casual Samplers"
BAD — no noun naming the people, so the phrase dangles:
  "Category-Committed" · "Value-Led" · "Eco-Conscious Quality" ·
  "High-Engagement Performance" · "Comfort-Led Recreational"
GOOD:
  "High-Frequency Subscribers" · "Price-Sensitive Buyers" ·
  "Occasional Users" · "Quality-Led Shoppers" · "Multi-Service Households" ·
  "Lapsed Customers" · "Performance-Focused Owners"

Worked example for a streaming category: "High-Frequency Subscribers" (heavy
daily viewers, multiple services, low price sensitivity, driven by exclusives) /
"Price-Sensitive Subscribers" (price-led, cancel and re-subscribe, compare
plans, respond to promotions) / "Multi-Person Households" (a shared family
resource, driven by child-appropriate content and simultaneous streams) /
"Occasional Users" (low frequency, subscribe for one title then lapse, least
loyal).

STEP 1 — THE SECTIONS ARE FIXED. DO NOT INVENT THEM.
This study uses the standard CMI survey categories for its domain. They are
given below and are NOT yours to rename, reorder, add to or drop:

{framework_block}

Echo these four sections back exactly as given — same `section_id`, same
`label`, same `canonical`, same order. Your creative work is STEP 2, where you
choose the beats INSIDE each fixed section. That is where this category's
particular story gets told.

The four sections are the underlying arc: who uses it → how they buy it →
what they need from it → how they feel about it and what they do next.

STEP 2 — For EACH fixed section, produce an ordered list of 4-6 BEATS. A beat
is one narrative slot that will hold one or more questions. Beats are asked in
the order you give them. Use the section_id given above as the `tab` value.

BEAT BUDGET — each section carries 6-9 questions (about 32 across the study).
Give 4 beats when the section has one clear story, up to 6 when it genuinely
needs them. A section must never have more beats than it has questions.

BEATS ARE A SEQUENCE, NOT A LIST. Each beat must follow from the one before it
so the section reads as one continuous line of thought. Order them the way the
story actually unfolds:
  - establish WHAT they use / own / buy, before asking about the detail of it
  - a beat that depends on the previous beat's answer comes straight after it
  - the widest, easiest beat opens the section; the most specific or most
    sensitive one closes it
Two beats that would produce questions on the same topic must be adjacent —
never separated by a beat on a different topic. If a topic would be split
across non-adjacent beats, merge them into one beat.

RULES FOR A GOOD ARC (never violate):
1. STRICTLY PROGRESSIVE within each section: broad → specific, easy → effortful,
   neutral → sensitive. The first beat of a section must be the widest and
   safest thing to ask there. Money, satisfaction judgments and switching
   belong late in their section.
2. ADAPT TO THE CATEGORY. The canonical arc below is a starting point, not a
   template to copy. Rename, reorder, drop beats that do not apply, and add
   category-specific beats that a generic survey would miss. Examples of good
   adaptation:
   - a consumable (milk, coffee): consumption occasion and restock cycle matter;
     a "replacement/upgrade" beat does not.
   - a durable (smartwatch, air fryer): research effort, ownership tenure,
     upgrade trigger and compatibility matter; "restock frequency" does not.
   - a subscription or service: onboarding, renewal moment, cancellation
     pressure and perceived ongoing value matter.
   - an apparel/footwear category: fit and sizing confidence is a real beat.
3. ADAPT TO THE REGION. Use the regional note and market profile below. If
   local shopping reality changes the story (dominant channel, informal retail,
   mobile-first discovery, seasonal availability), reflect it in beat choice
   and beat order for THIS region.
4. EVERY BEAT MUST DECLARE A `canonical` VALUE — the closest canonical beat id
   for that section from the list below, or null ONLY if the beat is genuinely
   category-specific with no canonical equivalent. This is how the same
   narrative moment is matched across regions, so map honestly. Do NOT map two
   different beats in the same section to the same canonical id.
5. NO OVERLAP. Two beats in the same section must never cover the same ground —
   each beat must be a distinct question-able moment in the story. If two beats
   would produce similar questions, merge them into one.
6. Beats must be answerable with CLOSED questions about {segment}, and must not
   require asking age, income, geography of residence, or brand/company names.

CANONICAL ARC (the shared vocabulary — use these ids for `canonical`):
{canonical_json}

MARKET KNOWLEDGE PROFILE (JSON):
{mkp_json}

REGIONAL NOTE for {region}: {regional_note}

TARGET AUDIENCE (JSON):
{audience_json}

PERSONA SNAPSHOT (the archetypes this survey must be able to tell apart):
{persona_json}

Return `study_purpose`, `segments` (EXACTLY 4 behavioural segments), `sections`
(your category-specific section names, in narrative order, each with its
canonical and remit) and `storyline` (the beats for each of those sections,
keyed by YOUR section ids), plus a short narrative_summary and
adaptation_notes explaining what you changed and why — including how the
beats you chose will separate the segments you defined.

IN adaptation_notes AND narrative_summary, if you name a persona/segment,
name ONLY one of the 4 segment names you just defined in `segments` above —
never a name from the PERSONA SNAPSHOT above (those are a different,
internal archetype list you do not control and must not reference here).
Confirmed live: a note referenced "an Eco-Conscious Urbanite", a persona
snapshot archetype name that was never one of the 4 segments actually
defined in that same response — a mismatch a reviewer immediately notices
and reads as sloppy, inconsistent work. Before writing either field, check
every persona/segment name you are about to write against your own
`segments` list; if it is not an exact match, do not write it.

Do NOT describe how a specific question will be WORDED or FORMATTED
(e.g. "kept price in relative low/mid/premium terms", "phrased as a
5-point scale") — you plan the beats and narrative order, a separate later
step writes the actual question text and decides its concrete format, and
your note has no way to know whether that later step will follow your
suggestion or override it with something better. Confirmed live: a note
claimed price was "kept in relative terms (low/mid/premium)" while the
question that was actually written used concrete currency-denominated
bands instead — a stale claim describing a decision that was reversed
downstream. Describe WHAT the beat covers and WHY it matters to the
segments, not HOW a question about it will be phrased.
"""


def _canonical_json() -> str:
    payload = {
        tab: [
            {"canonical_id": bid, "title": title, "covers": intent}
            for bid, title, intent in narrative.CANONICAL_BEATS[tab]
        ]
        for tab in narrative.TAB_ORDER
    }
    return json.dumps(payload, ensure_ascii=False)


def _tab_remits() -> str:
    """The canonical research jobs a generated section may claim."""
    return "\n".join(
        f"  {i}. canonical=\"{sid}\" — {_TAB_REMIT.get(sid, remit)}"
        for i, (sid, _label, remit) in enumerate(
            narrative.CANONICAL_SECTIONS, start=1,
        )
    )


def _mkp_digest(mkp: dict) -> dict:
    return {
        k: mkp.get(k)
        for k in (
            "market_maturity", "avg_income_band", "price_sensitivity",
            "growth_outlook", "technology_adoption", "environmental_awareness",
            "brand_loyalty", "consumer_sentiment", "evidence_confidence",
        )
    } | {
        "top_pain_points": (mkp.get("top_pain_points") or [])[:8],
        "top_purchase_drivers": (mkp.get("top_purchase_drivers") or [])[:8],
        "consumer_language": (mkp.get("consumer_language") or [])[:8],
        "feature_requests": (mkp.get("feature_requests") or [])[:6],
    }


def _dual_motion_block(state: SurveyState) -> str:
    """Extra planner instruction when the business sells to consumers AND firms.

    # change for b2c questionarie — A11

    Empty for an ordinary consumer market, so the prompt is unchanged for the
    common case and the storyline cache for existing markets stays valid.
    """
    if not state.get("sells_to_both"):
        return ""
    why = (state.get("dual_motion_reason") or "").strip()
    return (
        "\nTHIS BUSINESS SELLS TO BOTH CONSUMERS AND BUSINESSES.\n"
        + (f"{why}\n" if why else "")
        + "This survey covers the CONSUMER motion — people buying on their own\n"
        "account. Say so explicitly in `narrative_summary`, and name which motion\n"
        "each segment belongs to in its description.\n"
        "Where a segment sits on the business side (someone buying for a team),\n"
        "its distinguishing signals must include the firmographic and buying-role\n"
        "facts that separate it — team size, who else signs off, whether it is\n"
        "expensed — because those buyers do not behave like consumers.\n"
    )


# change for b2c questionarie — render the matched CMI framework for the prompt.
def _framework_block(match: dict) -> str:
    """The fixed sections, plus each one's CMI seed topics where we have them."""
    lines = [
        f"CMI domain: {match.get('domain') or 'Consumer Goods'}"
        + (f"  ·  sub-sector: {match['sub_sector']}" if match.get("sub_sector") else "")
    ]
    topics = cmi_match.must_cover_topics(match)
    for i, sec in enumerate(cmi_match.blueprint_sections(match), start=1):
        cid = sec["section_id"]
        lines.append(
            f'\n{i}. section_id: "{cid}"\n'
            f'   label:      "{sec["label"]}"\n'
            f'   canonical:  "{cid}"\n'
            f'   covers:     {sec["remit"]}'
        )
        seeds = topics.get(cid) or []
        if seeds:
            lines.append(
                "   CMI expects this section to cover: " + "; ".join(seeds[:5])
            )
    return "\n".join(lines)


def story_planner(state: SurveyState) -> dict:
    """Plan the ordered storyline beats for this segment + region."""
    segment = (
        state.get("normalized_segment")
        or state.get("market_segment")
        or "the segment"
    )
    region = state.get("region") or "Global"
    # change for b2c questionarie -- CHECK (Section 27f): the geography this
    # blueprint (and its 4 named behavioural segments) is written for. Was
    # missing entirely -- the cache key used only the PARENT region, so a
    # China job and an India job in the same region could collide on this
    # key exactly like the question_architect module-cache bug fixed
    # earlier this session.
    geo_label = state.get("country") or state.get("geography_label") or region
    mkp = state.get("market_knowledge_profile") or {}
    catalog = state.get("persona_catalog") or {}
    audience = state.get("audience_profile") or {}
    today = current_date_context()["iso"]

    # change for b2c questionarie -- CHECK (Section 27f): the key used to
    # include mkp.evidence_confidence and top_purchase_drivers, both derived
    # from freshly-fetched evidence that is essentially never identical
    # across two separate CLI runs -- confirmed live: two separate "Running
    # Shoes, China" generations produced two different 4th-segment concepts
    # (a style/fashion segment vs a sustainability segment) with no
    # indication this was a deliberate revision. The 4 named behavioural
    # segments are a client-facing segmentation scheme meant to stay stable
    # across regenerations of the same category+geography, so the key is now
    # geography-aware (fixing the collision above) and evidence-independent
    # (fixing the run-to-run drift) -- segment + geo_label only.
    cache_key = cache_store.make_key(_SCHEMA_V, segment, geo_label)
    cached = cache_store.get_json("storyline", cache_key)
    if cached and cached.get("survey_blueprint"):
        return {"survey_blueprint": cached["survey_blueprint"]}

    personas = [
        {
            "name": p.get("name"),
            "price_sensitivity": p.get("price_sensitivity"),
            "pain_points": (p.get("pain_points") or [])[:3],
        }
        for p in (catalog.get("personas") or [])[:12]
    ]

    fw = state.get("cmi_framework") or cmi_match.match_domain(segment)
    prompt = PROMPT.format(
        segment=segment,
        region=region,
        today=today,
        dual_motion_block=_dual_motion_block(state),
        framework_block=_framework_block(fw),
        tab_remits=_tab_remits(),
        canonical_json=_canonical_json(),
        mkp_json=json.dumps(_mkp_digest(mkp), ensure_ascii=False)[:4000],
        regional_note=(mkp.get("regional_notes") or {}).get(region, "n/a"),
        audience_json=json.dumps(audience, ensure_ascii=False)[:1500],
        persona_json=json.dumps(personas, ensure_ascii=False)[:2000],
    )

    # change for b2c questionarie — sections are pinned at SEGMENT level.
    # Beats may vary per region, but the tabs must not: if North America had 5
    # sections and Europe 4, the same market would show different tabs depending
    # on which region you opened, and A6 could not blend them. The first region
    # to run defines the section set; the rest inherit it.
    sections_key = cache_store.make_key(_SCHEMA_V, "sections", segment)
    pinned = (cache_store.get_json("storyline", sections_key) or {}).get("sections")

    # change for b2c questionarie — the CMI domain framework OUTRANKS both the
    # cache and the planner's own naming. The RAR defines the four survey
    # categories per domain (a food market opens on consumption occasions, an
    # electronics market on device ownership), so those names are a deliverable
    # requirement, not a stylistic choice. The planner keeps full freedom over
    # the beats INSIDE each section, which is where category adaptation belongs.
    cmi_match_result = state.get("cmi_framework") or cmi_match.match_domain(segment)
    framework_sections = cmi_match.blueprint_sections(cmi_match_result)
    if framework_sections:
        pinned = framework_sections

    try:
        result: SurveyBlueprint = get_structured_llm(
            SurveyBlueprint, temperature=0.25, max_tokens=8000
        ).invoke(prompt)
        raw = result.model_dump()
        blueprint = narrative.normalize_blueprint(
            raw, segment=segment, region=region, source="planner",
            sections_override=pinned,
        )
    except Exception as exc:  # noqa: BLE001 — a failed plan must not sink the run
        blueprint = narrative.default_blueprint(segment=segment, region=region)
        if pinned:
            blueprint = narrative.normalize_blueprint(
                {"sections": pinned}, segment=segment, region=region,
                source="default", sections_override=pinned,
            )
        blueprint["adaptation_notes"] = [
            f"Planner LLM failed ({type(exc).__name__}) — canonical arc used.",
        ]

    if not pinned:
        cache_store.set_json(
            "storyline", sections_key, {"sections": blueprint.get("sections")},
        )

    cache_store.set_json("storyline", cache_key, {"survey_blueprint": blueprint})
    return {"survey_blueprint": blueprint}


if __name__ == "__main__":
    demo = {
        "normalized_segment": "Smartwatches",
        "region": "Asia Pacific",
        "market_knowledge_profile": {
            "market_maturity": "growing",
            "price_sensitivity": "high",
            "technology_adoption": "early",
            "evidence_confidence": "medium",
            "top_pain_points": ["battery dies too fast", "strap irritation"],
            "top_purchase_drivers": ["fitness tracking", "price"],
            "regional_notes": {"Asia Pacific": "Mobile-first discovery; marketplace apps dominate."},
        },
        "persona_catalog": {"personas": [{"name": "Value Seeker", "price_sensitivity": "high"}]},
        "audience_profile": {"qualifiers": ["owns or plans to buy a smartwatch"]},
    }
    out = story_planner(demo)
    bp = out["survey_blueprint"]
    print(f"source: {bp['source']}")
    print(f"summary: {bp.get('narrative_summary')}\n")
    for tab in narrative.TAB_ORDER:
        print(f"=== {tab} ===")
        for b in narrative.beats_for_tab(bp, tab):
            print(f"  {b['order']}. {b['beat_id']:<24} canonical={b.get('canonical')}")
            print(f"       {b.get('intent', '')[:100]}")
