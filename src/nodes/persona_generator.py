"""Node 4 — persona_generator.

Creates 10–20 weighted consumer archetypes for the segment.

# change for b2c questionarie — A5.1: region jobs generate region-native
archetypes + a single weight pack for that region (no global reweight catalog).
Legacy Global runs still produce multi-region weight packs.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel, Field

from src import cache_store
from src.date_utils import current_date_context
from src.llm import get_structured_llm
from src.models import PersonaArchetype, PersonaCatalog
from src.question_plan import REGIONS
from src.state import SurveyState

_SCHEMA_V = "personas_v2"  # change for b2c questionarie — region-native personas
_MIN_PERSONAS = 10
_MAX_PERSONAS = 16


class _PersonaBatch(BaseModel):
    personas: list[PersonaArchetype] = Field(min_length=8, max_length=20)
    notes: list[str] = Field(default_factory=list)


class _WeightPack(BaseModel):
    region: str
    weights: dict[str, float] = Field(description="persona_id -> percent share of population")
    notes: list[str] = Field(default_factory=list)


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (name or "").lower()).strip("_")
    return s[:48] or "persona"


def _normalize_weights(weights: dict, persona_ids: list[str]) -> dict[str, float]:
    cleaned = {}
    for pid in persona_ids:
        try:
            cleaned[pid] = max(0.0, float(weights.get(pid, 0.0)))
        except (TypeError, ValueError):
            cleaned[pid] = 0.0
    total = sum(cleaned.values())
    if total <= 0:
        equal = round(100.0 / max(1, len(persona_ids)), 1)
        cleaned = {pid: equal for pid in persona_ids}
    else:
        cleaned = {pid: round(v * 100.0 / total, 1) for pid, v in cleaned.items()}
    # Absorb drift into largest.
    drift = round(100.0 - sum(cleaned.values()), 1)
    if abs(drift) >= 0.1 and cleaned:
        top = max(cleaned, key=cleaned.get)
        cleaned[top] = round(max(0.0, cleaned[top] + drift), 1)
    return cleaned


def _fallback_personas(segment: str) -> list[dict]:
    templates = [
        ("young_professional", "Young Professional", "25-35", "middle", "early"),
        ("budget_family", "Budget Family", "30-45", "lower-middle", "mainstream"),
        ("luxury_buyer", "Luxury Buyer", "35-55", "high", "early"),
        ("retired_consumer", "Retired Consumer", "60-75", "middle", "laggard"),
        ("eco_conscious", "Environmentally Conscious Buyer", "28-50", "upper-middle", "early"),
        ("value_seeker", "Value Seeker", "22-40", "low", "mainstream"),
        ("brand_loyalist", "Brand Loyalist", "30-55", "middle", "mainstream"),
        ("tech_enthusiast", "Tech Enthusiast", "20-35", "middle", "cutting_edge"),
        ("rural_practical", "Rural Practical Buyer", "35-60", "lower-middle", "laggard"),
        ("urban_convenience", "Urban Convenience Buyer", "25-45", "middle", "early"),
        ("health_focused", "Health-Focused Buyer", "30-50", "upper-middle", "early"),
        ("small_business_owner", "Small Business Owner", "30-55", "middle", "mainstream"),
    ]
    out = []
    for pid, name, age, income, tech in templates:
        out.append(PersonaArchetype(
            id=pid, name=name, age_band=age, income_band=income,
            education="mixed", lifestyle=f"Typical {name.lower()} buying {segment}",
            tech_adoption=tech, brand_loyalty="medium", price_sensitivity="medium",
            buying_motivation=["quality", "value"],
            pain_points=["price", "availability"],
            goals=["reliability"],
            typical_behaviors=["compares options online"],
        ).model_dump())
    return out


PERSONA_PROMPT = """\
You are a consumer-segmentation lead building PERSONA ARCHETYPES for a B2C
survey on "{segment}" focused on the region "{region}". Today is {today}.

Using the Market Knowledge Profile and audience frame, create {n_personas}
DISTINCT weighted-ready archetypes (not 2500 individuals) that reflect THIS
region's buyers — local channels, price sensitivity, and cultural patterns.

Each persona needs: id (snake_case), name, age_band, income_band, education,
lifestyle, tech_adoption, brand_loyalty, price_sensitivity, buying_motivation,
pain_points, goals, typical_behaviors.

RULES:
- Cover the real diversity implied by the MKP (price tiers, eco, luxury, family…).
- Absorb consumer_language / top_pain_points into pain_points with REAL wording.
- Do NOT invent precise population percentages here — weights come later.
- Personas must be about buyers/users of {segment} in {region}.

Audience frame (JSON):
{audience_json}

Market Knowledge Profile (JSON):
{mkp_json}
"""

WEIGHT_PROMPT = """\
You assign regional population WEIGHTS for consumer personas in "{segment}".
Today is {today}. Region: "{region}".

Return weights for EVERY persona id below so they SUM to 100 (±1).
Use MKP regional_notes + structural bands. Example patterns:
- Europe: higher eco / sustainability personas
- Asia Pacific: higher value / young professional / mobile-convenience
- Latin America & Middle East & Africa: higher price sensitivity / budget family
- North America: higher premium / brand / convenience mix

Persona ids: {persona_ids}

Compact persona cards (JSON):
{personas_json}

MKP regional note for this region: {regional_note}

Full MKP bands (JSON):
{mkp_bands_json}
"""


def persona_generator(state: SurveyState) -> dict:
    """Build PersonaCatalog — region-native when job is a single continent (A5.1)."""
    segment = state.get("normalized_segment") or state.get("market_segment") or "the segment"
    mkp = state.get("market_knowledge_profile") or {}
    audience = state.get("audience_profile") or {}
    regions = state.get("regions_to_simulate") or list(REGIONS)
    job_region = state.get("region") or "Global"
    today = current_date_context()["iso"]

    # change for b2c questionarie — cache key includes region
    cache_key = cache_store.make_key(
        _SCHEMA_V, segment, job_region,
        mkp.get("synthesis_notes"), mkp.get("evidence_confidence"),
    )
    cached = cache_store.get_json("personas", cache_key)
    if cached and cached.get("persona_catalog"):
        return {"persona_catalog": cached["persona_catalog"]}

    n = 12
    prompt = PERSONA_PROMPT.format(
        segment=segment, region=job_region, today=today, n_personas=n,
        audience_json=json.dumps(audience, ensure_ascii=False)[:2000],
        mkp_json=json.dumps(mkp, ensure_ascii=False)[:6000],
    )
    try:
        batch: _PersonaBatch = get_structured_llm(
            _PersonaBatch, temperature=0.35, max_tokens=8000
        ).invoke(prompt)
        personas = []
        seen = set()
        for p in batch.personas:
            d = p.model_dump()
            d["id"] = _slug(d.get("id") or d.get("name") or "persona")
            if d["id"] in seen:
                d["id"] = f"{d['id']}_{len(seen)}"
            seen.add(d["id"])
            personas.append(d)
        if len(personas) < _MIN_PERSONAS:
            personas = _fallback_personas(segment)
        personas = personas[:_MAX_PERSONAS]
        notes = list(batch.notes or [])
    except Exception:  # noqa: BLE001
        personas = _fallback_personas(segment)
        notes = ["Fallback personas — LLM generation failed."]

    persona_ids = [p["id"] for p in personas]
    cards = [{
        "id": p["id"], "name": p["name"], "income_band": p["income_band"],
        "price_sensitivity": p["price_sensitivity"], "tech_adoption": p["tech_adoption"],
        "brand_loyalty": p["brand_loyalty"], "pain_points": (p.get("pain_points") or [])[:3],
    } for p in personas]
    mkp_bands = {
        k: mkp.get(k) for k in (
            "market_maturity", "avg_income_band", "price_sensitivity",
            "environmental_awareness", "brand_loyalty", "technology_adoption",
            "consumer_sentiment", "growth_outlook",
        )
    }
    regional_notes = mkp.get("regional_notes") or {}

    # change for b2c questionarie — cost/time optimization step 2. Each
    # region's weight-pack call is fully independent: its own try/except, and
    # it only ever writes to its own `weights_by_region[region]` key. Run
    # them concurrently instead of one after another (matters when a legacy
    # multi-region job requests several regions at once).
    weights_by_region: dict = {}
    llm = get_structured_llm(_WeightPack, temperature=0.2, max_tokens=4000)

    def _weight_for_region(region: str):
        try:
            pack: _WeightPack = llm.invoke(WEIGHT_PROMPT.format(
                segment=segment, today=today, region=region,
                persona_ids=", ".join(persona_ids),
                personas_json=json.dumps(cards, ensure_ascii=False),
                regional_note=regional_notes.get(region, "n/a"),
                mkp_bands_json=json.dumps(mkp_bands, ensure_ascii=False),
            ))
            weights = _normalize_weights(pack.weights, persona_ids)
            return region, weights, list(pack.notes or [])
        except Exception:  # noqa: BLE001
            equal = round(100.0 / len(persona_ids), 1)
            weights = _normalize_weights({pid: equal for pid in persona_ids}, persona_ids)
            return region, weights, []

    with ThreadPoolExecutor(max_workers=max(1, len(regions))) as ex:
        for region, weights, region_notes in ex.map(_weight_for_region, regions):
            weights_by_region[region] = weights
            notes.extend(region_notes)

    # change for b2c questionarie — region job: no Global blend required
    single_geo = (
        len(regions) == 1
        and regions[0] != "Global"
    )
    if not single_geo and "Global" not in weights_by_region and weights_by_region:
        blended = {pid: 0.0 for pid in persona_ids}
        for w in weights_by_region.values():
            for pid in persona_ids:
                blended[pid] += w.get(pid, 0.0)
        n_reg = max(1, len(weights_by_region))
        weights_by_region["Global"] = _normalize_weights(
            {pid: blended[pid] / n_reg for pid in persona_ids}, persona_ids
        )

    conf = "low" if (mkp.get("evidence_confidence") == "low" or "Fallback" in " ".join(notes)) else "medium"
    catalog = PersonaCatalog(
        personas=personas,
        weights_by_region=weights_by_region,
        generation_notes=notes[:20],
        persona_confidence=conf,
    ).model_dump()

    cache_store.set_json("personas", cache_key, {"persona_catalog": catalog})
    return {"persona_catalog": catalog}


if __name__ == "__main__":
    demo = {
        "normalized_segment": "Organic Milk",
        "regions_to_simulate": ["Europe", "Asia Pacific"],
        "audience_profile": {"demographics": {"age": "25-55"}},
        "market_knowledge_profile": {
            "market_maturity": "growing",
            "avg_income_band": "medium",
            "price_sensitivity": "medium",
            "growth_outlook": "growing",
            "technology_adoption": "mainstream",
            "environmental_awareness": "high",
            "brand_loyalty": "medium",
            "top_pain_points": ["price premium", "stockouts"],
            "consumer_language": ["too expensive for everyday"],
            "evidence_confidence": "medium",
            "regional_notes": {
                "Europe": "Higher eco and organic salience.",
                "Asia Pacific": "Strong value orientation.",
            },
            "synthesis_notes": [],
        },
    }
    print(json.dumps(persona_generator(demo), indent=2)[:2000])
