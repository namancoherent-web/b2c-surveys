"""Node 3 — evidence_synthesizer.

Compresses the raw ``evidence_pool`` into a Market Knowledge Profile (MKP) —
the structured market model downstream persona + simulation nodes consume.

Hard rules
----------
- Structural bands (maturity, income, price sensitivity, growth, tech, eco,
  loyalty, sentiment) come from reports / web / stats — NEVER from social voice.
- Reddit / YouTube / Twitter / review language fills pain_points,
  consumer_language, feature_requests (language only — never percentages).
- This node does NOT emit answer percentages.
"""

from __future__ import annotations

import json
from typing import List

from src import cache_store
from src.date_utils import current_date_context
from src.evidence_utils import VOICE_ONLY_ORIGINS
from src.llm import get_structured_llm
from src.models import MarketKnowledgeProfile
from src.question_plan import GEOGRAPHIC_REGIONS, REGIONS
from src.state import SurveyState

_SCHEMA_V = "mkp_v1"
_MAX_STRUCT_FACTS = 40
_MAX_VOICE_FACTS = 30


def _is_voice(fact: dict) -> bool:
    origin = (fact.get("origin") or "").lower()
    topic = (fact.get("topic") or "").lower()
    # change for b2c questionarie — youtube/twitter/reddit are voice-only
    if origin in VOICE_ONLY_ORIGINS:
        return True
    return any(k in topic for k in ("complaint", "customer", "voice", "review"))


def _compact(facts: List[dict], limit: int) -> List[dict]:
    out = []
    for f in facts[:limit]:
        out.append({
            "fact": (f.get("fact") or "")[:350],
            "origin": f.get("origin"),
            "authority": f.get("authority"),
            "topic": f.get("topic"),
            "value": f.get("value"),
            "source_ref": (f.get("source_ref") or "")[:160],
        })
    return out


def _fallback_mkp(segment: str, thin: bool) -> dict:
    return MarketKnowledgeProfile(
        market_maturity="growing",
        avg_income_band="medium",
        price_sensitivity="medium",
        growth_outlook="growing",
        technology_adoption="mainstream",
        environmental_awareness="medium",
        brand_loyalty="medium",
        government_incentives=[],
        top_pain_points=[f"Price of {segment}", "Availability", "Product quality consistency"],
        top_purchase_drivers=["Quality", "Value for money", "Brand trust"],
        competitor_weaknesses=["After-sales support", "Repair cost"],
        consumer_sentiment="mixed",
        consumer_language=[],
        feature_requests=[],
        regional_notes={r: "Insufficient regional evidence; using global defaults." for r in GEOGRAPHIC_REGIONS},
        evidence_confidence="low",
        citations=[],
        synthesis_notes=[
            "Fallback MKP — synthesizer LLM failed or evidence empty.",
            "Reddit excluded from quantitative bands.",
        ] + (["evidence_empty=True"] if thin else []),
    ).model_dump()


PROMPT = """\
You are a senior market-research analyst synthesizing a Market Knowledge Profile
for a B2C consumer survey on "{segment}" (scope region: {region}). Today is {today}.

Build ONE MarketKnowledgeProfile from the evidence below.

HARD RULES:
1. Quantitative / structural fields (market_maturity, avg_income_band,
   price_sensitivity, growth_outlook, technology_adoption,
   environmental_awareness, brand_loyalty, consumer_sentiment, government_incentives)
   may ONLY be justified by STRUCTURAL EVIDENCE (cmi_database / stat / exa / web).
   NEVER set these fields from Reddit / YouTube / Twitter anecdotes alone.
   NEVER invent or paste percentages from social posts.
2. top_pain_points, consumer_language, feature_requests SHOULD prefer VOICE
   EVIDENCE (Reddit / YouTube / Twitter / complaints / reviews) — use real
   consumer wording only (language, not numbers).
3. regional_notes: short qualitative notes for each of: {regions}.
   No percentages in regional_notes.
4. evidence_confidence = low if structural evidence is thin/conflicting;
   medium if adequate; high if multiple authoritative sources agree.
5. citations: include up to 12 STRUCTURAL source_refs that support key judgments.
6. synthesis_notes: record conflicts and that social voice was excluded from bands.

STRUCTURAL EVIDENCE (JSON):
{struct_json}

VOICE EVIDENCE (JSON) — language/pain only:
{voice_json}
"""


def evidence_synthesizer(state: SurveyState) -> dict:
    """Compress evidence_pool → market_knowledge_profile."""
    pool = list(state.get("evidence_pool") or [])
    segment = state.get("normalized_segment") or state.get("market_segment") or "the segment"
    region = state.get("region") or "Global"
    empty = bool(state.get("evidence_empty") or not pool)

    digest = cache_store.make_key(
        _SCHEMA_V,
        segment,
        region,  # change for b2c questionarie — region-scoped MKP cache (A4b)
        [(f.get("source_ref"), (f.get("fact") or "")[:80]) for f in pool[:80]],
    )
    cached = cache_store.get_json("mkp", digest)
    if cached and cached.get("market_knowledge_profile"):
        mkp = cached["market_knowledge_profile"]
        thin = (mkp.get("evidence_confidence") or "low") == "low" or empty
        return {"market_knowledge_profile": mkp, "grounding_thin": thin}

    if empty:
        mkp = _fallback_mkp(segment, thin=True)
        cache_store.set_json("mkp", digest, {"market_knowledge_profile": mkp})
        return {"market_knowledge_profile": mkp, "grounding_thin": True}

    structural = [f for f in pool if not _is_voice(f)]
    voice = [f for f in pool if _is_voice(f)]
    # Prefer numeric structural facts first.
    structural.sort(key=lambda f: (1 if f.get("is_numeric") else 0, f.get("authority") == "high"), reverse=True)
    if not structural:
        structural = pool[:_MAX_STRUCT_FACTS]

    today = current_date_context()["iso"]
    prompt = PROMPT.format(
        segment=segment,
        region=region,
        today=today,
        regions=", ".join(REGIONS),
        struct_json=json.dumps(_compact(structural, _MAX_STRUCT_FACTS), ensure_ascii=False),
        voice_json=json.dumps(_compact(voice, _MAX_VOICE_FACTS), ensure_ascii=False),
    )

    try:
        result: MarketKnowledgeProfile = get_structured_llm(
            MarketKnowledgeProfile, temperature=0.1, max_tokens=8000
        ).invoke(prompt)
        mkp = result.model_dump()
    except Exception:  # noqa: BLE001
        mkp = _fallback_mkp(segment, thin=True)

    # Ensure all regions have a note key.
    notes = dict(mkp.get("regional_notes") or {})
    for r in REGIONS:
        notes.setdefault(r, "No specific regional signal; use global MKP bands.")
    mkp["regional_notes"] = notes

    thin = (mkp.get("evidence_confidence") or "low") == "low"
    cache_store.set_json("mkp", digest, {"market_knowledge_profile": mkp})
    return {"market_knowledge_profile": mkp, "grounding_thin": thin}


if __name__ == "__main__":
    demo = {
        "normalized_segment": "Organic Milk",
        "region": "Global",
        "evidence_empty": False,
        "evidence_pool": [
            {
                "fact": "Organic milk holds about 41% of premium dairy sales in North America in 2025.",
                "value": "41%", "is_numeric": True, "topic": "regional share",
                "origin": "stat", "authority": "high",
                "source_ref": "https://example.gov/dairy",
            },
            {
                "fact": "I switched because my toddler had reactions to conventional milk and organic was gentler.",
                "value": None, "is_numeric": False, "topic": "complaints",
                "origin": "reddit", "authority": "low",
                "source_ref": "https://reddit.com/r/example",
            },
        ],
    }
    out = evidence_synthesizer(demo)
    print(json.dumps(out.get("market_knowledge_profile"), indent=2))
