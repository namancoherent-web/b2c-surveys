"""Node 6 — survey_simulator.

Simulates how a representative weighted-persona panel would answer the universal
questionnaire. Does NOT paste market-report percentages onto options.

- Same questions for every region (count from QUESTIONS_PER_TAB_TARGET × 4)
- Regional answer distributions differ via persona weight packs
- sample_size always SAMPLE_SIZE (2500)
- Never ground percentages on Reddit anecdotes
"""

from __future__ import annotations

import asyncio
import json

from pydantic import BaseModel, Field

from src import config
from src.date_utils import current_date_context
from src.llm import get_structured_llm
from src.models import AnswerOption, AnsweredQuestion
from src.question_plan import REGION_SLUGS, REGIONS, SAMPLE_SIZE
from src.state import SurveyState

BATCH_SIZE = 8
_SUM_TO_100_TYPES = {"single_choice", "ranking", "likert_5", "likert_7"}
_CONF_RANK = {"high": 3, "medium": 2, "low": 1}


class _SimAnswer(BaseModel):
    label: str
    percentage: float


class _SimQuestion(BaseModel):
    id: str
    answers: list[_SimAnswer]
    distribution_note: str = Field(
        default="",
        description="Which personas / MKP bands drove the mix (no fake URLs).",
    )
    confidence: str = Field(default="medium", description="high|medium|low")


class _SimBatch(BaseModel):
    estimates: list[_SimQuestion]


def _enforce_sums(qtype: str, answers: list) -> list:
    # change for b2c questionarie — 2-decimal survey-style percentages
    for a in answers:
        try:
            a["percentage"] = float(a.get("percentage") or 0.0)
        except (TypeError, ValueError):
            a["percentage"] = 0.0
        a["percentage"] = max(0.0, min(100.0, a["percentage"]))

    if qtype not in _SUM_TO_100_TYPES or not answers:
        for a in answers:
            a["percentage"] = round(a["percentage"], 2)
        return answers

    total = sum(a["percentage"] for a in answers)
    if total <= 0:
        share = round(100.0 / len(answers), 2)
        for a in answers:
            a["percentage"] = share
    else:
        for a in answers:
            a["percentage"] = a["percentage"] * 100.0 / total
    # Largest-remainder to exactly 100.00 at 2 decimals
    cents = [int(a["percentage"] * 100) for a in answers]
    deficit = 10000 - sum(cents)
    order = sorted(
        range(len(answers)),
        key=lambda i: answers[i]["percentage"] * 100 - cents[i],
        reverse=True,
    )
    k = 0
    while deficit != 0 and order:
        i = order[k % len(order)]
        if deficit > 0:
            cents[i] += 1
            deficit -= 1
        elif cents[i] > 0:
            cents[i] -= 1
            deficit += 1
        k += 1
        if k > len(order) * 5:
            break
    for a, c in zip(answers, cents):
        a["percentage"] = c / 100.0
    return answers


def _fallback(q: dict, confidence: str = "low") -> dict:
    opts = q.get("options") or ["Other"]
    answers = []
    for n, o in enumerate(opts):
        answers.append({
            "label": o,
            "percentage": max(1.0, 100.0 / len(opts) * (1.0 - 0.06 * n)),
            "grounded_in": None,
            "source_market": None,
            "grounding_label": "",
            "confidence": confidence,
        })
    answers = _enforce_sums(q.get("type"), answers)
    return {
        "answers": answers,
        "distribution_note": "Simulated with fallback mix (LLM batch failed).",
        "confidence": confidence,
    }


def _compact_personas(catalog: dict, region: str) -> list:
    personas = catalog.get("personas") or []
    weights = (catalog.get("weights_by_region") or {}).get(region) or {}
    if not weights and catalog.get("weights_by_region"):
        # Prefer Global blend, else first available pack.
        weights = (catalog["weights_by_region"].get("Global")
                   or next(iter(catalog["weights_by_region"].values()), {}))
    out = []
    for p in personas:
        pid = p.get("id")
        out.append({
            "id": pid,
            "name": p.get("name"),
            "weight_pct": weights.get(pid, 0),
            "income_band": p.get("income_band"),
            "price_sensitivity": p.get("price_sensitivity"),
            "tech_adoption": p.get("tech_adoption"),
            "brand_loyalty": p.get("brand_loyalty"),
            "buying_motivation": (p.get("buying_motivation") or [])[:4],
            "pain_points": (p.get("pain_points") or [])[:4],
            "goals": (p.get("goals") or [])[:3],
        })
    out.sort(key=lambda x: x.get("weight_pct") or 0, reverse=True)
    return out


def _mkp_digest(mkp: dict) -> dict:
    return {
        "market_maturity": mkp.get("market_maturity"),
        "avg_income_band": mkp.get("avg_income_band"),
        "price_sensitivity": mkp.get("price_sensitivity"),
        "growth_outlook": mkp.get("growth_outlook"),
        "technology_adoption": mkp.get("technology_adoption"),
        "environmental_awareness": mkp.get("environmental_awareness"),
        "brand_loyalty": mkp.get("brand_loyalty"),
        "consumer_sentiment": mkp.get("consumer_sentiment"),
        "top_pain_points": (mkp.get("top_pain_points") or [])[:8],
        "top_purchase_drivers": (mkp.get("top_purchase_drivers") or [])[:8],
        "competitor_weaknesses": (mkp.get("competitor_weaknesses") or [])[:6],
        "government_incentives": (mkp.get("government_incentives") or [])[:6],
        "evidence_confidence": mkp.get("evidence_confidence"),
    }


PROMPT = """\
You are simulating a representative consumer panel of {sample_size} respondents
for a B2C survey on {segment} in region "{region}". Today is {today}.

You will NOT paste market-report statistics onto answer options.
Instead, reason over the WEIGHTED PERSONA MIX below as if those archetypes
compose the panel. Heavier-weight personas dominate the distribution.

Personas for this region (JSON):
{personas_json}

Market Knowledge Profile digest (constraints on behavior — not option %):
{mkp_json}

Regional note: {regional_note}

THE STUDY'S SEGMENTS — the ONLY groups you may name in a distribution_note:
{segments_block}

HARD RULES:
1. Output one estimate per question id with percentages for EVERY option label
   exactly as given (same labels, same order preferred).
2. single_choice / likert_* percentages MUST sum to 100 (±1).
   multiple_choice: each option's percentage is an INDEPENDENT select-all
   rate — the probability a respondent picked THAT option, on its own,
   nothing to do with what any other option got. Do not mentally divide 100
   points across the options the way you would for single_choice; that habit
   silently produces multi-select totals that land at or near 100% even
   though nothing forced it to. A multi-select question with real options
   should usually total well over 100% (respondents pick more than one) —
   a multi-select total landing suspiciously close to 100% is a sign you
   allocated it as a constrained share instead of independent probabilities.
   Rethink each option's rate on its own before finalizing.
3. Distributions must be NON-UNIFORM and consistent with persona mix + MKP.
4. Do NOT cite Reddit as a numeric source. Do NOT invent source URLs.
5. distribution_note: brief — which of THE STUDY'S SEGMENTS listed above drove
   the answer, e.g. "Price-Driven Subscribers pull the cheaper options; Content-
   Committed Subscribers concentrate on the premium end."
   NEVER name a persona archetype from the panel mix. The personas are an
   internal simulation device; the reader has never seen them and they are not
   defined anywhere in the deliverable. Naming "Budget Binger" or "Eco-Conscious"
   in a client document makes the study look like it was assembled from a
   generic persona library. Use ONLY the segment names given above, spelled
   EXACTLY as written — copy-paste them, do not paraphrase, shorten, pluralize
   differently, or invent a related-sounding group. THE LIST ABOVE IS CLOSED:
   there are no other segments in this study. If you cannot attribute an
   answer to one of the segments above, write a plain descriptive note with NO
   segment name in it rather than naming something new.
   NEVER describe a group by a demographic trait as if it were a segment
   ("younger users", "family-oriented buyers", "premium shoppers") — that is
   inventing a fifth segment through a side door. Age, gender, household and
   income are PROFILING questions elsewhere in the study; their own
   distribution notes must never attribute a finding to one of the segments
   above OR to an invented demographic group — describe the demographic split
   plainly, or write no note at all.
   NEVER claim a market trend, region, growth, income, or age level CAUSES an
   answer ("growth in the region pushes spending up", "income explains this").
   This is a modelled panel, not a measured causal study — describe the
   pattern ("higher-spending responses are more common among X"), never the
   cause.
6. confidence: high if MKP confidence is high and personas strongly constrain
   the question; medium normally; low if question is weakly related to personas.

Questions (JSON):
{questions_json}
"""


def _build_answered(q: dict, est: dict) -> dict:
    answers = []
    by_label = {a.get("label"): a for a in (est.get("answers") or [])}
    for opt in q.get("options") or []:
        raw = by_label.get(opt) or {}
        answers.append({
            "label": opt,
            "percentage": raw.get("percentage", 0.0),
            "grounded_in": None,
            "source_market": None,
            "grounding_label": "",
            "confidence": est.get("confidence") or "medium",
        })
    answers = _enforce_sums(q.get("type"), answers)
    conf = est.get("confidence") or "medium"
    if conf not in _CONF_RANK:
        conf = "medium"
    base = {k: q.get(k) for k in (
        "id", "tab", "text", "type", "options", "chart_type",
        "funnel_position",  # change for b2c questionarie
        "serves_topic", "evidence_aligned", "options_from_evidence",
    )}
    # change for b2c questionarie — Phase A8: story position must survive
    # simulation, or the assembler cannot preserve the narrative order. Absent
    # keys are skipped so the model's own defaults apply (None would not validate).
    for key in (
        "beat_id", "beat_title", "beat_canonical",
        "narrative_order", "question_layer",
    ):
        value = q.get(key)
        if value is not None:
            base[key] = value

    payload = {
        **base,
        "answers": answers,
        "sample_size": SAMPLE_SIZE,
        "distribution_note": est.get("distribution_note") or "",
        "is_grounded": conf in ("high", "medium"),
        "question_confidence": conf,
    }
    return AnsweredQuestion(**payload).model_dump()


async def _simulate_batch(batch: list, personas: list, mkp: dict, region: str,
                          segment: str, today: str, segments_block: str = "") -> list:
    payload = [{
        "id": q.get("id"),
        "text": q.get("text"),
        "type": q.get("type"),
        "options": q.get("options"),
        "tab": q.get("tab"),
    } for q in batch]
    # Testing phase: Sonnet gets its own tuned prompt (see
    # survey_simulator_prompts_sonnet.py for why); DeepSeek's PROMPT below
    # is completely untouched either way.
    active_prompt = PROMPT
    if config.LLM_PROVIDER == "anthropic":
        from src.nodes.survey_simulator_prompts_sonnet import (
            PROMPT as _SONNET_PROMPT,
        )
        active_prompt = _SONNET_PROMPT
    prompt = active_prompt.format(
        sample_size=SAMPLE_SIZE,
        segment=segment,
        region=region,
        today=today,
        personas_json=json.dumps(personas, ensure_ascii=False),
        mkp_json=json.dumps(_mkp_digest(mkp), ensure_ascii=False),
        regional_note=(mkp.get("regional_notes") or {}).get(region, "n/a"),
        segments_block=segments_block or "(none defined — describe the mix in plain words)",
        questions_json=json.dumps(payload, ensure_ascii=False),
    )
    try:
        result: _SimBatch = await get_structured_llm(
            _SimBatch, temperature=0.35, max_tokens=12000
        ).ainvoke(prompt)
        by_id = {e.id: e.model_dump() for e in result.estimates}
    except Exception:  # noqa: BLE001
        by_id = {}

    out = []
    for q in batch:
        est = by_id.get(q.get("id")) or _fallback(q)
        if "confidence" not in est:
            est["confidence"] = "medium"
        # Map structured answers if present.
        if est.get("answers") and isinstance(est["answers"][0], dict):
            pass
        out.append(_build_answered(q, est))
    return out


def _segments_block(blueprint: dict) -> str:
    """The study's own segments, for the simulator's distribution notes."""
    from src import narrative

    segs = narrative.segments_for(blueprint)
    if not segs:
        return ""
    return "\n".join(
        f"- {s.get('name')}: {(s.get('description') or '')[:160]}" for s in segs
    )


async def _simulate_region(questions: list, catalog: dict, mkp: dict,
                           region: str, segment: str, today: str,
                           segments_block: str = "") -> list:
    personas = _compact_personas(catalog, region)
    batches = [questions[i:i + BATCH_SIZE] for i in range(0, len(questions), BATCH_SIZE)]
    parts = await asyncio.gather(*[
        _simulate_batch(b, personas, mkp, region, segment, today, segments_block)
        for b in batches
    ])
    # Preserve question order.
    by_id = {aq["id"]: aq for part in parts for aq in part}
    return [by_id[q["id"]] for q in questions if q.get("id") in by_id]


async def survey_simulator(state: SurveyState) -> dict:
    """Persona-weighted simulation — one question set per region job (A5)."""
    questions = state.get("questions") or []
    catalog = state.get("persona_catalog") or {}
    mkp = state.get("market_knowledge_profile") or {}
    segment = state.get("normalized_segment") or state.get("market_segment") or "the segment"
    today = current_date_context()["iso"]

    regions = state.get("regions_to_simulate") or list(REGIONS)
    # change for b2c questionarie — single-continent job: simulate that region only
    single_geo = len(regions) == 1 and regions[0] != "Global"
    if single_geo:
        sim_regions = list(regions)
    else:
        sim_regions = list(dict.fromkeys(["Global", *regions]))

    # Revision: re-sim questions flagged for simulator/estimator fixes.
    critique = state.get("critique") or {}
    revise_ids = set(critique.get("revise_questions") or [])
    for qi in critique.get("per_question") or []:
        if qi.get("fix_target") in ("simulator", "estimator"):
            if qi.get("id"):
                revise_ids.add(qi["id"])
    prior = {aq.get("id"): aq for aq in (state.get("answered_questions") or [])}

    if revise_ids:
        to_process = [q for q in questions if q.get("id") in revise_ids]
    else:
        to_process = questions

    # change for b2c questionarie — A12: the distribution notes must talk about
    # THIS study's segments. Passing the panel personas alone produced notes
    # naming a dozen archetypes the reader has never seen and the document
    # never defines.
    segments_block = _segments_block(state.get("survey_blueprint") or {})

    # Parallel per-region simulation of the (possibly partial) question set.
    region_results = await asyncio.gather(*[
        _simulate_region(to_process, catalog, mkp, r, segment, today, segments_block)
        for r in sim_regions
    ])

    by_region_partial = {
        r: {aq["id"]: aq for aq in res}
        for r, res in zip(sim_regions, region_results)
    }

    # Primary answered_questions: Global when multi-region; else the sole region.
    primary = regions[0] if single_geo else "Global"
    primary_map = dict(prior) if revise_ids else {}
    primary_map.update(by_region_partial.get(primary) or {})
    if not primary_map and by_region_partial:
        # Fallback to first available sim region.
        first = next(iter(by_region_partial.values()))
        primary_map.update(first)
    answered = [primary_map[q["id"]] for q in questions if q.get("id") in primary_map]

    # Attach data_by_region onto each answered question (legacy multi-region).
    answered_by_region: dict = {}
    for region in sim_regions:
        slug = REGION_SLUGS.get(region) or region.lower().replace(" ", "-").replace("&", "")
        rmap = by_region_partial.get(region) or {}
        prior_region = {}
        for aq in (state.get("answered_questions") or []):
            dbr = aq.get("data_by_region") or {}
            if slug in dbr:
                prior_region[aq["id"]] = dbr[slug]
        answered_by_region[slug] = []
        for q in questions:
            qid = q.get("id")
            aq = rmap.get(qid)
            if aq:
                vals = [a.get("percentage", 0.0) for a in aq.get("answers") or []]
            else:
                vals = prior_region.get(qid) or [
                    a.get("percentage", 0.0) for a in (primary_map.get(qid) or {}).get("answers") or []
                ]
            answered_by_region.setdefault(slug, []).append({"id": qid, "values": vals})

    for aq in answered:
        dbr = dict(aq.get("data_by_region") or {})
        for region in sim_regions:
            slug = REGION_SLUGS.get(region) or region.lower().replace(" ", "-").replace("&", "")
            rmap = by_region_partial.get(region) or {}
            other = rmap.get(aq.get("id"))
            if other:
                dbr[slug] = [a.get("percentage", 0.0) for a in other.get("answers") or []]
        aq["data_by_region"] = dbr

    confs = [aq.get("question_confidence") or "low" for aq in answered]
    avg = (
        sum(_CONF_RANK.get(c, 1) for c in confs) / max(1, len(confs)) / 3.0 * 100.0
    )
    grounded = sum(1 for aq in answered if aq.get("is_grounded"))

    return {
        "answered_questions": answered,
        "answered_by_region": answered_by_region,
        "grounded_question_count": grounded,
        "simulation_confidence_avg": round(avg, 1),
    }


# LangGraph uses the async ``survey_simulator`` directly.