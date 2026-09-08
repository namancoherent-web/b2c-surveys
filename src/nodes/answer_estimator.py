"""Node 5 — answer_estimator.

For each Node-4 question, produce an answer distribution (a percentage per
option) that is GROUNDED in real figures from the evidence index wherever a
figure matches, and honestly INFERRED (and marked) otherwise. This turns the
questions into a simulated consumer survey with numbers.

Core discipline: LOOKUP FIRST, INFER SECOND (per option)
--------------------------------------------------------
For each option we first try to match a real figure (by serves_topic, then by
option-label ↔ figure-category). A match carries the figure's source_ref
(grounded_in), source_market, and Node 3's grounding_label end-to-end; an
inferred option has grounded_in=None. The integrity contract: a borrowed
adjacent figure NEVER loses its label, and nothing inferred is dressed up as
direct real data.

Sum rules: single_choice / ranking / likert_* sum to 100 (±1); multiple_choice
options are INDEPENDENT 0-100 and are NOT summed to 100.
"""

import asyncio

from pydantic import BaseModel

from src.date_utils import current_date_context
from src.llm import get_structured_llm
from src.models import AnswerOption, AnsweredQuestion
from src.question_plan import REGIONS, SAMPLE_SIZE
from src.state import SurveyState

BATCH_SIZE = 12
MAX_EVIDENCE_PER_Q = 8
_SUM_TO_100_TYPES = {"single_choice", "ranking", "likert_5", "likert_7"}
_CONF_RANK = {"high": 3, "medium": 2, "low": 1}
# Relevance floor: only medium+ facts may ground an answer. "low" (e.g. body
# scrub / jaggery vs organic milk) is context-only — never grounding-eligible.
_GROUNDABLE_RELEVANCE = {"high", "medium"}


def _norm(s: str) -> str:
    return "".join(ch for ch in (s or "").lower() if ch.isalnum())


class _QuestionEstimate(BaseModel):
    id: str
    answers: list[AnswerOption]
    distribution_note: str = ""


class _EstimateBatch(BaseModel):
    estimates: list[_QuestionEstimate]


# --- Regional pass: LLM-estimated per-region variants ------------------------
# Compact arrays aligned to option order (keeps DeepSeek's output small).
_REGION_FIELD_SLUGS = {
    "north_america": "north-america",
    "europe": "europe",
    "asia_pacific": "asia-pacific",
    "latin_america": "latin-america",
    "middle_east_africa": "middle-east-africa",
}
REGIONAL_SUBBATCH = 6  # questions per regional LLM call (5 regions × options each)


class _RegionalEstimate(BaseModel):
    id: str
    north_america: list[float]
    europe: list[float]
    asia_pacific: list[float]
    latin_america: list[float]
    middle_east_africa: list[float]


class _RegionalBatch(BaseModel):
    estimates: list[_RegionalEstimate]


REGIONAL_PROMPT_TEMPLATE = """\
You are a market-research data modeler estimating REGIONAL answer distributions
for a B2C survey on {segment}. Today's date is {today}.

For each question you get: its ordered options, its GLOBAL percentages (same
order), its type, and evidence (real figures from market reports and the web —
regional shares, regional consumer statistics — where available).

For EACH of the 5 regions — North America, Europe, Asia Pacific, Latin America,
Middle East & Africa — output a percentages array aligned EXACTLY to the option
order (same length as options).
- GROUND the regional numbers in the evidence wherever it carries regional
  figures for the topic.
- Where evidence is absent, INFER realistic regional differences from known
  consumer patterns (e.g., price sensitivity is higher in Latin America and
  Middle East & Africa; e-commerce/mobile penetration higher in Asia Pacific;
  sustainability emphasis higher in Europe; premium willingness higher in North
  America) — plausible deviations from the global baseline, NOT random noise,
  and NOT identical to the global values.
- single_choice / ranking / likert questions: each region's array must sum to
  100 (±1). multiple_choice: values are independent 0-100; do NOT force a sum.
- Keep distributions realistic; no flat/uniform arrays.

Questions (JSON):
{questions_json}
"""


def _regional_llm():
    return get_structured_llm(_RegionalBatch, temperature=0.3, max_tokens=16000)


def _llm():
    return get_structured_llm(_EstimateBatch, temperature=0.3, max_tokens=16000)


def _enforce_region_values(qtype: str, values: list, n_opts: int) -> list:
    """Clamp/pad a region's value array and enforce the sum contract."""
    vals = [max(0.0, min(100.0, float(v or 0))) for v in (values or [])[:n_opts]]
    while len(vals) < n_opts:
        vals.append(0.0)
    if qtype in _SUM_TO_100_TYPES:
        total = sum(vals)
        if total <= 0:
            vals = [round(100.0 / n_opts, 1)] * n_opts
        else:
            vals = [round(v * 100.0 / total, 1) for v in vals]
        drift = round(100.0 - sum(vals), 1)
        if abs(drift) >= 0.1 and vals:
            i = max(range(len(vals)), key=lambda k: vals[k])
            vals[i] = round(max(0.0, vals[i] + drift), 1)
    else:
        vals = [round(v, 1) for v in vals]
    return vals


async def _estimate_regions(answered: list, slices: dict, segment: str, today: str) -> None:
    """Second LLM pass: attach data_by_region to each answered question.

    Evidence-first (regional figures from DB/web in the slice), realistic
    inference otherwise. Failures leave data_by_region absent — the assembler
    then falls back to its deterministic approximation, so a survey always
    ships with regional variants.
    """
    import json

    for start in range(0, len(answered), REGIONAL_SUBBATCH):
        chunk = answered[start:start + REGIONAL_SUBBATCH]
        payload = [{
            "id": aq["id"],
            "question": aq.get("text"),
            "type": aq.get("type"),
            "options": aq.get("options"),
            "global_percentages": [a.get("percentage") for a in aq.get("answers") or []],
            "evidence": slices.get(aq["id"], []),
        } for aq in chunk]
        prompt = REGIONAL_PROMPT_TEMPLATE.format(
            segment=segment, today=today,
            questions_json=json.dumps(payload, ensure_ascii=False),
        )
        try:
            result: _RegionalBatch = await _regional_llm().ainvoke(prompt)
            by_id = {e.id: e for e in result.estimates}
        except Exception:  # noqa: BLE001 — assembler falls back deterministically
            continue
        for aq in chunk:
            est = by_id.get(aq["id"])
            if not est:
                continue
            n_opts = len(aq.get("options") or [])
            aq["data_by_region"] = {
                slug: _enforce_region_values(aq.get("type"), getattr(est, field), n_opts)
                for field, slug in _REGION_FIELD_SLUGS.items()
            }


PROMPT_TEMPLATE = """\
You are a market-research data modeler estimating answer distributions for a
B2C survey on {segment} ({region}). Today's date is {today}.

For each question you get its id, text, type, options, serves_topic, and a
matching evidence slice (real figures with source_ref, source_market,
grounding_label, verbatim value).

For EACH option:
1. LOOK UP first: if a provided figure's category/measure matches this option,
   USE its value as the basis. Carry grounded_in = that figure's source_ref,
   source_market, and its grounding_label EXACTLY as given (do not drop or alter
   it). Set confidence: high if the figure is a direct match + recent/stat/
   corroborated; medium if adjacent or older.
2. If no figure matches, INFER a realistic percentage from category norms, the
   audience, and context. Set grounded_in=null, source_market=null,
   grounding_label="", confidence=low (or medium if strongly context-backed).
NEVER attach a source_ref that doesn't match the option. NEVER drop a borrowed
figure's grounding_label. NEVER ground a percentage on a lone Reddit anecdote —
Reddit is for customer-language options, not stat-quality numbers unless
corroborated by stat/cmi_database/web/exa sources. An honest inferred number
beats a misattributed one.

GEOGRAPHY: this file covers ONE geography and every respondent is in it. Where
the evidence carries a figure for that exact geography, prefer it; where it does
not, a figure for the parent region is an acceptable approximation. Never mix
two geographies inside one question, and never produce a breakdown BY geography
— the geography is the file, not a variable inside it.

Sum rules (STRICT):
- single_choice / ranking / likert_5 / likert_7 → option percentages SUM to 100
  (±1 for rounding).
- multiple_choice → each option is an INDEPENDENT 0-100% (share selecting it);
  they do NOT sum to 100. Do not normalize these.
Make distributions REALISTIC, not uniform (reflect evidence + audience; if
context stresses price, price ranks high). If a figure gives a FULL distribution
(e.g. all regional shares), reuse the whole distribution across matching options.
Write a short distribution_note per question: which figures grounded which
options, and what was inferred.

Example:
- single_choice "Which region are you in?" with a regional figure available →
  options grounded (direct → grounding_label=""; adjacent → grounding_label=
  "based on related Sheep Milk market data"), summing to 100.
- multiple_choice "Which factors influence your purchase?" → Price 73 and
  Quality 78 grounded from evidence, the rest inferred (e.g. Reviews 60,
  Packaging 41); these do NOT sum to 100.

Questions (JSON):
{questions_json}
"""


def _question_evidence(q: dict, index: dict) -> list:
    """Compact evidence slice for a question (serves_topic + tab matches first)."""
    grounded = index.get("grounded_facts") or []
    by_tab = index.get("by_tab") or {}
    serves = q.get("serves_topic")

    facts, seen = [], set()
    for i in by_tab.get(q.get("tab"), []):
        if i < len(grounded):
            facts.append(grounded[i]); seen.add(i)
    if serves:
        for i, f in enumerate(grounded):
            if i not in seen and f.get("topic") == serves:
                facts.append(f); seen.add(i)

    # RELEVANCE FLOOR (BUG 3a): drop low-relevance facts from the grounding
    # slice — they may inform context but must never ground an answer.
    facts = [f for f in facts if (f.get("relevance") or "low") in _GROUNDABLE_RELEVANCE]

    facts.sort(
        key=lambda f: (
            1 if serves and f.get("topic") == serves else 0,
            1 if f.get("is_direct_match") else 0,
            0 if f.get("origin") in ("reddit", "youtube", "twitter") else 1,
            _CONF_RANK.get("high" if f.get("origin") == "stat" else "low", 0),
        ),
        reverse=True,
    )
    out = []
    for f in facts[:MAX_EVIDENCE_PER_Q]:
        fig = f.get("figure") or {}
        out.append({
            "source_ref": f.get("source_ref"),
            "source_market": f.get("source_market"),
            "origin": f.get("origin"),
            "is_direct_match": f.get("is_direct_match"),
            "relevance": f.get("relevance"),
            "grounding_label": f.get("grounding_label", ""),
            "topic": f.get("topic"),
            "measure": fig.get("measure"),
            "verbatim": fig.get("verbatim"),
            "unit": fig.get("unit"),
            "fact": (f.get("fact") or "")[:200],
        })
    return out


def _enforce_sums(qtype: str, answers: list) -> list:
    """Guarantee sum-to-100 types normalize to 100.00 at 2 decimals;
    multiple_choice stays independent (clamped 0-100, 2dp)."""
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


def _fallback_estimate(q: dict) -> dict:
    """Inferred-only distribution if the LLM didn't return this question."""
    opts = q.get("options") or ["Other"]
    answers = [{
        "label": o, "percentage": 0.0, "grounded_in": None,
        "source_market": None, "grounding_label": "", "confidence": "low",
    } for o in opts]
    # Mild non-uniform shape rather than a flat cop-out.
    for n, a in enumerate(answers):
        a["percentage"] = max(1.0, 100.0 / len(answers) * (1.0 - 0.05 * n))
    return {
        "answers": _enforce_sums(q.get("type"), answers),
        "distribution_note": "Inferred (no estimate returned); not grounded.",
    }


def _reconcile_answers(answers: list, slice_: list, segment: str) -> list:
    """Enforce grounding integrity in CODE (don't trust the LLM to be consistent):

    - BUG 1: grounded_in is set to a REAL source_ref iff the option maps to a
      grounding-eligible figure; otherwise grounded_in AND source_market are null.
      (grounded_in != None) ⟺ (a real figure was used).
    - BUG 3a: figures are only in ``slice_`` if relevance >= medium, so grounding
      on a low-relevance market can't happen — such options get demoted.
    - BUG 3b: an adjacent-market ground carries the honest label; a direct-match
      (source_market == target segment) carries an empty label.
    """
    by_market, by_ref = {}, {}
    for f in slice_:
        if f.get("source_ref"):
            by_ref[f["source_ref"]] = f
            if f.get("source_market"):
                by_market.setdefault(_norm(f["source_market"]), f)
    seg_n = _norm(segment)

    used_refs = set()  # a single fact may ground at most ONE option per question
    for a in answers:
        gi, sm = a.get("grounded_in"), a.get("source_market")
        claims = bool(gi) or bool(sm) or bool((a.get("grounding_label") or "").strip())
        fig = by_ref.get(gi) or (by_market.get(_norm(sm)) if sm else None)

        # Anti-fabrication + social-voice numeric guard.
        # change for b2c questionarie — youtube/twitter/reddit never ground % alone
        if claims and fig and fig.get("source_ref") not in used_refs:
            if fig.get("origin") in ("reddit", "youtube", "twitter") and not (
                fig.get("corroborated_by") or []
            ):
                claims = False
            market = fig.get("source_market")
            ref = fig.get("source_ref")
            used_refs.add(ref)
            a["grounded_in"] = ref
            a["source_market"] = market
            a["grounding_label"] = "" if _norm(market) == seg_n else \
                f"based on related {market} data"
        else:
            # Not verifiably grounded on an eligible figure → honest inference.
            a["grounded_in"] = None
            a["source_market"] = None
            a["grounding_label"] = ""
            if a.get("confidence") == "high":
                a["confidence"] = "low"
    return answers


def _build_answered(q: dict, est: dict, slice_: list, segment: str) -> dict:
    """Merge a question with its estimate into a validated AnsweredQuestion dict."""
    answers = est.get("answers") or []
    # Align option count to the question's options (don't lose/duplicate options).
    if len(answers) != len(q.get("options") or []):
        labels = {a.get("label") for a in answers}
        for o in q.get("options") or []:
            if o not in labels:
                answers.append({"label": o, "percentage": 0.0, "grounded_in": None,
                                "source_market": None, "grounding_label": "", "confidence": "low"})
    answers = _reconcile_answers(answers, slice_, segment)
    answers = _enforce_sums(q.get("type"), answers)

    is_grounded = any(a.get("grounded_in") for a in answers)
    rollup = "low"
    if answers:
        rollup = max((a.get("confidence", "low") for a in answers),
                     key=lambda c: _CONF_RANK.get(c, 1))

    payload = {
        **{k: q.get(k) for k in (
            "id", "tab", "text", "type", "options", "chart_type",
            "funnel_position",  # change for b2c questionarie
            "serves_topic", "evidence_aligned",
        )},
        "answers": answers,
        "sample_size": SAMPLE_SIZE,
        "distribution_note": est.get("distribution_note", ""),
        "is_grounded": is_grounded,
        "question_confidence": rollup,
    }
    return AnsweredQuestion(**payload).model_dump()


async def _estimate_batch(batch: list, index: dict, segment: str, region: str, today: str) -> list:
    """Estimate one batch of questions; fall back per-question on failure."""
    import json

    slices = {q.get("id"): _question_evidence(q, index) for q in batch}
    payload = []
    for q in batch:
        payload.append({
            "id": q.get("id"),
            "text": q.get("text"),
            "type": q.get("type"),
            "options": q.get("options"),
            "serves_topic": q.get("serves_topic"),
            "evidence": slices.get(q.get("id"), []),
        })
    prompt = PROMPT_TEMPLATE.format(
        segment=segment, region=region, today=today,
        regions=", ".join(REGIONS),
        questions_json=json.dumps(payload, ensure_ascii=False),
    )
    try:
        result: _EstimateBatch = await _llm().ainvoke(prompt)
        by_id = {e.id: e.model_dump() for e in result.estimates}
    except Exception:  # noqa: BLE001 — keep the run alive; infer this batch
        by_id = {}

    answered = [
        _build_answered(q, by_id.get(q.get("id")) or _fallback_estimate(q),
                        slices.get(q.get("id"), []), segment)
        for q in batch
    ]
    # Second pass: LLM-estimated per-region variants (evidence-first; assembler
    # falls back deterministically for any question this pass doesn't cover).
    await _estimate_regions(answered, slices, segment, today)
    return answered


async def answer_estimator(state: SurveyState) -> dict:
    """Estimate grounded/inferred answer distributions for all questions (Node 5)."""
    questions = state.get("questions") or []
    index = state.get("evidence_index") or {}
    segment = state.get("normalized_segment") or state.get("market_segment") or "the segment"
    region = state.get("region") or "Global"
    today = current_date_context()["iso"]

    # Revision loop: re-estimate only flagged questions, carry the rest over.
    critique = state.get("critique") or {}
    revise_ids = set(critique.get("revise_questions") or [])
    if revise_ids:
        prior = {aq.get("id"): aq for aq in (state.get("answered_questions") or [])}
        to_process = [q for q in questions if q.get("id") in revise_ids]
    else:
        prior, to_process = {}, questions

    batches = [to_process[i:i + BATCH_SIZE] for i in range(0, len(to_process), BATCH_SIZE)]
    results = await asyncio.gather(
        *[_estimate_batch(b, index, segment, region, today) for b in batches]
    )
    processed = {aq["id"]: aq for batch in results for aq in batch}

    # Reassemble in original question order, reusing carried-over answers.
    answered = []
    for q in questions:
        qid = q.get("id")
        if qid in processed:
            answered.append(processed[qid])
        elif qid in prior:
            answered.append(prior[qid])

    grounded_count = sum(1 for aq in answered if aq.get("is_grounded"))
    return {"answered_questions": answered, "grounded_question_count": grounded_count}


if __name__ == "__main__":
    import json

    index = {
        "grounded_facts": [
            {
                "fact": "North America 41%, Europe 33%, Asia Pacific 18%, Rest of World 8% of organic milk sales.",
                "origin": "stat", "source_ref": "https://example.gov/dairy",
                "source_market": "Organic Milk", "is_direct_match": True, "relevance": "high",
                "grounding_label": "", "tab": "consumer_profile", "topic": "regional market share",
                "figure": {"measure": "regional share", "number": 41.0, "unit": "%",
                           "period": "2025", "verbatim": "41%"},
                "source_date": "2025-11-01", "corroborated_by": [],
            },
            {
                "fact": "Quality is the top purchase factor for 78% of organic dairy buyers (sheep milk study).",
                "origin": "cmi_database", "source_ref": "report:9589",
                "source_market": "Sheep Milk", "is_direct_match": False, "relevance": "low",
                "grounding_label": "based on related Sheep Milk market data",
                "tab": "buying_behavior", "topic": "purchase factors",
                "figure": {"measure": "quality importance", "number": 78.0, "unit": "%",
                           "period": None, "verbatim": "78%"},
                "source_date": None, "corroborated_by": [],
            },
        ],
        "context_facts": [],
        "by_tab": {"consumer_profile": [0], "buying_behavior": [1]},
        "grounded_count": 2, "direct_match_count": 1, "adjacent_count": 1,
    }
    questions = [
        {
            "id": "cp1", "tab": "consumer_profile",
            "text": "In which region do you currently live?", "type": "single_choice",
            "options": ["North America", "Europe", "Asia Pacific", "Rest of World"],
            "chart_type": "donut", "serves_topic": "regional market share", "evidence_aligned": True,
        },
        {
            "id": "bb5", "tab": "buying_behavior",
            "text": "Which factors influence your organic milk purchase? (Select all that apply)",
            "type": "multiple_choice",
            "options": ["Quality", "Price", "Brand", "Reviews", "Packaging"],
            "chart_type": "horizontal_bar", "serves_topic": "purchase factors", "evidence_aligned": True,
        },
    ]
    state = {
        "questions": questions, "evidence_index": index,
        "normalized_segment": "Organic Milk", "segment_type": "b2c",
        "region": "Global", "grounding_thin": False,
    }
    out = asyncio.run(answer_estimator(state))
    print(f"grounded_question_count: {out['grounded_question_count']}\n")
    for aq in out["answered_questions"]:
        total = round(sum(a["percentage"] for a in aq["answers"]), 1)
        print(f"[{aq['id']}] type={aq['type']} sum={total} is_grounded={aq['is_grounded']} conf={aq['question_confidence']}")
        print(f"   note: {aq['distribution_note']}")
        for a in aq["answers"]:
            print(f"     {a['label']:<16} {a['percentage']:>5}%  grounded_in={a['grounded_in']} "
                  f"market={a['source_market']} label={a['grounding_label']!r} conf={a['confidence']}")
        print()
