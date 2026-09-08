"""Node 3 — evidence_indexer.

Turns the raw ``evidence_pool`` (DB + web facts, many from ADJACENT markets due
to Node 2's wide token matching) into a structured, relevance-scored, labeled
grounding lookup table — the "grounding currency" Node 5 uses to attach real
percentages to answers.

Grounding policy: PERMISSIVE-WITH-LABELING
------------------------------------------
No numeric fact is discarded. Adjacent-market facts are KEPT and grounding-
eligible, but each carries honest provenance so a related-market number can be
surfaced as e.g. "based on related Sheep Milk market data" — never masquerading
as the target segment's own figure. The integrity guarantee is the LABEL: an
adjacent number is fine; an unlabeled or mislabeled one is not. Unsure →
is_direct_match=False (so it gets labeled rather than passing as direct).

Authoritative weighting: stat > cmi_database > exa/web > social voice for numeric
claims; reddit/youtube/twitter are high-value for customer_voice/complaints
context but must not ground percentages unless corroborated. When sources agree
on a figure, that corroboration is the strongest grounding.
"""

from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel

from src.date_utils import current_date_context
from src.evidence_utils import VOICE_ONLY_ORIGINS
from src.llm import get_structured_llm
from src.models import EvidenceIndex, IndexedFact
from src.state import SurveyState

# Facts per LLM call. Smaller batches + parallel calls keep DeepSeek responsive
# and avoid 8k-output truncation; deterministic (ordered) chunks → stable reruns.
BATCH_SIZE = 15
MAX_PARALLEL_BATCHES = 6

_ORIGIN_RANK = {
    "stat": 4, "cmi_database": 3, "exa": 2, "web": 2,
    "reddit": 1, "youtube": 1, "twitter": 1,  # change for b2c questionarie
}
_RELEVANCE_RANK = {"high": 3, "medium": 2, "low": 1}
_VOICE_TOPICS = frozenset({
    "customer_voice", "customer_language", "complaints", "consumer voice (complaints/questions)",
})

VALID_TABS = (
    "consumer_profile",
    "buying_behavior",
    "preferences_expectations",
    "satisfaction_future_intent",
    "context",
)


class _IndexBatch(BaseModel):
    """Structured-output wrapper: one IndexedFact per input fact in the batch."""

    facts: list[IndexedFact]


PROMPT_TEMPLATE = """\
You are a meticulous research analyst building a LABELED grounding index for a
B2C consumer survey on "{segment}" (region: {region}). Today's date is {today}.

For EACH input fact below, produce one IndexedFact:
- source_market: the market the fact ACTUALLY describes (the target segment
  itself, or a related/adjacent market like "Sheep Milk", "Organic Oat Milk").
- is_direct_match: True ONLY if the fact is about "{segment}" itself. If unsure,
  set False (so adjacent numbers get labeled rather than passing as direct).
- relevance: high (the target segment, or a very close substitute), medium (a
  related category), low (a far/adjacent market).
- grounding_label: "" when is_direct_match is True; otherwise exactly
  "based on related {{source_market}} market data".
- tab: one of consumer_profile | buying_behavior | preferences_expectations |
  satisfaction_future_intent | context. Use "context" for background facts that
  don't map to a survey tab.
- topic: a clean, specific label (e.g. "regional market share", "purchase
  frequency", "price tier", "market size", "growth rate").
- figure: if the fact carries a number, extract measure / number (float) /
  unit ("%", "USD_mn", "USD", "CAGR_%", ...) / period (year or range, if any) /
  verbatim (the ORIGINAL figure string EXACTLY as written — e.g. "46.8%",
  "USD 7.4 Mn"; never round or rephrase). If the fact states several figures
  (e.g. multiple regional shares), pick the single most important one for this
  fact. If the fact has no number, set figure to null.
- Copy fact / origin / source_ref / source_date through unchanged.
KEEP ALL numeric facts — direct AND adjacent. Prefer recent + authoritative
(stat) sources. Return the facts in the SAME ORDER as given.

Examples:
1. Fact "Sheep milk cheese accounts for 46.8% share of the sheep milk market"
   (survey is on Organic Milk) → source_market="Sheep Milk",
   is_direct_match=False, relevance="low",
   grounding_label="based on related Sheep Milk market data",
   tab="buying_behavior", topic="segment market share",
   figure={{measure:"sheep milk cheese share", number:46.8, unit:"%",
   period:null, verbatim:"46.8%"}}. KEPT, labeled.
2. Fact "Organic milk holds 41% of dairy sales in North America"
   → source_market="Organic Milk", is_direct_match=True, relevance="high",
   grounding_label="", tab="consumer_profile", topic="regional market share",
   figure={{measure:"organic milk share North America", number:41, unit:"%",
   period:null, verbatim:"41%"}}.

Input facts (JSON):
{facts_json}
"""


def _llm():
    return get_structured_llm(_IndexBatch, temperature=0)


def _compact(raw: dict) -> dict:
    """Trim a raw pool fact to the fields the LLM needs to judge it."""
    return {
        "fact": (raw.get("fact") or "")[:1200],
        "origin": raw.get("origin", "web"),
        "source_ref": raw.get("source_ref", ""),
        "value": raw.get("value"),
        "source_date": raw.get("source_date"),
    }


def _fallback_fact(raw: dict, segment: str) -> dict:
    """Minimal IndexedFact dict if an LLM batch fails — preserve, don't drop."""
    value = raw.get("value")
    figure = None
    if raw.get("is_numeric") and value:
        figure = {
            "measure": raw.get("topic") or "figure",
            "number": 0.0,
            "unit": "%" if "%" in str(value) else "raw",
            "period": None,
            "verbatim": str(value),
        }
    tab = raw.get("tab_hint") if raw.get("tab_hint") in VALID_TABS else "context"
    return {
        "fact": raw.get("fact", ""),
        "origin": raw.get("origin", "web"),
        "source_ref": raw.get("source_ref", ""),
        "source_market": segment,
        "is_direct_match": False,
        "relevance": "low",
        "grounding_label": f"based on related {segment} market data",
        "tab": tab or "context",
        "topic": raw.get("topic") or "general",
        "figure": figure,
        "source_date": raw.get("source_date"),
        "corroborated_by": [],
    }


def _index_batch(batch: list, segment: str, region: str, today: str) -> list:
    """Run one LLM batch; on failure, fall back to minimal preservation."""
    import json

    prompt = PROMPT_TEMPLATE.format(
        segment=segment,
        region=region,
        today=today,
        facts_json=json.dumps([_compact(f) for f in batch], ensure_ascii=False),
    )
    try:
        result: _IndexBatch = _llm().invoke(prompt)
        indexed = [f.model_dump() for f in result.facts]
    except Exception:  # noqa: BLE001 — keep the run alive, preserve the facts
        return [_fallback_fact(f, segment) for f in batch]

    # Provenance integrity: when counts align, force our own source data over
    # anything the LLM may have altered while "copying through".
    if len(indexed) == len(batch):
        for out, raw in zip(indexed, batch):
            out["fact"] = raw.get("fact", out.get("fact", ""))
            out["origin"] = raw.get("origin", out.get("origin", "web"))
            out["source_ref"] = raw.get("source_ref", out.get("source_ref", ""))
            out["source_date"] = raw.get("source_date", out.get("source_date"))
    return indexed


def _merge_corroborate(grounded: list) -> list:
    """Light cross-batch pass: collapse facts stating the same figure for the
    same tab+topic into the most authoritative one, recording corroboration."""
    import re

    groups = {}
    for f in grounded:
        fig = f.get("figure") or {}
        norm = re.sub(r"[^0-9.]", "", str(fig.get("verbatim", "")))
        key = (f.get("tab"), f.get("topic"), norm)
        groups.setdefault(key, []).append(f)

    merged = []
    for (_, _, norm), items in groups.items():
        if len(items) == 1 or not norm:
            merged.extend(items)
            continue
        items.sort(
            key=lambda f: (
                _ORIGIN_RANK.get(f.get("origin"), 0),
                f.get("source_date") or "",
            ),
            reverse=True,
        )
        best = items[0]
        others = [it["source_ref"] for it in items[1:] if it.get("source_ref")]
        best["corroborated_by"] = list(
            dict.fromkeys((best.get("corroborated_by") or []) + others)
        )
        merged.append(best)
    return merged


def _context_sort_key(f: dict):
    """Social/customer-voice context ranks high for personalization."""
    topic = (f.get("topic") or "").lower()
    origin = f.get("origin") or ""
    # change for b2c questionarie
    voice_boost = 2 if origin in VOICE_ONLY_ORIGINS or topic in _VOICE_TOPICS else 0
    return (voice_boost, _RELEVANCE_RANK.get(f.get("relevance"), 0))


def _sort_key(f: dict):
    origin = f.get("origin") or ""
    # Social numeric anecdotes rank below web/exa unless corroborated.
    # change for b2c questionarie
    origin_penalty = -1 if origin in VOICE_ONLY_ORIGINS and f.get("figure") else 0
    return (
        1 if f.get("is_direct_match") else 0,
        _ORIGIN_RANK.get(origin, 0) + origin_penalty,
        1 if f.get("corroborated_by") else 0,
        _RELEVANCE_RANK.get(f.get("relevance"), 0),
        f.get("source_date") or "",
    )


def _empty_index() -> dict:
    return {
        "grounded_facts": [],
        "context_facts": [],
        "by_tab": {},
        "grounded_count": 0,
        "direct_match_count": 0,
        "adjacent_count": 0,
    }


def evidence_indexer(state: SurveyState) -> dict:
    """Build the labeled, relevance-scored grounding index (Node 3)."""
    pool = state.get("evidence_pool") or []
    segment = state.get("normalized_segment") or state.get("market_segment") or "the segment"
    region = state.get("region") or "Global"

    # 1. EMPTY GUARD — nothing to ground.
    if state.get("evidence_empty") or not pool:
        return {"evidence_index": _empty_index(), "grounding_thin": True}

    today = current_date_context()["iso"]

    # 2-5. Relevance/provenance/tagging/figure extraction, batched IN PARALLEL
    # (each _index_batch is a blocking LLM call; threads overlap their latency).
    batches = [pool[i:i + BATCH_SIZE] for i in range(0, len(pool), BATCH_SIZE)]
    all_indexed = []
    with ThreadPoolExecutor(max_workers=min(len(batches), MAX_PARALLEL_BATCHES)) as ex:
        futures = [ex.submit(_index_batch, b, segment, region, today) for b in batches]
        for fut in futures:
            all_indexed.extend(fut.result())

    # Split numeric (grounding currency) vs non-numeric (context).
    grounded = [f for f in all_indexed if f.get("figure")]
    context = [f for f in all_indexed if not f.get("figure")]

    # 6. Dedup/merge + corroboration, then 7. sort best-first.
    grounded = _merge_corroborate(grounded)
    grounded.sort(key=_sort_key, reverse=True)
    context.sort(key=_context_sort_key, reverse=True)

    # by_tab: tab -> indices into grounded_facts (Node 5's lookup).
    by_tab = {}
    for i, f in enumerate(grounded):
        tab = f.get("tab") if f.get("tab") in VALID_TABS else "context"
        by_tab.setdefault(tab, []).append(i)

    direct = sum(1 for f in grounded if f.get("is_direct_match"))
    index = {
        "grounded_facts": grounded,
        "context_facts": context,
        "by_tab": by_tab,
        "grounded_count": len(grounded),
        "direct_match_count": direct,
        "adjacent_count": len(grounded) - direct,
    }
    # Validate shape against the model (raises early if drift creeps in).
    index = EvidenceIndex(**index).model_dump()

    return {"evidence_index": index, "grounding_thin": len(grounded) == 0}


if __name__ == "__main__":
    import json

    demo_pool = [
        {
            "fact": "Organic milk holds about 41% of premium dairy sales in North America in 2025.",
            "value": "41%", "is_numeric": True, "topic": "regional share",
            "tab_hint": "consumer_profile", "origin": "stat",
            "source_ref": "https://example.gov/dairy", "source_date": "2025-11-01",
        },
        {
            "fact": "Sheep milk cheese accounts for the largest share, 46.8%, of the global sheep milk market.",
            "value": "46.8%", "is_numeric": True, "topic": "segment share",
            "tab_hint": "buying_behavior", "origin": "cmi_database",
            "source_ref": "report:9589", "source_date": None,
        },
        {
            "fact": "Health-conscious households are the primary buyers of organic dairy products.",
            "value": None, "is_numeric": False, "topic": "demographics",
            "tab_hint": "consumer_profile", "origin": "web",
            "source_ref": "https://example.com/organic", "source_date": None,
        },
    ]
    state = {
        "evidence_pool": demo_pool,
        "normalized_segment": "Organic Milk",
        "segment_type": "b2c",
        "region": "Global",
        "evidence_empty": False,
    }
    out = evidence_indexer(state)
    idx = out["evidence_index"]
    print(f"grounding_thin:     {out['grounding_thin']}")
    print(f"grounded_count:     {idx['grounded_count']}")
    print(f"direct_match_count: {idx['direct_match_count']}")
    print(f"adjacent_count:     {idx['adjacent_count']}")
    print(f"by_tab:             {idx['by_tab']}")
    print("\nGrounded facts (provenance + verbatim figure):")
    for f in idx["grounded_facts"]:
        fig = f.get("figure") or {}
        print(f"  - source_market={f['source_market']!r} direct={f['is_direct_match']} "
              f"relevance={f['relevance']} origin={f['origin']}")
        print(f"    verbatim={fig.get('verbatim')!r}  label={f['grounding_label']!r}")
    print("\nContext facts:", [c["fact"][:60] for c in idx["context_facts"]])
