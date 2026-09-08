"""Node 2 — evidence_harvester.

Gathers AS MUCH real, citable evidence (especially numbers/percentages) as it
can into the ``evidence_pool``, so later nodes can ground answer distributions
in fact rather than pure inference.

Two sources run CONCURRENTLY (``asyncio.gather(..., return_exceptions=True)``):

    SOURCE A — CMI database (Neon Postgres, sync psycopg2 wrapped in
        ``asyncio.to_thread``): match reports to the segment, then unpivot each
        report's cmi_reportsummery heading/content pairs and pull its
        cmi_report_dynamic field/description pairs, extracting figure-bearing
        facts.

    SOURCE B — Web (Exa + Reddit + DuckDuckGo via ``web_search.search_web``): run a
        small set of consumer-focused queries; each result becomes a candidate fact.

Either source failing must NOT crash the node — a failed source is treated as an
empty list and the other proceeds. DB connections are always closed in finally.

The count of ``is_numeric`` facts is the early signal of how much real grounding
downstream nodes will actually have to work with.
"""

import asyncio

from src import db, web_search
from src.date_utils import current_date_context
from src.evidence_utils import (
    build_fact,
    cap_facts,
    dedupe_facts,
    infer_tab_hint,
    infer_topic,
)
from src.search_relevance import product_phrase_for_query
from src.state import SurveyState

# Pull a wide candidate set by token match, then mine the richest few.
CANDIDATE_LIMIT = 25
# Up to this many populated reports are mined (maximize recall; Node 3 trims).
MAX_REPORTS = 8


def _build_web_queries(segment: str, region: str, year: int) -> list:
    """Consumer-focused web queries with angle tags for source routing.

    Multi-word products are phrase-quoted (``\"air fryer\"``) so DDG/Exa do not
    match unrelated hits on a single word (e.g. Air India for air fryer).

    # change for b2c questionarie — geo terms on ALL angles for region jobs (A4b)
    """
    geo = "" if region.lower() == "global" else f" {region}"
    product = product_phrase_for_query(segment)
    return [
        {"query": f"{product} consumer survey statistics{geo} {year}", "angle": "trends",
         "serves_goal": "consumer statistics"},
        {"query": f"{product} consumer purchase frequency and preferences{geo} {year}",
         "angle": "needs", "serves_goal": "purchase behavior"},
        {"query": f"{product} market share regional breakdown{geo} {year}",
         "angle": "market_share", "serves_goal": "market share"},
        {"query": f"{product} consumer demographics buyers{geo} {year}",
         "angle": "trends", "serves_goal": "demographics"},
        {"query": f"{product} average price and willingness to pay{geo} {year}",
         "angle": "pricing", "serves_goal": "pricing"},
        {"query": f"{product} customer complaints problems reviews{geo}",
         "angle": "complaints", "serves_goal": "consumer voice"},
        {"query": f"{product} shopping channels brands{geo}",
         "angle": "customer_language", "serves_goal": "local channels and brands"},
    ]


def _harvest_database(segment: str) -> list:
    """SOURCE A (sync, run via asyncio.to_thread). Returns unified fact dicts.

    Opens one connection, mines up to MAX_REPORTS matched reports, and always
    closes the connection in ``finally``. Per-report failures are swallowed so
    one bad report doesn't sink the whole harvest.
    """
    conn = None
    facts = []
    try:
        conn = db.get_connection()
        # Wide token-matched candidate set, ranked by token overlap...
        candidates = db.find_reports(segment, conn, limit=CANDIDATE_LIMIT)
        # ...then prefer reports that actually have enrichment data, so one
        # empty ancient report can't make a segment look ungrounded. Preserve
        # find_reports rank order within each group.
        with_data = db.reports_with_data([c[0] for c in candidates], conn)
        prioritized = (
            [c for c in candidates if c[0] in with_data]
            + [c for c in candidates if c[0] not in with_data]
        )[:MAX_REPORTS]
        for report_id, title, _category in prioritized:
            try:
                for s in db.get_report_summary_facts(report_id, conn):
                    section = s.get("section", "summary")
                    f = build_fact(
                        fact=s["fact"],
                        origin="cmi_database",
                        source_ref=f"report:{report_id}",
                        title=f"{title} — {section}",
                        topic=infer_topic(f"{section} {s['fact']}"),
                        tab_hint=infer_tab_hint(f"{section} {s['fact']}"),
                        value=s.get("value"),
                    )
                    if f:
                        facts.append(f)
                for d in db.get_report_dynamic_facts(report_id, conn):
                    field = d.get("field", "detail")
                    f = build_fact(
                        fact=d["fact"],
                        origin="cmi_database",
                        source_ref=f"report:{report_id}",
                        title=f"{title} — {field}",
                        topic=infer_topic(f"{field} {d['fact']}"),
                        tab_hint=infer_tab_hint(f"{field} {d['fact']}"),
                        value=d.get("value"),
                    )
                    if f:
                        facts.append(f)
            except Exception:  # noqa: BLE001 — skip a bad report, keep harvesting
                continue
    finally:
        if conn is not None:
            conn.close()
    return facts


async def _harvest_web(queries: list, segment: str) -> list:
    """SOURCE B. Runs the blocking multi-source web search in a worker thread."""
    return await asyncio.to_thread(web_search.search_web, queries, 6, segment)


async def evidence_harvester(state: SurveyState) -> dict:
    """Harvest evidence from the CMI database + web in parallel (Node 2)."""
    segment = state.get("normalized_segment") or state.get("market_segment")
    if not segment:
        raise ValueError(
            "evidence_harvester requires 'normalized_segment' (or 'market_segment')."
        )
    region = state.get("region") or "Global"
    year = current_date_context()["year"]

    queries = _build_web_queries(segment, region, year)

    db_result, web_result = await asyncio.gather(
        asyncio.to_thread(_harvest_database, segment),
        _harvest_web(queries, segment),
        return_exceptions=True,
    )

    if isinstance(db_result, Exception):
        print(f"[evidence_harvester] DB source failed: {db_result}")
        db_facts = []
    else:
        db_facts = db_result if isinstance(db_result, list) else []

    if isinstance(web_result, Exception):
        print(f"[evidence_harvester] Web source failed: {web_result}")
        web_facts = []
    else:
        web_facts = web_result if isinstance(web_result, list) else []

    combined = cap_facts(dedupe_facts(db_facts + web_facts))
    # change for b2c questionarie — stamp region on every harvested fact (A4b)
    for f in combined:
        if not f.get("region"):
            f["region"] = region
    if not combined:
        print(
            "[evidence_harvester] 0 facts — check EXA_API_KEY, DDG reachability, "
            "and DB token matches for this segment."
        )

    out = {"evidence_pool": combined}
    if not combined:
        out["evidence_empty"] = True
    return out


if __name__ == "__main__":
    import json

    # Hardcoded Node-1 output for a B2C segment, so this runs standalone.
    demo_state = {
        "normalized_segment": "Organic Milk",
        "market_segment": "global organic milk market",
        "segment_type": "b2c",
        "region": "Global",
        "classification_reason": "Bought by individual shoppers for household use.",
        "audience_profile": {
            "demographics": {"age": "25-55"},
            "behaviors": ["buys groceries weekly"],
            "qualifiers": ["purchases milk for the household"],
            "exclusions": [],
            "rationale": "Routine grocery purchase.",
            "assumptions": [],
        },
    }

    try:
        result = asyncio.run(evidence_harvester(demo_state))
    except Exception as exc:  # noqa: BLE001 — standalone smoke test must degrade gracefully
        print(f"[evidence_harvester] run failed: {exc}")
        raise SystemExit(0)

    pool = result.get("evidence_pool", [])
    by_origin = {}
    for f in pool:
        by_origin[f["origin"]] = by_origin.get(f["origin"], 0) + 1
    numeric_count = sum(1 for f in pool if f.get("is_numeric"))

    print(f"\nTotal facts: {len(pool)}")
    print(f"By origin:   {by_origin}")
    print(f"Numeric facts (is_numeric=True): {numeric_count}")
    print(f"evidence_empty: {result.get('evidence_empty', False)}")
    print("\nFirst ~10 facts:")
    for f in pool[:10]:
        print(json.dumps(f, indent=2))
