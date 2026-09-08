"""Web + statistical source layer for the evidence harvester (Node 2, SOURCE B).

Orchestrates free evidence sources by query *angle*, with DuckDuckGo as the
reliability fallback:

  - **Exa** — semantic search (primary for trends/pricing/competitors/market_share)
  - **YouTube / Reddit / Twitter** — consumer voice (language only; never %)
  - **DDGWorkerPool** — keyword search fallback for every angle

# change for b2c questionarie — YouTube + Twitter wired under voice angles

All sources return the same unified evidence-fact dicts the DB side produces.
Current year is injected at runtime into time-sensitive queries; numeric-sentence
extraction, STAT_DOMAIN tagging, dedup, and graceful degradation are preserved.
"""

import asyncio
import logging
import re

from src.date_utils import current_date_context
from src.evidence_utils import dedupe_facts
from src.searchers.ddg_pool import DDGWorkerPool
from src.searchers.exa_search import exa_to_findings, search_exa
from src.searchers.reddit_search import reddit_to_findings, search_reddit
from src.searchers.twitter_search import search_twitter, twitter_to_findings
from src.searchers.youtube_search import search_youtube, youtube_to_findings

logger = logging.getLogger(__name__)

WEB_FACT_CEILING = 50
MAX_RESULTS_PER_QUERY = 6

_STAT_ANGLES = frozenset({"trends", "pricing", "competitors", "market_share"})
_VOICE_ANGLES = frozenset({"complaints", "needs", "customer_language", "reviews"})

_ddg_pool = DDGWorkerPool(workers=4)


def _normalize_queries(queries) -> list[dict]:
    """Accept Node-2 query objects or plain strings; return dicts with angle."""
    out = []
    for item in queries or []:
        if isinstance(item, str):
            s = item.strip()
            if s:
                out.append({"query": s, "angle": "default", "serves_goal": s})
        elif isinstance(item, dict):
            q = (item.get("query") or item.get("topic") or item.get("angle") or "").strip()
            if q:
                out.append({
                    "query": q,
                    "angle": (item.get("angle") or "default").strip().lower(),
                    "serves_goal": item.get("serves_goal") or q,
                })
        else:
            s = str(item).strip()
            if s:
                out.append({"query": s, "angle": "default", "serves_goal": s})
    return out


def _expand_queries(queries: list[dict], year: int) -> list[dict]:
    """Stamp year on general queries; add statistical variant where useful."""
    out, seen = [], set()
    for item in queries:
        q = item["query"]
        has_year = bool(re.search(r"\b(19|20)\d{2}\b", q))
        general = q if has_year else f"{q} {year}"
        variants = [general]
        low = q.lower()
        if "statistic" not in low and "survey" not in low and item["angle"] in _STAT_ANGLES | {"default"}:
            variants.append(f"{q} consumer survey statistics {year}")
        for v in variants:
            key = v.lower()
            if key not in seen:
                seen.add(key)
                out.append({**item, "query": v})
    return out


def _cap_web_facts(facts: list, ceiling: int = WEB_FACT_CEILING) -> list:
    if len(facts) <= ceiling:
        return facts

    def _key(f):
        return (
            1 if f.get("is_numeric") else 0,
            1 if f.get("source_date") else 0,
            f.get("source_date") or "",
        )

    return sorted(facts, key=_key, reverse=True)[:ceiling]


def _dispatch_query(item: dict, max_results: int, segment: str | None = None) -> list[dict]:
    """Run one query through the angle-appropriate source chain."""
    query = item["query"]
    angle = item["angle"]
    serves_goal = item.get("serves_goal") or query
    findings: list[dict] = []
    seg = segment or ""

    def _exa() -> None:
        hits = search_exa(query, num_results=max_results)
        findings.extend(exa_to_findings(hits, serves_goal, angle, segment=seg))

    def _ddg() -> None:
        from src import config as _cfg
        if not getattr(_cfg, "DDG_ENABLED", True):
            return
        findings.extend(
            _ddg_pool.search_to_findings(
                query, max_results, serves_goal, angle, segment=seg
            )
        )

    if angle in _VOICE_ANGLES:
        # change for b2c questionarie — YouTube + Reddit + optional Twitter
        yt_rows = search_youtube(query, limit=max_results + 4)
        findings.extend(youtube_to_findings(yt_rows, serves_goal, angle))
        reddit_posts = search_reddit(query, limit=max_results + 5)
        findings.extend(reddit_to_findings(reddit_posts, serves_goal, angle))
        tw_rows = search_twitter(query, limit=max_results + 5)
        findings.extend(twitter_to_findings(tw_rows, serves_goal, angle))
        if len(findings) < 3:
            _exa()
        if len(findings) < 2:
            _ddg()
    else:
        # Stats / default: Exa primary when keyed; DDG (lite) is the workhorse.
        _exa()
        if len(findings) < 2:
            _ddg()

    return findings


def search_web(queries, max_results_per_query: int = 6, segment: str = None) -> list:
    """Fan out across Exa / YouTube / Reddit / Twitter / DDG by angle.

    Args:
        queries: strings or dicts with ``query``, ``angle``, ``serves_goal``.
        max_results_per_query: results requested per source call.
        segment: market segment label (used for voice-angle query enrichment
            and off-topic hit filtering).

    Returns:
        Unified evidence-fact dicts. Empty when all sources fail — harvester
        treats that gracefully.
    """
    ctx = current_date_context()
    year = ctx["year"]

    normalized = _normalize_queries(queries)
    if segment and normalized:
        for item in normalized:
            if item["angle"] in _VOICE_ANGLES and segment.lower() not in item["query"].lower():
                item["query"] = f"{segment} {item['query']}"

    expanded = _expand_queries(normalized, year)
    if not expanded:
        return []

    all_facts = []
    for item in expanded:
        try:
            all_facts.extend(_dispatch_query(item, max_results_per_query, segment))
        except Exception as exc:  # noqa: BLE001
            logger.warning("web search dispatch failed for %r: %s", item.get("query"), exc)

    return _cap_web_facts(dedupe_facts(all_facts))


async def search_web_async(queries, max_results_per_query: int = 6, segment: str = None) -> list:
    """Async wrapper — runs blocking search_web in a worker thread."""
    return await asyncio.to_thread(
        search_web, queries, max_results_per_query, segment
    )


def get_ddg_pool() -> DDGWorkerPool:
    """Expose DDG pool for diagnostics."""
    return _ddg_pool


if __name__ == "__main__":
    import json

    demo_queries = [
        {"query": "organic milk consumer purchase frequency", "angle": "trends"},
        {"query": "organic milk customer complaints", "angle": "complaints"},
    ]
    results = search_web(demo_queries)

    numeric = sum(1 for f in results if f.get("is_numeric"))
    by_origin = {}
    for f in results:
        by_origin[f["origin"]] = by_origin.get(f["origin"], 0) + 1

    print(f"\nTotal web facts: {len(results)}")
    print(f"Numeric facts:   {numeric}")
    print(f"By origin:       {by_origin}")
    print("\nFirst ~8 facts:")
    for f in results[:8]:
        print(json.dumps(
            {
                "origin": f["origin"],
                "authority": f.get("authority"),
                "value": f["value"],
                "title": f["title"][:60],
                "fact": f["fact"][:120],
            },
            indent=2,
        ))
