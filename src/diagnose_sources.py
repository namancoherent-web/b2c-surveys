"""Pre-flight diagnostics for all web evidence sources.

Run before a full pipeline to see which sources are live:

    python -m src.diagnose_sources
"""

import json
import os
import shutil

from src import config
from src.fetchers.jina_reader import fetch_clean
from src.searchers.ddg_pool import DDGWorkerPool
from src.searchers.exa_search import search_exa
from src.searchers.reddit_search import search_reddit
from src.searchers.twitter_search import search_twitter
from src.searchers.youtube_search import search_youtube
from src.web_search import search_web


def _status(ok: bool, detail: str) -> dict:
    return {"ok": ok, "detail": detail}


def check_jina() -> dict:
    url = "https://en.wikipedia.org/wiki/Washing_machine"
    text = fetch_clean(url, timeout=20)
    if text and len(text) > 500:
        return _status(True, f"r.jina.ai returned {len(text)} chars for Wikipedia")
    return _status(False, "r.jina.ai returned no/short content")


def check_exa() -> dict:
    if not config.EXA_API_KEY:
        return _status(False, "EXA_API_KEY not set")
    results = search_exa("washing machine consumer market share", num_results=3)
    if results:
        return _status(True, f"Exa returned {len(results)} results")
    return _status(False, "Exa returned 0 results (check key/quota)")


def check_youtube() -> dict:
    # change for b2c questionarie
    if not config.YOUTUBE_ENABLED:
        return _status(False, "YOUTUBE_ENABLED=false")
    if not (config.YOUTUBE_API_KEY or "").strip():
        return _status(False, "YOUTUBE_API_KEY not set")
    rows = search_youtube("washing machine complaints", limit=5, max_videos=2)
    if rows:
        return _status(True, f"YouTube returned {len(rows)} comments")
    return _status(False, "YouTube returned 0 comments (check key/quota/comments disabled)")


def check_twitter() -> dict:
    # change for b2c questionarie — default off
    if not config.TWITTER_ENABLED:
        return _status(False, "TWITTER_ENABLED=false (default — leave off unless paid API)")
    if not (config.TWITTER_BEARER_TOKEN or "").strip():
        return _status(False, "TWITTER_ENABLED but TWITTER_BEARER_TOKEN missing")
    hits = search_twitter("washing machine complaints", limit=5)
    if hits:
        return _status(True, f"Twitter/X returned {len(hits)} tweets")
    return _status(False, "Twitter/X returned 0 tweets")


def check_reddit() -> dict:
    from src.searchers.reddit_auth import (
        find_rdt_binary,
        load_reddit_cookies,
        oauth_configured,
    )

    if not config.REDDIT_ENABLED:
        return _status(False, "REDDIT_ENABLED=false")
    if oauth_configured():
        posts = search_reddit("washing machine complaints", limit=5, timeout=45)
        if posts:
            return _status(True, f"Reddit OAuth returned {len(posts)} posts")
        return _status(False, "Reddit OAuth configured but search returned 0")
    cookies = load_reddit_cookies()
    if not cookies:
        return _status(
            False,
            "No Reddit OAuth (REDDIT_CLIENT_ID/SECRET) and no cookies. "
            "Set OAuth vars or run scripts/setup_reddit_cookies.py",
        )
    posts = search_reddit("washing machine complaints", limit=5, timeout=45)
    if posts:
        return _status(True, f"Reddit returned {len(posts)} posts (rdt={bool(find_rdt_binary())})")
    return _status(False, "Cookies present but search returned 0 — reddit_session may be expired")


def check_ddg() -> dict:
    pool = DDGWorkerPool(workers=2)
    facts = pool.search_to_findings("organic milk market statistics", max_results=3)
    if facts:
        return _status(True, f"DDG pool returned {len(facts)} findings")
    return _status(False, "DDG pool returned 0 findings")


def check_orchestrator() -> dict:
    results = search_web([
        {"query": "skincare consumer survey", "angle": "trends"},
        {"query": "skincare complaints", "angle": "complaints"},
    ], max_results_per_query=3)
    by_origin = {}
    for f in results:
        by_origin[f["origin"]] = by_origin.get(f["origin"], 0) + 1
    if results:
        return _status(True, f"orchestrator merged {len(results)} facts: {by_origin}")
    return _status(False, "orchestrator returned 0 facts")


def main() -> None:
    checks = {
        "jina_reader": check_jina(),
        "exa_search": check_exa(),
        "youtube_search": check_youtube(),
        "reddit_search": check_reddit(),
        "twitter_search": check_twitter(),
        "ddg_pool": check_ddg(),
        "web_search_orchestrator": check_orchestrator(),
    }
    print(json.dumps(checks, indent=2))
    live = sum(1 for c in checks.values() if c["ok"])
    print(f"\n{live}/{len(checks)} sources live")
    if live < 2:
        print("WARNING: fewer than 2 sources live — grounding may be thin.")


if __name__ == "__main__":
    main()
