"""Twitter/X consumer-voice search (OPTIONAL — OFF by default).

# change for b2c questionarie — Phase A4

The official X API is costly and gated. Scraping is fragile and should go
through the VPN-rotation path if ever enabled. Keep TWITTER_ENABLED=false
unless you have a paid bearer token and accept the cost.

When disabled (default), search_twitter returns [] immediately.
When enabled + TWITTER_BEARER_TOKEN set, uses X API v2 recent search.
Facts are language-only (origin="twitter", never numeric).
"""

from __future__ import annotations

import logging
import os

import requests

from src import config
from src.date_utils import current_date_context
from src.evidence_utils import build_voice_fact, infer_tab_hint
from src.ratelimit import acquire

logger = logging.getLogger(__name__)

_X_RECENT_SEARCH = "https://api.twitter.com/2/tweets/search/recent"
_WEB_RATE = float(os.getenv("WEB_RATE_PER_SEC", "1"))
_WEB_BURST = float(os.getenv("WEB_BURST", "3"))

_COMPLAINT_MARKERS = (
    "hate", "wish", "disappoint", "frustrat", "annoying", "too expensive",
    "overpriced", "worst", "refund", "not worth", "waste", "broke", "issue",
    "problem", "complaint", "avoid",
)


def _twitter_ready() -> bool:
    """Enabled only when flag is on AND a bearer token is present."""
    if not config.TWITTER_ENABLED:
        return False
    if not (config.TWITTER_BEARER_TOKEN or "").strip():
        logger.debug("TWITTER_ENABLED but TWITTER_BEARER_TOKEN missing — skipping")
        return False
    return True


def _infer_voice_topic(text: str) -> str:
    low = (text or "").lower()
    if any(m in low for m in _COMPLAINT_MARKERS):
        return "complaints"
    if "?" in (text or ""):
        return "customer_voice"
    return "customer_language"


def search_twitter(query: str, limit: int = 15) -> list[dict]:
    """Return [{id, text, created_at, url}] or [] when disabled / failed."""
    if not _twitter_ready():
        return []

    acquire("web", _WEB_RATE, _WEB_BURST)
    headers = {"Authorization": f"Bearer {config.TWITTER_BEARER_TOKEN}"}
    params = {
        "query": f"{query} -is:retweet lang:en",
        "max_results": max(10, min(limit, 50)),
        "tweet.fields": "created_at,lang",
    }
    try:
        resp = requests.get(_X_RECENT_SEARCH, headers=headers, params=params, timeout=20)
        if resp.status_code in (401, 403):
            logger.warning("Twitter/X auth error (%s): %s", resp.status_code, resp.text[:200])
            return []
        if resp.status_code == 429:
            logger.warning("Twitter/X rate limited")
            return []
        if resp.status_code == 402:
            logger.warning("Twitter/X payment required — keep TWITTER_ENABLED=false")
            return []
        resp.raise_for_status()
        data = resp.json().get("data") or []
    except Exception as exc:  # noqa: BLE001
        logger.warning("Twitter/X search failed: %s", exc)
        return []

    out = []
    for tw in data[:limit]:
        tid = tw.get("id") or ""
        text = (tw.get("text") or "").strip()
        if not tid or len(text) < 20:
            continue
        out.append({
            "id": tid,
            "text": text,
            "created_at": (tw.get("created_at") or "")[:10] or None,
            "url": f"https://x.com/i/web/status/{tid}",
        })
    return out


def twitter_to_findings(tweets: list[dict], serves_goal: str, angle: str) -> list[dict]:
    """Convert tweets into language-only unified findings."""
    retrieved_at = current_date_context()["iso"]
    findings = []
    for tw in tweets or []:
        text = (tw.get("text") or "").strip()
        url = tw.get("url") or ""
        if len(text) < 20 or not url:
            continue
        f = build_voice_fact(
            fact=text[:3000],
            origin="twitter",
            source_ref=url,
            title="twitter post",
            topic=_infer_voice_topic(text),
            tab_hint=infer_tab_hint(f"{serves_goal} {angle} {text}"),
        )
        if f:
            f["source_date"] = tw.get("created_at")
            f["retrieved_at"] = retrieved_at
            findings.append(f)
    return findings


if __name__ == "__main__":
    import json
    print(json.dumps({
        "enabled": config.TWITTER_ENABLED,
        "ready": _twitter_ready(),
        "hits": len(search_twitter("air fryer complaints", limit=5)),
    }, indent=2))
