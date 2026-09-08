"""YouTube Data API v3 — consumer-voice comments (language only).

# change for b2c questionarie — Phase A4

Uses the official YouTube Data API (YOUTUBE_API_KEY). Searches for segment-
relevant videos, then pulls top-level comments. Emits unified facts with
origin="youtube", authority="low", and NEVER carries numeric values
(synthesizer / estimator must not turn these into percentages).
"""

from __future__ import annotations

import logging
import os
import time

import requests

from src import config
from src.date_utils import current_date_context
from src.evidence_utils import build_voice_fact, infer_tab_hint
from src.ratelimit import acquire

logger = logging.getLogger(__name__)

_YT_SEARCH = "https://www.googleapis.com/youtube/v3/search"
_YT_COMMENTS = "https://www.googleapis.com/youtube/v3/commentThreads"
_WEB_RATE = float(os.getenv("WEB_RATE_PER_SEC", "1"))
_WEB_BURST = float(os.getenv("WEB_BURST", "3"))

_COMPLAINT_MARKERS = (
    "hate", "wish", "disappoint", "frustrat", "annoying", "too expensive",
    "overpriced", "worst", "refund", "not worth", "waste", "broke", "issue",
    "problem", "complaint", "avoid", "never buy",
)


def _youtube_ready() -> bool:
    return bool(config.YOUTUBE_ENABLED and (config.YOUTUBE_API_KEY or "").strip())


def _infer_voice_topic(text: str) -> str:
    low = (text or "").lower()
    if any(m in low for m in _COMPLAINT_MARKERS):
        return "complaints"
    if "?" in (text or ""):
        return "customer_voice"
    return "customer_language"


def search_youtube_videos(query: str, max_results: int = 5) -> list[dict]:
    """Return [{video_id, title, channel}] or [] on failure / disabled."""
    if not _youtube_ready():
        return []
    acquire("web", _WEB_RATE, _WEB_BURST)
    try:
        resp = requests.get(
            _YT_SEARCH,
            params={
                "part": "snippet",
                "q": query,
                "type": "video",
                "maxResults": max(1, min(max_results, 10)),
                "order": "relevance",
                "safeSearch": "moderate",
                "key": config.YOUTUBE_API_KEY,
            },
            timeout=20,
        )
        if resp.status_code in (401, 403):
            logger.warning("YouTube auth/quota error (%s): %s", resp.status_code, resp.text[:200])
            return []
        if resp.status_code == 429:
            logger.warning("YouTube rate limited")
            return []
        resp.raise_for_status()
        items = resp.json().get("items") or []
    except Exception as exc:  # noqa: BLE001
        logger.warning("YouTube search failed: %s", exc)
        return []

    out = []
    for it in items:
        vid = ((it.get("id") or {}).get("videoId")) or ""
        sn = it.get("snippet") or {}
        if not vid:
            continue
        out.append({
            "video_id": vid,
            "title": sn.get("title") or "youtube video",
            "channel": sn.get("channelTitle") or "",
            "url": f"https://www.youtube.com/watch?v={vid}",
        })
    return out


def fetch_youtube_comments(video_id: str, max_comments: int = 15) -> list[dict]:
    """Return [{text, author, published_at}] top-level comments."""
    if not _youtube_ready() or not video_id:
        return []
    acquire("web", _WEB_RATE, _WEB_BURST)
    try:
        resp = requests.get(
            _YT_COMMENTS,
            params={
                "part": "snippet",
                "videoId": video_id,
                "maxResults": max(1, min(max_comments, 50)),
                "order": "relevance",
                "textFormat": "plainText",
                "key": config.YOUTUBE_API_KEY,
            },
            timeout=20,
        )
        if resp.status_code in (401, 403):
            # Comments disabled / quota — soft fail
            logger.debug("YouTube comments unavailable for %s (%s)", video_id, resp.status_code)
            return []
        if resp.status_code == 429:
            time.sleep(1.0)
            return []
        resp.raise_for_status()
        items = resp.json().get("items") or []
    except Exception as exc:  # noqa: BLE001
        logger.debug("YouTube comments failed for %s: %s", video_id, exc)
        return []

    out = []
    for it in items:
        top = ((it.get("snippet") or {}).get("topLevelComment") or {}).get("snippet") or {}
        text = (top.get("textDisplay") or top.get("textOriginal") or "").strip()
        if len(text) < 20:
            continue
        out.append({
            "text": text,
            "author": top.get("authorDisplayName") or "",
            "published_at": (top.get("publishedAt") or "")[:10] or None,
        })
    return out


def search_youtube(query: str, limit: int = 12, max_videos: int = 4) -> list[dict]:
    """Search videos + pull comments; return normalized comment rows."""
    videos = search_youtube_videos(query, max_results=max_videos)
    rows = []
    for v in videos:
        for c in fetch_youtube_comments(v["video_id"], max_comments=max(3, limit // max(1, len(videos)))):
            rows.append({
                **c,
                "video_id": v["video_id"],
                "video_title": v["title"],
                "url": v["url"],
            })
            if len(rows) >= limit:
                return rows
    return rows


def youtube_to_findings(rows: list[dict], serves_goal: str, angle: str) -> list[dict]:
    """Convert YouTube comment rows into language-only unified findings."""
    retrieved_at = current_date_context()["iso"]
    findings = []
    for row in rows or []:
        text = (row.get("text") or "").strip()
        if len(text) < 20:
            continue
        url = row.get("url") or (
            f"https://www.youtube.com/watch?v={row.get('video_id')}" if row.get("video_id") else ""
        )
        if not url:
            continue
        title = row.get("video_title") or "youtube comment"
        topic = _infer_voice_topic(text)
        f = build_voice_fact(
            fact=text[:3000],
            origin="youtube",
            source_ref=url,
            title=f"{title} — comment",
            topic=topic,
            tab_hint=infer_tab_hint(f"{serves_goal} {angle} {text}"),
        )
        if f:
            f["source_date"] = row.get("published_at")
            f["retrieved_at"] = retrieved_at
            findings.append(f)
    return findings


if __name__ == "__main__":
    import json
    rows = search_youtube("air fryer complaints", limit=5)
    print(json.dumps({"rows": len(rows), "sample": rows[:1]}, indent=2))
