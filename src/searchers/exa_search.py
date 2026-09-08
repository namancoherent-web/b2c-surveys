"""Exa semantic web search (free tier) for consumer-behavior queries."""

import hashlib
import json
import logging
import re
import time
from datetime import datetime, timezone

import requests

from src import config
from src.date_utils import current_date_context
from src.evidence_utils import build_fact, extract_value, infer_tab_hint, infer_topic, is_stat_domain
from src.fetchers.jina_reader import fetch_clean
from src.search_relevance import is_hit_relevant

logger = logging.getLogger(__name__)

EXA_API_URL = "https://api.exa.ai/search"
MAX_RETRIES = 2
THIN_TEXT_CHARS = 500
JINA_PAGE_CAP = 20000
MAX_FACTS_PER_RESULT = 6
CACHE_TTL_SECONDS = 24 * 3600

_cache: dict[str, tuple[float, list]] = {}


def _cache_get(key: str) -> list | None:
    entry = _cache.get(key)
    if not entry:
        return None
    ts, data = entry
    if time.time() - ts > CACHE_TTL_SECONDS:
        _cache.pop(key, None)
        return None
    return data


def _cache_set(key: str, data: list) -> None:
    _cache[key] = (time.time(), data)


def search_exa(
    query: str,
    num_results: int = 6,
    use_autoprompt: bool = True,
    type: str = "auto",
) -> list[dict]:
    """Call Exa REST API; return normalized result dicts or [] on failure."""
    if not config.EXA_API_KEY:
        return []

    cache_key = hashlib.sha256(
        f"{query}|{num_results}|{use_autoprompt}|{type}".encode()
    ).hexdigest()
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    payload = {
        "query": query,
        "numResults": num_results,
        "useAutoprompt": use_autoprompt,
        "type": type,
        "contents": {"text": True},
    }
    headers = {
        "x-api-key": config.EXA_API_KEY,
        "Content-Type": "application/json",
    }

    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = requests.post(EXA_API_URL, headers=headers, json=payload, timeout=30)
            if resp.status_code in (401, 403):
                logger.warning("Exa auth error (%s): %s", resp.status_code, resp.text[:200])
                return []
            if resp.status_code == 429:
                logger.warning("Exa rate limit; backing off")
                time.sleep((attempt + 1) * 2.0)
                continue
            if resp.status_code != 200:
                logger.warning("Exa HTTP %s: %s", resp.status_code, resp.text[:200])
                return []
            raw = resp.json().get("results", []) or []
            out = []
            for r in raw:
                out.append({
                    "url": r.get("url") or "",
                    "title": r.get("title") or "web result",
                    "publishedDate": r.get("publishedDate"),
                    "text": r.get("text") or "",
                    "score": r.get("score"),
                })
            _cache_set(cache_key, out)
            return out
        except requests.RequestException as exc:
            logger.warning("Exa request failed: %s", exc)
            time.sleep((attempt + 1) * 1.0)
    return []


def _md_inline_clean(md: str) -> str:
    md = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", md)
    md = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", md)
    return md


def _candidate_snippets(text: str, snippet: str) -> list[tuple[str, str | None]]:
    out = []
    if text:
        for s in re.split(r"(?<=[.!?])\s+", text):
            s = s.strip()
            if len(s) < 30:
                continue
            value = extract_value(s)
            if value:
                out.append((s, value))
                if len(out) >= MAX_FACTS_PER_RESULT:
                    break
    if not out:
        base = (snippet or text or "").strip()
        if base:
            out.append((base, extract_value(base)))
    return out


def exa_to_findings(
    results: list[dict],
    serves_goal: str,
    angle: str,
    segment: str = "",
) -> list[dict]:
    """Convert Exa hits to unified evidence-fact dicts."""
    retrieved_at = current_date_context()["iso"]
    findings = []
    relevance_segment = segment or serves_goal
    for r in results:
        url = r.get("url") or ""
        if not url:
            continue
        title = r.get("title") or "web result"
        text = (r.get("text") or "").strip()
        if not is_hit_relevant(
            title=title, url=url, snippet=text[:400], segment=relevance_segment
        ):
            logger.info("Exa skip off-topic hit: %s (%s)", title[:80], url[:80])
            continue
        if len(text) < THIN_TEXT_CHARS:
            # change for b2c questionarie — skip Jina deep-fetch when disabled
            from src import config as _cfg
            if getattr(_cfg, "JINA_ENABLED", True):
                deep = fetch_clean(url)
                if deep:
                    text = _md_inline_clean(deep)[:JINA_PAGE_CAP]
        source_date = r.get("publishedDate")
        origin = "stat" if is_stat_domain(url) else "exa"
        authority = "high" if origin == "stat" else "medium"
        ctx = f"{serves_goal} {angle} {title}"
        for sentence, value in _candidate_snippets(text, text[:400]):
            f = build_fact(
                fact=sentence[:3000],
                origin=origin,
                authority=authority,
                source_ref=url,
                title=title,
                topic=infer_topic(f"{ctx} {sentence}"),
                tab_hint=infer_tab_hint(f"{ctx} {sentence}"),
                value=value,
            )
            if not f:
                continue
            f["source_date"] = source_date
            f["retrieved_at"] = retrieved_at
            findings.append(f)
    return findings


if __name__ == "__main__":
    demo = search_exa("front load washing machine consumer complaints", num_results=3)
    print(json.dumps({"count": len(demo), "sample": demo[:1]}, indent=2))
