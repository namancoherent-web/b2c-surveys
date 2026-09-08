"""Reddit consumer-voice search.

# change for b2c questionarie — prefer official OAuth; cookie/rdt remain fallbacks.

Honest tradeoff: cookie-auth scraping carries account-ban risk. Prefer
REDDIT_CLIENT_ID + REDDIT_CLIENT_SECRET (application-only OAuth). Cookies /
rdt-cli are optional fallbacks. Disable via REDDIT_ENABLED — pipeline degrades
to YouTube/Exa/DDG.

Windows note: ``rdt login`` often fails on Edge/Chrome without Administrator
because browser cookie files are encrypted/locked. Use
``scripts/setup_reddit_cookies.py`` to paste ``reddit_session`` manually from
Edge DevTools instead (fallback path only).
"""

import json
import logging
import random
import subprocess
import time
from datetime import datetime, timezone

import requests

from src import config
from src.date_utils import current_date_context
from src.evidence_utils import build_voice_fact, infer_tab_hint
from src.searchers.reddit_auth import (
    REDDIT_SEARCH_URL,
    REDDIT_UA,
    find_rdt_binary,
    get_oauth_token,
    load_reddit_cookies,
    oauth_configured,
    stage_credential_file,
)

logger = logging.getLogger(__name__)

_COMPLAINT_MARKERS = (
    "complaint", "problem", "issue", "hate", "wish", "disappoint", "frustrat",
    "annoying", "too expensive", "overpriced", "can't find", "hard to find",
    "why does", "why is", "how do i", "anyone else", "struggle", "concern",
    "avoid", "worst", "refund", "return", "not worth", "waste", "broke",
    "leak", "smell", "rash", "irritat", "burn", "sting",
)

_last_call_at = 0.0
_MIN_INTERVAL = 3.0

# After one slow rdt-cli timeout, skip further CLI calls for this process so
# evidence harvest is not blocked for minutes on a dead Reddit path.
_rdt_skip = False


def _rate_limit() -> None:
    global _last_call_at
    now = time.time()
    wait = _MIN_INTERVAL + random.uniform(0.2, 0.8) - (now - _last_call_at)
    if wait > 0:
        time.sleep(wait)
    _last_call_at = time.time()


def _reddit_ready() -> bool:
    if not config.REDDIT_ENABLED:
        return False
    if oauth_configured():
        return True
    return load_reddit_cookies() is not None


def _cookie_header(cookies: dict[str, str]) -> str:
    return "; ".join(f"{k}={v}" for k, v in cookies.items())


def _normalize_post(row: dict) -> dict | None:
    if not isinstance(row, dict):
        return None
    data = row.get("data") if "data" in row else row
    if not isinstance(data, dict):
        return None
    url = data.get("url") or ""
    permalink = data.get("permalink") or ""
    if permalink and not url:
        url = f"https://www.reddit.com{permalink}"
    if not url:
        return None
    comments = []
    for c in (data.get("top_comments") or data.get("comments") or [])[:5]:
        if isinstance(c, dict):
            body = c.get("body") or c.get("text") or ""
            if body:
                comments.append(body)
        elif isinstance(c, str):
            comments.append(c)
    return {
        "url": url,
        "title": data.get("title") or "reddit post",
        "subreddit": data.get("subreddit") or "",
        "selftext": data.get("selftext") or data.get("body") or "",
        "top_comments": comments,
        "created_utc": data.get("created_utc") or data.get("created"),
        "score": data.get("score"),
    }


def _search_reddit_oauth(
    query: str,
    subreddits: list[str] | None,
    limit: int,
    timeout: int,
) -> list[dict]:
    """Official Reddit API via application-only OAuth (client credentials)."""
    token = get_oauth_token()
    if not token:
        return []
    headers = {
        "Authorization": f"Bearer {token}",
        "User-Agent": config.REDDIT_USER_AGENT or REDDIT_UA,
        "Accept": "application/json",
    }
    params = {
        "q": query,
        "limit": min(limit, 25),
        "sort": "relevance",
        "type": "link",
        "raw_json": 1,
    }
    if subreddits:
        params["q"] = f"{query} subreddit:{subreddits[0].lstrip('r/')}"
    try:
        resp = requests.get(
            REDDIT_SEARCH_URL, params=params, headers=headers, timeout=timeout,
        )
        if resp.status_code in (401, 403):
            logger.warning("Reddit OAuth search auth failed (%s)", resp.status_code)
            return []
        resp.raise_for_status()
        children = resp.json().get("data", {}).get("children", [])
    except (requests.RequestException, json.JSONDecodeError, KeyError) as exc:
        logger.warning("Reddit OAuth search failed: %s", exc)
        return []
    posts = []
    for child in children:
        post = _normalize_post(child)
        if post:
            posts.append(post)
    return posts


def _search_reddit_http(
    query: str,
    subreddits: list[str] | None,
    limit: int,
    timeout: int,
) -> list[dict]:
    """Search Reddit via oauth API using saved session cookies (fallback)."""
    cookies = load_reddit_cookies()
    if not cookies:
        return []

    headers = {
        "User-Agent": REDDIT_UA,
        "Cookie": _cookie_header(cookies),
        "Accept": "application/json",
    }
    params = {
        "q": query,
        "limit": min(limit, 25),
        "sort": "relevance",
        "type": "link",
        "raw_json": 1,
    }
    if subreddits:
        params["q"] = f"{query} subreddit:{subreddits[0].lstrip('r/')}"

    try:
        resp = requests.get(REDDIT_SEARCH_URL, params=params, headers=headers, timeout=timeout)
        if resp.status_code in (401, 403):
            logger.warning("Reddit API auth failed (%s) — refresh reddit_session cookie", resp.status_code)
            return []
        resp.raise_for_status()
        children = resp.json().get("data", {}).get("children", [])
    except (requests.RequestException, json.JSONDecodeError, KeyError) as exc:
        logger.warning("Reddit HTTP search failed: %s", exc)
        return []

    posts = []
    for child in children:
        post = _normalize_post(child)
        if post:
            posts.append(post)
    return posts


def _search_reddit_rdt(
    query: str,
    subreddits: list[str] | None,
    limit: int,
    timeout: int,
) -> list[dict]:
    """Fallback: shell out to rdt-cli when HTTP path fails."""
    rdt = find_rdt_binary()
    if not rdt:
        logger.warning("rdt-cli binary not found")
        return []

    stage_credential_file()
    cmd = [rdt, "search", query, "-n", str(limit), "--compact", "--json"]
    if subreddits:
        for sr in subreddits[:5]:
            cmd.extend(["-r", sr.lstrip("r/")])

    global _rdt_skip
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        if proc.returncode != 0:
            logger.warning("rdt-cli exit %s: %s", proc.returncode, (proc.stderr or proc.stdout or "")[:300])
            return []
        payload = json.loads(proc.stdout or "{}")
    except subprocess.TimeoutExpired as exc:
        logger.warning("rdt-cli search timed out: %s — skipping further rdt calls", exc)
        _rdt_skip = True
        return []
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("rdt-cli search failed: %s", exc)
        return []

    if isinstance(payload, dict) and payload.get("ok") is False:
        return []

    rows = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        rows = payload.get("posts", []) if isinstance(payload, dict) else []

    posts = []
    for row in rows:
        post = _normalize_post(row)
        if post:
            posts.append(post)
    return posts


def search_reddit(
    query: str,
    subreddits: list[str] | None = None,
    limit: int = 15,
    timeout: int = 18,
) -> list[dict]:
    """Search Reddit: OAuth → cookie HTTP → rdt-cli. Soft-fail always.

    # change for b2c questionarie
    """
    global _rdt_skip
    if not _reddit_ready():
        return []

    _rate_limit()
    # Prefer official OAuth when configured.
    if oauth_configured():
        posts = _search_reddit_oauth(query, subreddits, limit, min(timeout, 12))
        if posts:
            return posts
    posts = _search_reddit_http(query, subreddits, limit, min(timeout, 12))
    if posts:
        return posts
    if _rdt_skip:
        return []
    try:
        posts = _search_reddit_rdt(query, subreddits, limit, timeout)
    except Exception as exc:  # noqa: BLE001
        logger.warning("rdt-cli unexpected error: %s", exc)
        _rdt_skip = True
        return []
    return posts


def _infer_voice_topic(text: str) -> str:
    low = (text or "").lower()
    if any(m in low for m in _COMPLAINT_MARKERS):
        return "complaints"
    if "?" in text:
        return "customer_voice"
    return "customer_language"


def _utc_iso(ts) -> str | None:
    if ts is None:
        return None
    try:
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
        return str(ts)[:10]
    except (TypeError, ValueError, OSError):
        return None


def reddit_to_findings(posts: list[dict], serves_goal: str, angle: str) -> list[dict]:
    """Convert Reddit posts + comments into language-only unified findings."""
    retrieved_at = current_date_context()["iso"]
    findings = []

    for post in posts:
        url = post.get("url") or ""
        if not url:
            continue
        title = post.get("title") or "reddit post"
        source_date = _utc_iso(post.get("created_utc"))
        selftext = (post.get("selftext") or "").strip()
        topic = _infer_voice_topic(f"{selftext} {title}")

        if selftext and len(selftext) >= 20:
            f = build_voice_fact(
                fact=selftext[:3000],
                origin="reddit",
                source_ref=url,
                title=title,
                topic=topic,
                tab_hint=infer_tab_hint(f"{serves_goal} {angle} {selftext}"),
            )
            if f:
                f["source_date"] = source_date
                f["retrieved_at"] = retrieved_at
                findings.append(f)

        for body in (post.get("top_comments") or [])[:5]:
            body = (body or "").strip()
            if len(body) < 20:
                continue
            ctopic = _infer_voice_topic(body)
            f = build_voice_fact(
                fact=body[:3000],
                origin="reddit",
                source_ref=url,
                title=f"{title} — comment",
                topic=ctopic,
                tab_hint=infer_tab_hint(body),
            )
            if f:
                f["source_date"] = source_date
                f["retrieved_at"] = retrieved_at
                findings.append(f)
    return findings


if __name__ == "__main__":
    posts = search_reddit("washing machine mold smell", limit=5)
    print(json.dumps({"posts": len(posts)}, indent=2))
