"""Jina Reader — default page-fetch mechanism for URL content.

Jina Reader (https://r.jina.ai) is a free public endpoint that returns clean,
LLM-ready markdown for any URL — no API key: ``GET https://r.jina.ai/{url}``.
Use it as the DEFAULT page-fetch across the codebase; ``fetch_with_fallback``
degrades to requests + ``html_utils.clean_html`` when Jina fails.

``html_utils.clean_html`` is still used for CMI database HTML (not URL fetches).
"""

import os
import time

import requests

from src.html_utils import clean_html
from src.ratelimit import acquire
from src import vpn_client

JINA_PREFIX = "https://r.jina.ai/"
MAX_RETRIES = 2
_UA = "Mozilla/5.0 (compatible; cmi-survey-agent/1.0; market-research fetcher)"
_WEB_RATE = float(os.getenv("WEB_RATE_PER_SEC", "1"))
_WEB_BURST = float(os.getenv("WEB_BURST", "3"))

# Per-run cache so the same URL isn't re-fetched across queries. Failures are
# cached too (None) so a dead URL costs one attempt per run.
_cache: dict = {}

# Bot-walls come back as HTTP 200 with an error page — treat as failure.
_BLOCK_MARKERS = (
    "access to this page has been denied",
    "just a moment",
    "verify you are human",
    "enable javascript and cookies",
    "checking your browser",
)

_vpn_inited = False


def _ensure_vpn() -> None:
    global _vpn_inited
    if not _vpn_inited:
        vpn_client.health()
        _vpn_inited = True


def _vpn_client_id() -> str:
    return os.getenv("VPN_CLIENT_ID") or f"jina/{os.getenv('HOSTNAME', 'worker')}"


def _looks_blocked_response(status_code: int | None, body: str | None) -> bool:
    if status_code in (202, 403, 429, 503):
        return True
    if body:
        head = body[:400].lower()
        if any(m in head for m in _BLOCK_MARKERS):
            return True
    return False


def _postprocess(text: str):
    """Reject blocked-page bodies; strip the Reader's Title/URL preamble so only
    the page's markdown content remains."""
    head = text[:400].lower()
    if any(m in head for m in _BLOCK_MARKERS):
        return None
    if "Markdown Content:" in text[:600]:
        text = text.split("Markdown Content:", 1)[1].lstrip()
    return text or None


def fetch_clean(url: str, timeout: int = 30):
    """Fetch a URL as clean markdown via Jina Reader.

    Returns the markdown text, or None on failure (non-200 / timeout /
    exception, after 1-2 short-backoff retries on transient errors). The
    returned markdown structure (headings, lists, tables) is preserved as-is —
    it is already clean and LLM-ready; do not strip it further.
    """
    # change for b2c questionarie — skip Jina entirely when disabled (no VPN)
    from src import config as _cfg
    if not getattr(_cfg, "JINA_ENABLED", True):
        _cache[url] = None
        return None

    if url in _cache:
        return _cache[url]

    _ensure_vpn()
    client_id = _vpn_client_id()
    result = None
    for attempt in range(MAX_RETRIES + 1):
        acquire("web", _WEB_RATE, _WEB_BURST)
        status = None
        body = None
        try:
            resp = requests.get(
                JINA_PREFIX + url,
                headers={"User-Agent": _UA, "Accept": "text/plain"},
                timeout=timeout,
            )
            status = resp.status_code
            body = resp.text
            if status == 200 and body.strip():
                result = _postprocess(body)
                if result is not None:
                    break
                # 200 but block-page content
                if _looks_blocked_response(status, body):
                    vpn_client.heartbeat(client_id, "blocked")
                    if vpn_client.rotate(client_id):
                        vpn_client.heartbeat(client_id, "running")
                        continue
                    break
            if status in (401, 404, 422, 451):
                break  # permanent for this URL — retrying won't help
            if _looks_blocked_response(status, body):
                vpn_client.heartbeat(client_id, "blocked")
                if vpn_client.rotate(client_id):
                    vpn_client.heartbeat(client_id, "running")
                    continue
                break
        except requests.RequestException:
            pass
        time.sleep(0.8 * (attempt + 1))

    _cache[url] = result
    return result


def fetch_with_fallback(url: str, timeout: int = 30):
    """Try Jina Reader first; on failure fall back to requests + clean_html.

    Logs which path succeeded (jina | fallback | failed) for diagnostics.
    Returns text or None.
    """
    md = fetch_clean(url, timeout=timeout)
    if md:
        print(f"[fetch] jina: {url}")
        return md
    try:
        acquire("web", _WEB_RATE, _WEB_BURST)
        resp = requests.get(url, headers={"User-Agent": _UA}, timeout=timeout)
        if resp.status_code == 200 and "html" in resp.headers.get("Content-Type", "").lower():
            text = clean_html(resp.text)
            if text:
                print(f"[fetch] fallback: {url}")
                return text
    except requests.RequestException:
        pass
    print(f"[fetch] failed: {url}")
    return None


if __name__ == "__main__":
    rich = "https://en.wikipedia.org/wiki/Sneakers"
    md = fetch_clean(rich)
    if md:
        print(f"fetch_clean OK — {len(md)} chars total. First ~500:\n")
        print(md[:500])
    else:
        print("fetch_clean returned None for the rich URL")

    print("\n--- fallback demo (URL Jina cannot reach) ---")
    out = fetch_with_fallback("http://localhost:1/unreachable")
    print("result:", "None" if out is None else f"{len(out)} chars")
