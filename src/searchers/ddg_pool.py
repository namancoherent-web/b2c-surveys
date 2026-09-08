"""DuckDuckGo search pool — reliability fallback for all query angles."""

import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from ddgs import DDGS

from src.date_utils import current_date_context
from src.evidence_utils import build_fact, extract_value, infer_tab_hint, infer_topic, is_stat_domain
from src.fetchers.jina_reader import fetch_with_fallback
from src.ratelimit import acquire
from src.search_relevance import is_hit_relevant
from src import vpn_client

logger = logging.getLogger(__name__)

MAX_RETRIES = 2
MAX_FACTS_PER_PAGE = 6
JINA_PAGE_CAP = 20000
_WEB_RATE = float(os.getenv("WEB_RATE_PER_SEC", "1"))
_WEB_BURST = float(os.getenv("WEB_BURST", "3"))
_NOISE_MARKERS = (
    "% off", "off on", "free delivery", "free shipping", "secure payment",
    "add to cart", "rights reserved", "use code", "cod available", "sale is live",
    "mystery gift", "shop by", "buy now", "sign in", "log in", "newsletter",
    "checkout", "wishlist", "cookie", "subscribe", "best price", "best quality",
)


def _looks_like_noise(sentence: str) -> bool:
    low = sentence.lower()
    return any(marker in low for marker in _NOISE_MARKERS)


def _md_inline_clean(md: str) -> str:
    md = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", md)
    md = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", md)
    return md


def _candidate_snippets(text: str, snippet: str) -> list[tuple[str, str | None]]:
    out = []
    if text:
        for s in re.split(r"(?<=[.!?])\s+", text):
            s = s.strip()
            if len(s) < 30 or _looks_like_noise(s):
                continue
            value = extract_value(s)
            if value:
                out.append((s, value))
                if len(out) >= MAX_FACTS_PER_PAGE:
                    break
    if not out:
        base = (snippet or text or "").strip()
        if base and not _looks_like_noise(base):
            out.append((base, extract_value(base)))
    return out


def _looks_blocked_hits(hits: list | None, exc: BaseException | None = None) -> bool:
    """Heuristic: empty under pressure, or exception text suggesting a block."""
    if exc is not None:
        msg = str(exc).lower()
        if any(x in msg for x in ("403", "429", "202", "ratelimit", "rate limit", "blocked", "captcha")):
            return True
    if hits is not None and len(hits) == 0:
        return True
    return False


def _vpn_client_id() -> str:
    return os.getenv("VPN_CLIENT_ID") or f"ddg/{os.getenv('HOSTNAME', 'worker')}"


class DDGWorkerPool:
    """Thread-pooled DuckDuckGo search + Jina page fetch → unified findings."""

    def __init__(self, workers: int = 4):
        self.workers = max(1, workers)
        vpn_client.health()  # one-shot; skip rotation if tool is down

    def search_to_findings(
        self,
        query: str,
        max_results: int = 6,
        serves_goal: str = "",
        angle: str = "",
        segment: str = "",
    ) -> list[dict]:
        """Run one DDG query and return unified evidence facts."""
        hits = self._search(query, max_results)
        if not hits:
            return []

        retrieved_at = current_date_context()["iso"]
        facts = []
        ctx = f"{serves_goal} {angle} {query}"
        # Prefer the market segment for relevance; fall back to query text.
        relevance_segment = segment or query

        def _process(hit: dict) -> list[dict]:
            url = hit.get("href") or hit.get("url") or ""
            if not url:
                return []
            title = hit.get("title") or "web result"
            snippet = (hit.get("body") or hit.get("snippet") or "").strip()
            if not is_hit_relevant(title=title, url=url, snippet=snippet, segment=relevance_segment):
                logger.info("DDG skip off-topic hit: %s (%s)", title[:80], url[:80])
                return []
            text = ""
            # change for b2c questionarie — snippet-only when Jina off (avoid blocks)
            from src import config as _cfg
            deep = None
            if getattr(_cfg, "JINA_ENABLED", True):
                deep = fetch_with_fallback(url)
            if deep:
                text = _md_inline_clean(deep)[:JINA_PAGE_CAP]
            origin = "stat" if is_stat_domain(url) else "web"
            authority = "high" if origin == "stat" else "medium"
            out = []
            for sentence, value in _candidate_snippets(text, snippet):
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
                f["source_date"] = None
                f["retrieved_at"] = retrieved_at
                out.append(f)
            return out

        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = [pool.submit(_process, h) for h in hits]
            for fut in as_completed(futures):
                try:
                    facts.extend(fut.result())
                except Exception as exc:  # noqa: BLE001
                    logger.warning("DDG page fetch failed: %s", exc)
        return facts

    def _search_once(self, query: str, max_results: int, backend: str) -> list[dict]:
        acquire("web", _WEB_RATE, _WEB_BURST)
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results, backend=backend))

    def _search(self, query: str, max_results: int) -> list[dict]:
        # ``lite`` is currently the most reliable DDGS backend; ``auto`` often
        # returns empty under rate-limits. Fall back across backends.
        # On block signals, ask Proton automation-tool to rotate IP, then retry.
        backends = ("lite", "html", "auto")
        client_id = _vpn_client_id()
        empty_streak = 0

        for attempt in range(MAX_RETRIES + 1):
            last_exc: BaseException | None = None
            for backend in backends:
                try:
                    hits = self._search_once(query, max_results, backend)
                    if hits:
                        return hits
                    empty_streak += 1
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    logger.warning(
                        "DDG search failed (backend=%s attempt %s): %s",
                        backend, attempt + 1, exc,
                    )
                    if _looks_blocked_hits(None, exc):
                        empty_streak += 1
                        break

            # Rotate only on block-like errors, or repeated empty across backends
            # (first empty pass is often "no hits", not a block).
            should_rotate = False
            if last_exc is not None and _looks_blocked_hits(None, last_exc):
                should_rotate = True
            elif empty_streak >= len(backends) and attempt >= 1:
                should_rotate = True

            if should_rotate:
                vpn_client.heartbeat(client_id, "blocked")
                if vpn_client.rotate(client_id):
                    vpn_client.heartbeat(client_id, "running")
                    empty_streak = 0
                    continue
                # Rotation unavailable → degrade (return empty below)

            time.sleep((attempt + 1) * 1.0)
        return []


if __name__ == "__main__":
    pool = DDGWorkerPool(workers=2)
    results = pool.search_to_findings("organic milk market share 2026", max_results=3)
    print(f"DDG findings: {len(results)}")
