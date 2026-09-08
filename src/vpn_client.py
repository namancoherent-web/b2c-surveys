"""HTTP client for the Proton VPN rotate service (automation-tool on :8765).

Used only by the web fetch layer (DDG / Jina) — never from llm.py or db.py.
Fails soft when the tool is off or unreachable so harvest can degrade, not crash.

Host tool: https://github.com/Aman-Coherent/automation-tool
Containers reach it via VPN_ROTATE_URL=http://host.docker.internal:8765
"""

from __future__ import annotations

import os
import socket

import requests

BASE = os.getenv("VPN_ROTATE_URL", "http://host.docker.internal:8765").rstrip("/")
ENABLED = os.getenv("VPN_ROTATE_ENABLED", "true").lower() in ("1", "true", "yes")

_health_checked = False
_health_ok = False


def _client_id_default() -> str:
    # automation-tool requires "/" in scraper client_ids (e.g. region/label).
    return os.getenv("VPN_CLIENT_ID") or f"survey-agent/{socket.gethostname()}"


def health() -> bool:
    """Call once at fetch-layer init; if down, skip rotation and continue."""
    global _health_checked, _health_ok
    if not ENABLED:
        _health_checked = True
        _health_ok = False
        return False
    try:
        _health_ok = requests.get(f"{BASE}/health", timeout=3).ok
    except Exception:
        _health_ok = False
    _health_checked = True
    return _health_ok


def _ensure_health() -> bool:
    global _health_checked
    if not _health_checked:
        return health()
    return _health_ok


def heartbeat(client_id: str | None, status: str) -> None:
    if not ENABLED or not _ensure_health():
        return
    try:
        requests.post(
            f"{BASE}/scraper/heartbeat",
            json={"client_id": client_id or _client_id_default(), "status": status},
            timeout=3,
        )
    except Exception:
        pass


def rotate(client_id: str | None = None) -> bool:
    """Block until a new IP is ready (tool may wait up to ~2 min for Free cooldown).

    ``client_id`` must contain a slash (e.g. ``survey-agent/worker-1``) —
    Aman-Coherent/automation-tool returns HTTP 400 otherwise.
    """
    if not ENABLED or not _ensure_health():
        return False
    cid = client_id or _client_id_default()
    if "/" not in cid:
        cid = f"survey-agent/{cid}"
    try:
        r = requests.post(
            f"{BASE}/rotate",
            json={
                "client_id": cid,
                "trigger": "scraper_block",
            },
            timeout=180,
        )
        if not r.ok:
            try:
                detail = r.json()
            except Exception:
                detail = r.text[:300]
            print(f"[vpn] rotate HTTP {r.status_code}: {detail}")
        return r.ok
    except Exception as exc:
        print(f"[vpn] rotate error: {exc}")
        return False


def stop(client_id: str | None = None) -> None:
    if not ENABLED or not _ensure_health():
        return
    try:
        requests.post(
            f"{BASE}/scraper/stop",
            json={"client_id": client_id or _client_id_default()},
            timeout=3,
        )
    except Exception:
        pass
