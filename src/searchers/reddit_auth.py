"""Resolve Reddit OAuth token, rdt-cli binary, and session cookies.

# change for b2c questionarie — prefer official OAuth (client credentials).
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
import time
from pathlib import Path

import requests

from src import config

logger = logging.getLogger(__name__)

REDDIT_SEARCH_URL = "https://oauth.reddit.com/search"
REDDIT_TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
REDDIT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36"
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_oauth_token: str | None = None
_oauth_expires_at: float = 0.0


def _resolve_cookie_path(path: str) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = _PROJECT_ROOT / p
    return p


def find_rdt_binary() -> str | None:
    """Return path to rdt executable (venv Scripts, then PATH)."""
    if sys.platform == "win32":
        venv_rdt = Path(sys.prefix) / "Scripts" / "rdt.exe"
    else:
        venv_rdt = Path(sys.prefix) / "bin" / "rdt"
    if venv_rdt.is_file():
        return str(venv_rdt)
    return shutil.which("rdt")


def credential_paths() -> list[Path]:
    paths: list[Path] = []
    custom = (config.REDDIT_CLI_COOKIE_PATH or "").strip()
    if custom:
        paths.append(_resolve_cookie_path(custom))
    paths.append(Path.home() / ".config" / "rdt-cli" / "credential.json")
    return paths


def load_reddit_cookies() -> dict[str, str] | None:
    """Load Reddit cookies from project path or rdt-cli default location."""
    for path in credential_paths():
        if not path.is_file():
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not read Reddit cookies from %s: %s", path, exc)
            continue
        if isinstance(raw, dict) and "cookies" in raw and isinstance(raw["cookies"], dict):
            cookies = {k: str(v) for k, v in raw["cookies"].items() if v}
        elif isinstance(raw, dict):
            cookies = {k: str(v) for k, v in raw.items() if v}
        else:
            continue
        if cookies.get("reddit_session") or cookies.get("session"):
            return cookies
    return None


def oauth_configured() -> bool:
    return bool(
        (config.REDDIT_CLIENT_ID or "").strip()
        and (config.REDDIT_CLIENT_SECRET or "").strip()
    )


def get_oauth_token() -> str | None:
    """Application-only OAuth token (client credentials). Cached until expiry."""
    global _oauth_token, _oauth_expires_at
    if not oauth_configured():
        return None
    now = time.time()
    if _oauth_token and now < _oauth_expires_at - 60:
        return _oauth_token
    try:
        resp = requests.post(
            REDDIT_TOKEN_URL,
            data={"grant_type": "client_credentials"},
            auth=(config.REDDIT_CLIENT_ID, config.REDDIT_CLIENT_SECRET),
            headers={"User-Agent": config.REDDIT_USER_AGENT or REDDIT_UA},
            timeout=20,
        )
        if resp.status_code >= 400:
            logger.warning(
                "Reddit OAuth token failed (%s): %s",
                resp.status_code, resp.text[:200],
            )
            return None
        payload = resp.json()
        token = payload.get("access_token")
        if not token:
            return None
        _oauth_token = token
        _oauth_expires_at = now + float(payload.get("expires_in") or 3600)
        return _oauth_token
    except Exception as exc:  # noqa: BLE001
        logger.warning("Reddit OAuth error: %s", exc)
        return None


def stage_credential_file() -> None:
    """Copy REDDIT_CLI_COOKIE_PATH into rdt-cli's config dir for subprocess use."""
    custom = (config.REDDIT_CLI_COOKIE_PATH or "").strip()
    if not custom:
        return
    src = _resolve_cookie_path(custom)
    if not src.is_file():
        return
    dest_dir = Path.home() / ".config" / "rdt-cli"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "credential.json"
    try:
        raw = json.loads(src.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return
    if "cookies" in raw:
        cred = raw
    elif isinstance(raw, dict):
        cred = {"cookies": raw, "source": "exported"}
    else:
        return
    dest.write_text(json.dumps(cred, indent=2), encoding="utf-8")
