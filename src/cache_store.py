"""Simple disk cache for segment-level artifacts (MKP, questionnaires, personas).

Cache keys are SHA-256 digests. Failures never raise — callers get a miss.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Optional

_CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache" / "survey_agent"


def _path(namespace: str, key: str) -> Path:
    safe_ns = "".join(c if c.isalnum() or c in "-_" else "_" for c in namespace)[:80]
    return _CACHE_DIR / safe_ns / f"{key}.json"


def make_key(*parts: Any) -> str:
    payload = json.dumps(parts, sort_keys=True, default=str, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:40]


def get_json(namespace: str, key: str) -> Optional[dict]:
    path = _path(namespace, key)
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def set_json(namespace: str, key: str, value: dict) -> None:
    path = _path(namespace, key)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    except OSError:
        pass
