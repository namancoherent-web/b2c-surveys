"""Summarize finished RQ survey jobs + local output JSON quality.

Usage (while / after a batch):
    python scripts/batch_status.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import redis
from rq import Queue
from rq.registry import FailedJobRegistry, FinishedJobRegistry, StartedJobRegistry


def _meta_from_output(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return (data.get("survey") or {}).get("metadata") or {}


def main() -> int:
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    queue_name = os.getenv("RQ_QUEUE_NAME", "surveys")
    conn = redis.from_url(redis_url)
    q = Queue(queue_name, connection=conn)

    started = StartedJobRegistry(queue=q)
    finished = FinishedJobRegistry(queue=q)
    failed = FailedJobRegistry(queue=q)

    print(f"queue={queue_name}  queued={len(q)}  started={len(started)}  "
          f"finished={len(finished)}  failed={len(failed)}")
    print()

    out_dir = _ROOT / "output"
    rows = []
    if out_dir.is_dir():
        for path in sorted(out_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            if "_20" in path.stem and path.stem.count("_") >= 2:
                # skip timestamped duplicates in summary unless recent — include all
                pass
            meta = _meta_from_output(path)
            if not meta:
                continue
            rows.append((
                path.name,
                meta.get("total_questions"),
                meta.get("grounded_pct"),
                len(meta.get("warnings") or []),
            ))

    print(f"{'file':<45} {'Qs':>4} {'grounded%':>10} {'warns':>5}")
    print("-" * 70)
    for name, qs, gp, wn in rows[:20]:
        print(f"{name:<45} {qs!s:>4} {gp!s:>10} {wn!s:>5}")

    if failed:
        print("\nFailed job ids:")
        for jid in list(failed.get_job_ids())[:10]:
            print(f"  {jid}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
