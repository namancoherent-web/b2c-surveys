"""Enqueue survey-generation jobs onto the RQ queue.

Usage (from host, with Redis on localhost:6379):

    python enqueue.py segments.json
    python enqueue.py segments-one.json

# change for b2c questionarie

Segment-only: each segment becomes one ``classify_and_fanout`` job.
That worker classifies once; on B2C accept it dynamically enqueues the fixed
5 regional jobs + global fan-in. Rejected segments enqueue nothing regional.
"""

from __future__ import annotations

import json
import os
import re
import sys

import redis
from rq import Queue, Retry

from src.tasks import classify_and_fanout


def _parse_timeout(raw: str) -> int:
    """Parse RQ job timeout — seconds or suffix s/m/h/d (e.g. 60m)."""
    raw = (raw or "60m").strip().lower()
    if raw.isdigit():
        return int(raw)
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([smhd])", raw)
    if m:
        n, unit = float(m.group(1)), m.group(2)
        mult = {"s": 1, "m": 60, "h": 3600, "d": 86400}
        return int(n * mult[unit])
    return int(float(raw))


def main() -> None:
    manifest_path = sys.argv[1] if len(sys.argv) > 1 else "segments.json"
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)

    if "region" in manifest:
        print(
            "WARNING: manifest 'region' is ignored — fan-out is always the fixed "
            "5 continents after B2C classify."
        )

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    queue_name = os.getenv("RQ_QUEUE_NAME", "surveys")
    # Classify is cheap; allow time for child enqueue + Redis round-trips.
    classify_timeout = _parse_timeout(os.getenv("RQ_CLASSIFY_TIMEOUT", "5m"))

    conn = redis.from_url(redis_url)
    q = Queue(queue_name, connection=conn)

    segments = manifest["segments"]
    for seg in segments:
        job = q.enqueue(
            classify_and_fanout,
            seg,
            job_timeout=classify_timeout,
            retry=Retry(max=2, interval=[15, 60]),
            meta={"phase": "classify", "segment": seg},
        )
        print(f"enqueued classify_and_fanout {seg!r} -> {job.id}")


if __name__ == "__main__":
    main()
