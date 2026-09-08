"""Phase 5 — prove RQ whole-job retries + isolation.

Enqueues:
  1) a job that always fails (Retry max=2, short intervals)
  2) a trivial sibling that must still succeed

Usage (Redis + at least one worker must be up; rebuild workers if tasks.py changed):
    python scripts/test_rq_retry.py

Pass when: failing job lands in FailedJobRegistry after retries,
and the sibling job finishes OK (other work unaffected).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import redis
from rq import Queue, Retry
from rq.job import Job
from rq.registry import FailedJobRegistry, FinishedJobRegistry

from src.tasks import deliberately_fail_for_retry_test, phase5_ok_marker


def _wait_terminal(conn, job_id: str, timeout: float = 120.0) -> Job:
    """Wait until finished/failed — not merely scheduled for a retry."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = Job.fetch(job_id, connection=conn)
        status = str(job.get_status()).lower()
        # JobStatus.FAILED / "failed" / enum name
        if "finished" in status or status.endswith("finished"):
            return job
        if "failed" in status or status.endswith("failed"):
            return job
        if status in ("stopped", "canceled", "cancelled"):
            return job
        time.sleep(0.5)
    return Job.fetch(job_id, connection=conn)


def main() -> int:
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    queue_name = os.getenv("RQ_QUEUE_NAME", "surveys")
    conn = redis.from_url(redis_url)
    q = Queue(queue_name, connection=conn)

    # Short intervals so the test finishes quickly (not production 30/120s).
    fail_job = q.enqueue(
        deliberately_fail_for_retry_test,
        job_timeout=60,
        retry=Retry(max=2, interval=[2, 4]),
        meta={"phase5_test": "fail"},
    )
    ok_job = q.enqueue(
        phase5_ok_marker,
        job_timeout=60,
        meta={"phase5_test": "ok"},
    )
    print(f"enqueued FAIL -> {fail_job.id}")
    print(f"enqueued OK   -> {ok_job.id}")
    print("waiting for both to reach a terminal state…")

    fail_done = _wait_terminal(conn, fail_job.id, timeout=90)
    ok_done = _wait_terminal(conn, ok_job.id, timeout=90)

    failed_reg = FailedJobRegistry(queue=q)
    finished_reg = FinishedJobRegistry(queue=q)

    fail_status = fail_done.get_status()
    ok_status = ok_done.get_status()
    fail_in_failed = fail_done.id in failed_reg
    ok_in_finished = ok_done.id in finished_reg or ok_status == "finished"

    # Retries: original + 2 = up to 3 attempts; RQ stores retries_left.
    retries_left = fail_done.retries_left
    print(f"FAIL status={fail_status} in_failed_registry={fail_in_failed} retries_left={retries_left}")
    print(f"OK   status={ok_status} in_finished_registry={ok_in_finished} result={ok_done.result}")

    passed = fail_in_failed and ok_in_finished and ok_status == "finished"
    if passed:
        print("PASS: failing job retried then failed; sibling job succeeded.")
        return 0
    print("FAIL: expected failed-registry + successful sibling.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
