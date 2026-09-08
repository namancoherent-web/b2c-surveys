"""Stage 1-2 end-to-end: classification and the gate, against the live model.

# change for b2c questionarie — A11

The gate LOGIC is covered without a model by ``python src/nodes/scope_classify.py``.
This exercises the part that logic cannot prove: whether the classifier actually
labels real inputs correctly, and whether a non-accepted outcome really stops
the pipeline.

A stop is verified by the ABSENCE OF OUTPUT FILES, not by the console message —
that is the difference between a gate and a prompt instruction.

Costs four LLM calls. Usage:
  python scripts/test_stages.py            # all four cases
  python scripts/test_stages.py b2b        # one case by name
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.survey_io import slugify  # noqa: E402
from src.tasks import classify_segment_async  # noqa: E402

OUTPUT = ROOT / "output"
PUBLISHED = ROOT / "published_surveys"

# name, input, expected status, extra predicate on the classification result
CASES = [
    (
        "b2c",
        "a video streaming subscription service",
        "accepted",
        lambda r: (r.get("classification") or {}).get("segment_type") == "b2c",
        "clear consumer market proceeds",
    ),
    (
        "b2b",
        "enterprise CRM software",
        "rejected",
        lambda r: "B2C and hybrid" in (r.get("reason") or ""),
        "pure B2B is refused with the methodology reason",
    ),
    (
        "hybrid",
        "a project management tool sold to both individual freelancers and "
        "enterprise teams, on personal plans and company-paid seats",
        "accepted",
        lambda r: bool((r.get("classification") or {}).get("sells_to_both")),
        "dual-motion proceeds and is flagged as selling to both",
    ),
    (
        "ambiguous",
        "Meridian",
        "needs_clarification",
        lambda r: len((r.get("clarifying_question") or "").strip()) > 10,
        "a bare company name is asked about, not guessed at",
    ),
]


def _artifacts_for(segment: str) -> list[Path]:
    """Every file a full run would have produced for this input."""
    slug = slugify(segment)
    found = [p for p in OUTPUT.glob(f"{slug}_*.json")
             if not p.name.endswith(("_rejected.json", "_needs_clarification.json"))]
    found += list(OUTPUT.glob(f"{slug}.json"))
    if (PUBLISHED / slug).is_dir():
        found += list((PUBLISHED / slug).glob("*.json"))
    return found


def run_case(name: str, text: str, expected: str, predicate, why: str) -> bool:
    print(f"\n[{name}] {text[:64]!r}")
    before = set(_artifacts_for(text))

    result = asyncio.run(classify_segment_async(text))
    status = result.get("status")
    clf = result.get("classification") or {}

    ok_status = status == expected
    print(f"  status: {status}  (expected {expected}) {'PASS' if ok_status else 'FAIL'}")
    if clf:
        print(f"  type={clf.get('segment_type')} conf={clf.get('classification_confidence')} "
              f"hybrid={clf.get('is_hybrid')} sells_to_both={clf.get('sells_to_both')}")
        print(f"  normalized: {clf.get('normalized_segment')!r}")
    reason = (result.get("reason") or clf.get("classification_reason") or "").strip()
    if reason:
        print(f"  reason: {reason[:220]}")

    ok_pred = bool(predicate(result))
    print(f"  {why}: {'PASS' if ok_pred else 'FAIL'}")

    # A non-accepted outcome must not have produced any survey artifact.
    ok_files = True
    if expected != "accepted":
        after = set(_artifacts_for(text))
        new = after - before
        ok_files = not new and result.get("regions_enqueued") == 0
        print(f"  no survey files written: {'PASS' if ok_files else 'FAIL ' + str(new)}")

    return ok_status and ok_pred and ok_files


def main() -> int:
    os.environ.setdefault("RUN_REGIONS", "North America")
    wanted = sys.argv[1:]
    cases = [c for c in CASES if not wanted or c[0] in wanted]
    results = [(c[0], run_case(*c)) for c in cases]

    print("\n" + "=" * 58)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    failed = [n for n, ok in results if not ok]
    print(f"  stage tests {'PASSED' if not failed else 'FAILED: ' + ', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
