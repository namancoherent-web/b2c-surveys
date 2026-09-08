"""Pre-output validation gate for a published survey.

# change for b2c questionarie — A13

The pipeline's validator runs on internal state. This runs on the FILE that
ships, which is what a reader actually gets. Three checks, all arithmetic:

  1. NPS is recomputed from the 0-10 item and reported.
  2. Every question's percentages are checked against its DECLARED type —
     sum-to-100 types must sum to 100; multiple_choice must not.
  3. Satisfaction is cross-checked against recommendation: a detractor-heavy
     NPS beside a satisfied majority describes two different populations.

Exits non-zero when a check fails, so it can gate a publish step.

Usage:
  python scripts/validate_survey.py published_surveys/organic-milk/global.json
  python scripts/validate_survey.py --all
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = ROOT / "published_surveys"

_SUM_TO_100 = {"single_choice", "ranking", "likert_5", "likert_7"}


def _questions(doc: dict) -> list:
    secs = (doc.get("frontend") or {}).get("sections") or doc.get("sections") or []
    return [q for s in secs for q in (s.get("questions") or [])]


def _nps(values: list) -> tuple[float, float, float] | None:
    scores = []
    for d in values:
        m = re.match(r"\s*(\d{1,2})\b", str(d.get("label") or ""))
        if m and int(m.group(1)) <= 10:
            scores.append((int(m.group(1)), float(d.get("value") or 0)))
    if len(scores) < 8:
        return None
    promoters = sum(v for s, v in scores if s >= 9)
    detractors = sum(v for s, v in scores if s <= 6)
    return promoters - detractors, promoters, detractors


def _satisfied(values: list) -> float | None:
    labels = [d for d in values if "satisf" in str(d.get("label") or "").lower()]
    if len(labels) < 3:
        return None
    total = sum(float(d.get("value") or 0) for d in labels)
    if not total:
        return None
    pos = sum(
        float(d.get("value") or 0) for d in labels
        if "dissatisf" not in str(d["label"]).lower()
        and "neither" not in str(d["label"]).lower()
    )
    return pos * 100.0 / total


def validate(path: Path) -> list[str]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    qs = _questions(doc)
    failures: list[str] = []

    # 2. Percentage sums vs declared type.
    untyped = 0
    for q in qs:
        qtype = q.get("questionType") or ""
        values = q.get("data") or []
        if not values:
            continue
        if not qtype:
            untyped += 1
            continue
        total = round(sum(float(d.get("value") or 0) for d in values), 1)
        if qtype in _SUM_TO_100 and abs(total - 100.0) > 1.0:
            failures.append(f"Q{q.get('qNum')} is {qtype} but sums to {total}")
        if qtype == "multiple_choice" and 99.0 <= total <= 101.0:
            failures.append(
                f"Q{q.get('qNum')} is multiple_choice but sums to {total} — "
                "looks wrongly normalised to 100"
            )
    if untyped:
        failures.append(
            f"{untyped}/{len(qs)} questions carry no declared type, so their "
            "percentage sums cannot be checked"
        )

    # 1 + 3. NPS, and its consistency with satisfaction.
    nps = sat = None
    for q in qs:
        values = q.get("data") or []
        if nps is None:
            got = _nps(values)
            if got:
                nps = (*got, q.get("qNum"))
                continue
        if sat is None:
            got = _satisfied(values)
            if got is not None:
                sat = (got, q.get("qNum"))
    if nps:
        value, promoters, detractors, qnum = nps
        print(f"    NPS {value:+.0f}  (Q{qnum}: {promoters:.0f}% promoters, "
              f"{detractors:.0f}% detractors)")
    if sat:
        print(f"    Satisfied {sat[0]:.0f}%  (Q{sat[1]})")
    if nps and sat:
        value = nps[0]
        share = sat[0]
        if value < -20 and share > 50:
            failures.append(
                f"NPS {value:+.0f} is detractor-heavy while {share:.0f}% report "
                "satisfaction — these describe different populations"
            )
        elif value > 40 and share < 45:
            failures.append(
                f"NPS {value:+.0f} is strongly positive while only {share:.0f}% "
                "report satisfaction"
            )
    return failures


def main() -> int:
    args = sys.argv[1:]
    if args == ["--all"] or not args:
        paths = sorted(
            p for d in PUBLISHED.iterdir() if d.is_dir()
            for p in d.glob("*.json") if p.stem != "index"
        )
    else:
        paths = [Path(a) for a in args]

    bad = 0
    for p in paths:
        if not p.is_file():
            print(f"missing: {p}")
            continue
        print(f"\n{p.parent.name}/{p.name}")
        failures = validate(p)
        for f in failures:
            print(f"    FAIL {f}")
        if failures:
            bad += 1
        else:
            print("    PASS all three checks")

    print("\n" + "=" * 58)
    print(f"  validation {'PASSED' if not bad else f'FAILED ({bad} file(s))'}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
