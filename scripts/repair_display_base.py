"""Collapse per-question display bases to one base per survey.

# change for b2c questionarie — A16

The generator now derives a single display base per survey, because the
instrument has no routing: every respondent sees every question, so the base
cannot move between them. Files published before that change carry a different
N on each question (310, 390, 470 …), which an analyst would immediately
question and which nothing in the document can explain.

N is display-only metadata — no percentage, chart or scoring weight depends on
it — so rewriting it in place is equivalent to regenerating, at a fraction of
the cost. The survey-level metadata N is the authority.

Usage:
  python scripts/repair_display_base.py            # every published survey
  python scripts/repair_display_base.py --dry-run
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = ROOT / "published_surveys"


def repair(path: Path, dry: bool) -> tuple[int, int] | None:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    sections = (doc.get("frontend") or {}).get("sections") or []
    questions = [q for s in sections for q in (s.get("questions") or [])]
    if not questions:
        return None

    bases = {q.get("N") for q in questions if q.get("N") is not None}
    if len(bases) <= 1:
        return (0, len(questions))

    # Prefer the survey-level base; fall back to the most common per-question one.
    base = (doc.get("metadata") or {}).get("N")
    if base is None:
        base = max(sorted(bases), key=lambda b: sum(1 for q in questions if q.get("N") == b))

    changed = 0
    for q in questions:
        if q.get("N") != base:
            changed += 1
            if not dry:
                q["N"] = int(base)
    if changed and not dry:
        path.write_text(
            json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8",
        )
    return (changed, len(questions))


def main() -> int:
    dry = "--dry-run" in sys.argv[1:]
    total = 0
    for market in sorted(p for p in PUBLISHED.iterdir() if p.is_dir()):
        for f in sorted(market.glob("*.json")):
            if f.stem == "index":
                continue
            got = repair(f, dry)
            if got and got[0]:
                total += got[0]
                print(f"  {market.name}/{f.name}: {got[0]}/{got[1]} questions "
                      f"aligned to one base")
    verb = "would align" if dry else "aligned"
    print(f"\n{verb} {total} question base(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
