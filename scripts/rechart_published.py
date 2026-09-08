"""Re-assign chart kinds on ALREADY-PUBLISHED surveys.

# change for b2c questionarie

Chart selection happens at assembly time, so surveys produced before the richer
ChartKinds existed still carry the original eight. This re-runs the same
least-used-eligible diversification over published JSON in place, so existing
surveys pick up diverging / lollipop / gauge / waffle without regenerating
anything (no LLM calls, no cost, percentages untouched).

Only the ``chart`` field changes. Questions, options and data are left alone.

Usage:
    python scripts/rechart_published.py                 # published_surveys/
    python scripts/rechart_published.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.nodes.assembler import _ALL_CHART_KINDS, _eligible_charts  # noqa: E402

_SCALE_WORDS = re.compile(
    r"(?i)\b(strongly|somewhat|very|extremely|neither|neutral|slightly|"
    r"moderately|dissatisfied|satisfied|agree|disagree|likely|unlikely|"
    r"important|not at all)\b"
)


def infer_type(q: dict) -> str:
    """Recover the answer format from a published frontend question."""
    if q.get("multiSelect"):
        return "multiple_choice"
    opts = [str(o) for o in (q.get("options") or [])]
    chart = (q.get("chart") or "").lower()
    if chart in ("hbar-likert", "vbar-likert", "diverging"):
        return "likert_5" if len(opts) != 7 else "likert_7"
    # A balanced scale: most options carry scale vocabulary.
    if len(opts) in (5, 7):
        hits = sum(1 for o in opts if _SCALE_WORDS.search(o))
        if hits >= max(3, len(opts) - 1):
            return "likert_5" if len(opts) == 5 else "likert_7"
    return "single_choice"


def rechart_payload(payload: dict) -> Counter:
    """Diversify chart kinds per section, in place. Returns the new usage."""
    used: Counter = Counter()
    for section in ((payload.get("frontend") or {}).get("sections") or []):
        for q in section.get("questions") or []:
            qtype = infer_type(q)
            eligible = _eligible_charts(qtype, len(q.get("options") or []))
            pick = min(
                eligible,
                key=lambda c: (
                    used.get(c, 0),
                    _ALL_CHART_KINDS.index(c) if c in _ALL_CHART_KINDS else 99,
                ),
            )
            q["chart"] = pick
            used[pick] += 1
    return used


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "published_surveys"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.is_dir():
        print(f"no such directory: {src}")
        raise SystemExit(1)

    files = sorted(src.glob("*/*.json"))
    files = [f for f in files if f.name != "index.json"]
    if not files:
        print(f"no published survey files under {src}")
        raise SystemExit(1)

    grand: Counter = Counter()
    for f in files:
        payload = json.loads(f.read_text(encoding="utf-8"))
        if not ((payload.get("frontend") or {}).get("sections")):
            continue
        used = rechart_payload(payload)
        grand.update(used)
        if not args.dry_run:
            f.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        mix = ", ".join(f"{k}:{v}" for k, v in sorted(used.items()))
        print(f"{'(dry) ' if args.dry_run else ''}{f.relative_to(ROOT)}  ->  {mix}")

    print("\ntotal chart mix across all files:")
    for k in _ALL_CHART_KINDS:
        print(f"  {k:<12} {grand.get(k, 0)}")
    if args.dry_run:
        print("\n(dry run — nothing written)")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    main()
