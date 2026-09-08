"""Measure how simple the generated questions actually are.

# change for b2c questionarie — A10

The prompt asks for 8-16 word stems that name the category and stand alone. Prompt rules have needed
more than one attempt before, so this reads the real output and reports the
distribution rather than trusting the instruction took.

Usage:
  python scripts/check_question_simplicity.py [output/**/global.json ...]
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("NEON_CONNECTION_STRING", "postgresql://u:p@localhost/db")
os.environ.setdefault("DEEPSEEK_API_KEY", "sk-dummy-not-used")

from src.nodes.validator_critic import (  # noqa: E402
    _is_overcomplicated,
    _stem_word_count,
)


def _sections(doc: dict) -> list:
    """Sections, wherever this file shape keeps them.

    Region payloads are flat; published rich files nest under ``frontend``.
    """
    for holder in (doc, doc.get("frontend") or {}, doc.get("survey") or {}):
        secs = holder.get("sections")
        if secs:
            return secs
    return []


def audit(path: Path) -> dict | None:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    secs = _sections(doc)
    if not secs:
        return None

    # change for b2c questionarie — A11: the category name is needed for the
    # standalone-stem check, so read it off the document.
    segment = (
        doc.get("segment")
        or (doc.get("surveyDefinition") or {}).get("category")
        or (doc.get("surveyDefinition") or {}).get("title")
        or ""
    )

    stems, flagged, screening = [], [], 0
    for sec in secs:
        if "screen" in str(sec.get("id") or sec.get("label") or "").lower():
            screening += len(sec.get("questions") or [])
        for q in sec.get("questions") or []:
            layer = (q.get("questionLayer") or q.get("question_layer") or "").strip()
            if layer in ("screening", "profiling"):
                continue
            text = q.get("question") or q.get("text") or ""
            stems.append((_stem_word_count(text), q.get("qNum"), text))
            why = _is_overcomplicated(text, segment)
            if why:
                flagged.append((q.get("qNum"), why, text))

    if not stems:
        return None
    counts = sorted(n for n, _, _ in stems)
    mid = len(counts) // 2
    return {
        "path": path,
        "n": len(stems),
        "median": counts[mid] if len(counts) % 2 else (counts[mid - 1] + counts[mid]) / 2,
        "max": counts[-1],
        "over_18": sum(1 for n in counts if n > 18),
        "under_6": sum(1 for n in counts if n < 6),
        "in_band": sum(1 for n in counts if 8 <= n <= 16),
        "sections": [s.get("label") for s in secs],
        "screening": screening,
        "flagged": flagged,
        "longest": max(stems)[2],
    }


def main() -> int:
    args = sys.argv[1:]
    paths = [Path(a).resolve() for a in args] if args else sorted(
        (ROOT / "output").rglob("*global*.json")
    )
    reports = [r for r in (audit(p) for p in paths) if r]
    if not reports:
        print("no survey JSON found")
        return 1

    bad = 0
    for r in reports:
        try:
            rel = r["path"].relative_to(ROOT)
        except ValueError:
            rel = r["path"]
        band = 100 * r["in_band"] // r["n"]
        print(f"\n{rel}")
        print(f"  questions {r['n']}   median stem {r['median']} words   "
              f"longest {r['max']}   in 8-16 band {band}%")
        print(f"  screening questions: {r['screening']}")
        print(f"  sections: {' | '.join(str(s) for s in r['sections'])}")
        if r["screening"]:
            bad += 1
            print("  FAIL screening section still present")
        if r["flagged"]:
            bad += 1
            print(f"  FAIL {len(r['flagged'])} complex stems:")
            for qnum, why, text in r["flagged"][:6]:
                print(f"    Q{qnum}: {why}\n         {text}")
        else:
            print("  PASS no stem trips the complexity check")

    print("\n" + "=" * 58)
    print("  simplicity audit " + ("PASSED" if not bad else f"FAILED ({bad})"))
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
