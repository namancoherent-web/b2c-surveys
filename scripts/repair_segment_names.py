"""Re-apply the segment-naming rule to already-published surveys.

# change for b2c questionarie — A10

Segment names are labels in ``surveyDefinition.segments``; nothing downstream
keys off their wording. When the naming rule is tightened, a full re-run costs
several minutes per market to change a handful of strings, so this applies the
same ``formalize_segment_name`` the planner path uses, in place.

Usage:
  python scripts/repair_segment_names.py            # every published survey
  python scripts/repair_segment_names.py --dry-run
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

from src.narrative import formalize_segment_name  # noqa: E402

PUBLISHED = ROOT / "published_surveys"


def repair(path: Path, market: str, dry: bool) -> list[tuple[str, str]]:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    defn = doc.get("surveyDefinition") or {}
    segs = defn.get("segments")
    if not isinstance(segs, list):
        return []

    changed: list[tuple[str, str]] = []
    for s in segs:
        if not isinstance(s, dict):
            continue
        old = str(s.get("name") or "")
        new = formalize_segment_name(old, market)
        if new and new != old:
            changed.append((old, new))
            if not dry:
                s["name"] = new
    if changed and not dry:
        path.write_text(
            json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8",
        )
    return changed


def main() -> int:
    dry = "--dry-run" in sys.argv[1:]
    total = 0
    for market_dir in sorted(p for p in PUBLISHED.iterdir() if p.is_dir()):
        market = market_dir.name.replace("-", " ")
        files = sorted(market_dir.glob("*.json")) + [
            p for p in [PUBLISHED / f"{market_dir.name}.json"] if p.is_file()
        ]
        seen: set[tuple[str, str]] = set()
        for f in files:
            for old, new in repair(f, market, dry):
                total += 1
                if (old, new) not in seen:
                    seen.add((old, new))
                    print(f"  {market_dir.name}: {old!r} -> {new!r}")
    verb = "would rename" if dry else "renamed"
    print(f"\n{verb} {total} segment name(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
