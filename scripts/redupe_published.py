"""Re-apply the uniqueness gate to ALREADY-PUBLISHED surveys.

# change for b2c questionarie

Deduplication happens at assembly time, so surveys published before a rule was
added keep their duplicates. This re-runs the current gate over published JSON
and renumbers what survives — no pipeline re-run, no LLM calls, no change to any
question's wording or percentages.

Usage:
    python scripts/redupe_published.py --dry-run
    python scripts/redupe_published.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.nodes.assembler import (  # noqa: E402
    _SAME_BEAT_OPTIONS,
    _SAME_BEAT_TEXT,
    _SAME_OPTIONS_JACCARD,
    _REWORD_TEXT_FLOOR,
    _option_token_overlap,
)
from src.question_similarity import (  # noqa: E402
    LOOSE_JACCARD,
    is_near_exact,
    norm_alnum,
    option_jaccard,
    token_set_jaccard,
)


def _clash_reason(q: dict, prior: dict, segment: str) -> str | None:
    """Why ``q`` repeats ``prior``, or None."""
    a, b = q.get("question") or "", prior.get("question") or ""
    oa, ob = q.get("options") or [], prior.get("options") or []
    if norm_alnum(a) and norm_alnum(a) == norm_alnum(b):
        return "identical text"
    if is_near_exact(a, b, oa, ob, segment=segment):
        return "near-exact"
    if token_set_jaccard(a, b, segment) >= LOOSE_JACCARD:
        return "loose text overlap"
    same_type = q.get("multiSelect") == prior.get("multiSelect")
    if (
        len(oa) >= 3 and same_type
        and option_jaccard(oa, ob) >= _SAME_OPTIONS_JACCARD
        and token_set_jaccard(a, b, segment) >= _REWORD_TEXT_FLOOR
    ):
        return "same options, reworded"
    beat = q.get("beatId") or q.get("beat_id")
    p_beat = prior.get("beatId") or prior.get("beat_id")
    if (
        beat and beat == p_beat
        and _option_token_overlap(oa, ob, segment) >= _SAME_BEAT_OPTIONS
        and token_set_jaccard(a, b, segment) >= _SAME_BEAT_TEXT
    ):
        return f"same beat '{beat}' restated"
    return None


def redupe(payload: dict) -> list[str]:
    """Drop repeats across the whole survey and renumber. Returns notes."""
    segment = payload.get("segment") or ""
    sections = ((payload.get("frontend") or {}).get("sections") or [])
    kept_all: list[dict] = []
    notes: list[str] = []

    for section in sections:
        survivors = []
        for q in section.get("questions") or []:
            clash = None
            reason = None
            for prior in kept_all:
                reason = _clash_reason(q, prior, segment)
                if reason:
                    clash = prior
                    break
            if clash is not None:
                notes.append(
                    f"Q{q.get('qNum')} dropped ({reason}, repeats Q{clash.get('qNum')}): "
                    f"{(q.get('question') or '')[:66]}"
                )
                continue
            survivors.append(q)
            kept_all.append(q)
        section["questions"] = survivors

    # Renumber qNum / id / narrativeOrder so the published survey stays dense.
    n = 0
    for section in sections:
        for i, q in enumerate(section["questions"], start=1):
            n += 1
            q["qNum"] = n
            q["id"] = f"q{n}"
            q["narrativeOrder"] = i
            q["funnelPosition"] = i

    meta = payload.setdefault("metadata", {})
    meta["total_questions"] = n
    if notes:
        warnings = list(meta.get("warnings") or [])
        warnings.append(f"{len(notes)} repeated question(s) removed in re-dedup.")
        warnings.extend(notes[:10])
        meta["warnings"] = warnings
    definition = payload.get("surveyDefinition") or {}
    if definition.get("structure"):
        definition["structure"]["totalQuestions"] = n
        for s in definition["structure"].get("sections") or []:
            match = next((x for x in sections if x.get("id") == s.get("id")), None)
            if match:
                s["questions"] = len(match["questions"])
    return notes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "published_surveys"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    files = [f for f in sorted(Path(args.src).glob("*/*.json")) if f.name != "index.json"]
    total_dropped = 0
    for f in files:
        payload = json.loads(f.read_text(encoding="utf-8"))
        if not ((payload.get("frontend") or {}).get("sections")):
            continue
        before = sum(
            len(s.get("questions") or [])
            for s in payload["frontend"]["sections"]
        )
        notes = redupe(payload)
        after = before - len(notes)
        total_dropped += len(notes)
        if notes:
            print(f"{'(dry) ' if args.dry_run else ''}{f.relative_to(ROOT)}  {before} -> {after}")
            for note in notes:
                print(f"    - {note}")
        if notes and not args.dry_run:
            f.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
    print(f"\n{total_dropped} repeated question(s) across {len(files)} file(s)")
    if args.dry_run:
        print("(dry run — nothing written)")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    main()
