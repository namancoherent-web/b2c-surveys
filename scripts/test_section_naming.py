"""Probe the story planner across diverse markets — sections only.

# change for b2c questionarie

Section naming is decided by a single story_planner call, so it can be tested
without running the whole pipeline. This runs the planner over a spread of
category types and prints the sections it invents, so you can see whether a
subscription, a durable, a consumable and a service really do get different
survey structures.

One LLM call per market. No questions generated, nothing published.

Usage:
    python scripts/test_section_naming.py
    python scripts/test_section_naming.py "Pet Insurance" "Electric Scooters"
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import narrative  # noqa: E402
from src.nodes.story_planner import story_planner  # noqa: E402

# A spread of category shapes — each should want a different story.
MARKETS: list[tuple[str, str]] = [
    ("Video Streaming Subscriptions", "recurring digital subscription"),
    ("Running Shoes", "apparel / footwear, fit matters"),
    ("Organic Milk", "everyday grocery consumable"),
    ("Electric Vehicles", "high-consideration durable, long cycle"),
    ("Gym Memberships", "recurring service, physical venue"),
    ("Baby Diapers", "consumable bought by a caregiver for someone else"),
    ("Home Insurance", "financial product, renewal-driven"),
    ("Craft Beer", "impulse / indulgence consumable"),
]

_MKP = {
    "market_maturity": "growing",
    "price_sensitivity": "medium",
    "technology_adoption": "mainstream",
    "evidence_confidence": "medium",
    "consumer_sentiment": "mixed",
    "regional_notes": {"Global": "Broad mainstream consumer base."},
}


def probe(segment: str, note: str) -> dict:
    state = {
        "normalized_segment": segment,
        "region": "Global",
        "market_knowledge_profile": _MKP,
        "persona_catalog": {"personas": []},
        "audience_profile": {"qualifiers": [f"buys or uses {segment}"]},
    }
    return story_planner(state)["survey_blueprint"]


def main() -> None:
    targets = (
        [(s, "") for s in sys.argv[1:]] if len(sys.argv) > 1 else MARKETS
    )
    canon_default = list(narrative.canonical_section_ids())
    summary: list[tuple[str, int, int, bool]] = []

    for segment, note in targets:
        print("=" * 92)
        print(f"{segment}" + (f"   ({note})" if note else ""))
        print("=" * 92)
        try:
            bp = probe(segment, note)
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {type(exc).__name__}: {exc}\n")
            continue

        sections = narrative.sections_for(bp)
        renamed = sum(
            1 for s in sections if s["section_id"] not in canon_default
        )
        total_beats = 0
        for s in sections:
            beats = narrative.beats_for_tab(bp, s["section_id"])
            total_beats += len(beats)
            flag = "  <-- renamed" if s["section_id"] not in canon_default else ""
            print(f"  {s['order']}. {s['label']}{flag}")
            print(f"       id={s['section_id']}   canonical={s['canonical']}   beats={len(beats)}")
            if s.get("remit"):
                print(f"       remit: {s['remit'][:96]}")
            print(f"       {' -> '.join(b['beat_id'] for b in beats)}")
        summary.append((segment, len(sections), renamed, bp.get("source") == "planner"))
        notes = bp.get("adaptation_notes") or []
        if notes:
            print("  adaptation notes:")
            for n in notes[:3]:
                print(f"    - {n[:150]}")
        print()

    print("=" * 92)
    print(f"{'MARKET':<36}{'SECTIONS':>9}{'RENAMED':>9}{'SOURCE':>10}")
    print("-" * 92)
    for segment, n_sec, renamed, planned in summary:
        print(f"{segment:<36}{n_sec:>9}{renamed:>9}{('planner' if planned else 'default'):>10}")
    if summary:
        adapted = sum(1 for _, _, r, _ in summary if r > 0)
        print("-" * 92)
        print(f"{adapted}/{len(summary)} markets produced category-specific section names")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    main()
