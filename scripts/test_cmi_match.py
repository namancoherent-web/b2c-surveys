"""Domain matcher regression tests.

# change for b2c questionarie

Run: python scripts/test_cmi_match.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cmi_frameworks import CANONICAL_IDS  # noqa: E402
from src.cmi_match import match_domain, must_cover_topics, section_labels  # noqa: E402

# (market, expected domain fragment, expected sub-sector fragment or None)
CASES = [
    ("global running shoes market", "Consumer Goods", "Footwear"),
    ("global organic milk market", "Food", "Milk"),
    ("global smart watch market", "Smart", "Wearable"),
    ("global electric vehicle market", "Automotive", "Electric Vehicle"),
    ("global video streaming subscriptions market", "Consumer Services", "Streaming"),
    ("global pet food market", "Food", "Pet Food"),
    ("global skincare serums market", "Consumer Goods", "Skin Care"),
    ("global bottled water market", "Food", None),
    ("global mattresses market", "Consumer Goods", "Bedding"),
    ("global air fryer market", "Consumer Goods", "Kitchenware"),
    ("global athletic apparel market", "Consumer Goods", "Apparel"),
    # Nothing in the taxonomy — must still return a usable framework.
    ("global widget flange market", "Consumer Goods", None),
]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    failures = 0

    for market, want_domain, want_sub in CASES:
        m = match_domain(market)
        ok = want_domain.split()[0] in m["domain"]
        if want_sub:
            ok = ok and want_sub.lower() in m["sub_sector"].lower()
        if not ok:
            failures += 1
        flag = "OK  " if ok else "FAIL"
        print(f"{flag} {market[:44]:46} {m['domain'][:22]:23} {m['sub_sector'][:26]}")

    # Every match must yield exactly the four canonical sections, in arc order.
    for market, _, _ in CASES:
        m = match_domain(market)
        labels = section_labels(m)
        if [cid for cid, _ in labels] != list(CANONICAL_IDS):
            print(f"FAIL {market}: sections not canonical/ordered -> {labels}")
            failures += 1
        if any(not lbl.strip() for _, lbl in labels):
            print(f"FAIL {market}: empty section label")
            failures += 1

    # A sub-sector match should carry seed topics for the architect.
    m = match_domain("global running shoes market")
    topics = must_cover_topics(m)
    if not topics:
        print("FAIL running shoes: no must-cover topics from sub-sector attributes")
        failures += 1
    else:
        print(f"\nseed topics for Footwear: "
              f"{ {k: v[:2] for k, v in topics.items()} }")

    print(f"\nFAILURES: {failures}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
