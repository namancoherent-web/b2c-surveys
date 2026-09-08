"""Standalone post-processor: condense a full B2C survey JSON down to ~15
high-priority questions, WITHOUT touching the main pipeline.

Scope (per user, 2026-09-07): user wants a short (~15 question) B2C survey
variant, but is worried a fresh short generation would let the LLM pick
random/weak questions instead of the ones that actually matter for a
purchase-decision study. Rather than change question_architect.py's
generation target (which the user explicitly told me NOT to touch — "dont
fuck right now pipeline"), this script runs the pipeline exactly as-is
(already-vetted, deduped, ~30-question output) and SELECTS the highest-
priority ~3 questions per behavioural section from what was already
generated, dropping the rest. Nothing about the pipeline's generation,
detection, or dedup logic changes; this only re-filters its output.

Priority rules per section are a plain-English adaptation of the structure
in the Google-sourced example the user shared (purchase trigger, price
paid, top decision driver, satisfaction, NPS, top switch-driver) -- keeping
OUR simplicity/no-jargon wording, not the example's stiff phrasing.
"""
import json
import re
import sys

PATH = sys.argv[1] if len(sys.argv) > 1 else None
if not PATH:
    print("Usage: python select_top15.py <path-to-survey.json>")
    sys.exit(1)

# Ordered priority patterns per canonical section title. Earlier patterns win
# when multiple questions match; first N matches (by section quota) are kept.
_PRIORITY_BY_SECTION = {
    "Consumer Profile, Ownership & Usage Behavior": [
        r"how often|how many days|typical week",          # usage frequency
        r"use.*(for|most)|mainly use",                    # primary use case
        r"how long.*(had|owned|kept)",                     # tenure/ownership duration
    ],
    "Purchase Journey & Decision Drivers": [
        r"made you (buy|get)|why did you buy",             # purchase trigger
        r"how much.*(pay|spend|cost)",                      # price paid
        r"matters most|mattered most",                      # top decision driver
    ],
    "Product Experience & Satisfaction": [
        r"must.*have|features must",                        # must-have features
        r"well made|good before you buy|tells you",         # quality perception
        r"compromise|give up",                               # price trade-off
    ],
    "Brand / Product Experience & Satisfaction": [   # in case section wasn't renamed
        r"must.*have|features must",
        r"well made|good before you buy|tells you",
        r"compromise|give up",
    ],
    "Unmet Needs, Switching & Future Purchase Intent": [
        r"how (satisfied|happy)",                            # satisfaction
        r"recommend|tell a friend|tell.{0,15}(others|someone)",  # NPS / advocacy
        r"switch|bothers you|problem.*(most|bother)|frustrat",  # top switch/pain driver
    ],
}
_SECTION_QUOTA = 3  # questions kept per behavioural section (Profiling untouched)


def _rank(text: str, patterns: list) -> int:
    low = text.lower()
    for i, pat in enumerate(patterns):
        if re.search(pat, low):
            return i
    return len(patterns)  # unranked -- kept last, only if quota has room


def select_top(section: dict) -> dict:
    if section["title"] == "Profiling":
        return section
    patterns = _PRIORITY_BY_SECTION.get(section["title"])
    if not patterns:
        section["questions"] = section["questions"][:_SECTION_QUOTA]
        return section
    ranked = sorted(
        enumerate(section["questions"]),
        key=lambda iq: (_rank(iq[1]["text"], patterns), iq[0]),
    )
    kept_idx = sorted(i for i, _ in ranked[:_SECTION_QUOTA])
    kept = [section["questions"][i] for i in kept_idx]
    for n, q in enumerate(kept, 1):
        q["id"] = f"Q{n}"
    section["questions"] = kept
    return section


def main():
    with open(PATH, encoding="utf-8") as f:
        data = json.load(f)

    for seg in data["segments"]:
        select_top(seg)

    for section in data["structure"]["sections"]:
        matching = next(
            (s for s in data["segments"] if s["title"] == section["label"]), None
        )
        if matching:
            section["questions"] = len(matching["questions"])
    data["structure"]["totalQuestions"] = sum(
        len(s["questions"]) for s in data["segments"]
    )

    with open(PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Condensed to {data['structure']['totalQuestions']} total questions:")
    for seg in data["segments"]:
        print(f"  {seg['title']}: {len(seg['questions'])}")
        for q in seg["questions"]:
            print(f"    - {q['text']}")


if __name__ == "__main__":
    main()
