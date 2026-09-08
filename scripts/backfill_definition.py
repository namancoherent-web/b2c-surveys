"""Backfill ``surveyDefinition`` into surveys published before it existed.

# change for b2c questionarie

Smartwatches was generated before the assembler emitted the definition block,
so its published JSON has no "what this survey is / whose context this is"
section and the frontend falls back to a generic description.

Everything the block needs is already present in the published payload
(targetCustomer, metadata, sections), so this reconstructs it in place — no
pipeline re-run, no LLM calls, no change to any question or percentage.

Usage:
    python scripts/backfill_definition.py            # all surveys missing it
    python scripts/backfill_definition.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.narrative import TAB_SECTION_IDS  # noqa: E402

_SECTION_COVERS = {
    "profile": ("Consumer Profile", "how they relate to and use the category"),
    "behavior": ("Buying Behavior", "how they shop for and buy it"),
    "preferences": ("Preferences & Expectations", "what they want from the product"),
    "satisfaction": ("Satisfaction & Future Intent",
                     "how they feel about it and what they will do next"),
}

_REGION_LABELS = {
    "global": "Global",
    "north-america": "North America",
    "europe": "Europe",
    "asia-pacific": "Asia Pacific",
    "latin-america": "Latin America",
    "middle-east-africa": "Middle East & Africa",
}


def build_definition(payload: dict) -> dict:
    segment = payload.get("segment") or "the category"
    region_slug = payload.get("region") or "global"
    region = _REGION_LABELS.get(region_slug, region_slug)
    tc = payload.get("targetCustomer") or {}
    meta = payload.get("metadata") or {}
    sections = ((payload.get("frontend") or {}).get("sections") or [])

    inclusion = [str(x) for x in (tc.get("inclusionCriteria") or []) if x]
    exclusions = [str(x) for x in (tc.get("exclusions") or []) if x]

    who = f"Adults in {region} who buy or use {segment} for personal or household use."
    if inclusion:
        who += " Respondents qualify if they: " + "; ".join(inclusion[:5]) + "."
    if exclusions:
        who += " Excluded: " + "; ".join(exclusions[:4]) + "."

    total = sum(len(s.get("questions") or []) for s in sections)
    sample = meta.get("sample_size") or 2500

    return {
        "title": f"{segment} — consumer survey ({region})",
        "category": segment,
        "region": region,
        "audienceType": "B2C consumer",
        "whatThisSurveyIs": (
            f"A business-to-consumer (B2C) market research questionnaire about "
            f"{segment}. Every question asks about {segment} specifically — not "
            "the wider market, not adjacent categories, and not any named brand. "
            f"It is designed to tell a client what buyers in {region} actually do "
            "and want in this category."
        ),
        "whoseContextThisIs": who,
        "whyTheseRespondents": (
            (meta.get("classification_reason") or "").strip()
            or f"{segment} is bought and used by individual consumers."
        ),
        "respondentBehaviours": [str(b) for b in (tc.get("behaviors") or [])][:6],
        "whatIsDeliberatelyNotAsked": [
            "Age or household income — these define who was recruited, not what is "
            "measured, so they appear in the audience definition rather than as questions.",
            "Brand, company or manufacturer names — the survey stays at category level "
            "so findings are not tied to one competitor set.",
            "Where the respondent lives — region is chosen when viewing the results, "
            "so asking it again would waste a question.",
        ],
        "howToReadTheNumbers": (
            f"Percentages are simulated from a persona-weighted panel of {sample:,} "
            "synthetic respondents built from real market evidence, not from "
            "fieldwork. Single-choice, ranking and rating-scale questions sum to "
            "100%. Multi-select questions do not — each option is an independent "
            "share of respondents choosing it. The N shown on each question is the "
            "display base for that item."
        ),
        "structure": {
            "totalQuestions": total,
            "sections": [
                {
                    "id": s.get("id"),
                    "label": _SECTION_COVERS.get(s.get("id"), (s.get("label"), ""))[0],
                    "covers": _SECTION_COVERS.get(s.get("id"), ("", ""))[1],
                    "questions": len(s.get("questions") or []),
                }
                for s in sections
            ],
            "storyline": (
                "Questions run in a planned narrative order: broad and easy first, "
                "specific and sensitive last."
            ),
            "storylineSummary": (meta.get("storyline_summary") or "").strip(),
        },
        "_backfilled": True,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "published_surveys"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    files = sorted(Path(args.src).glob("*/*.json"))
    files = [f for f in files if f.name != "index.json"]
    done = skipped = 0
    for f in files:
        payload = json.loads(f.read_text(encoding="utf-8"))
        if payload.get("surveyDefinition"):
            skipped += 1
            continue
        if not ((payload.get("frontend") or {}).get("sections")):
            continue
        definition = build_definition(payload)
        # Rebuild with the definition first, matching freshly-generated files.
        rebuilt = {"surveyDefinition": definition}
        rebuilt.update(payload)
        rebuilt["surveyDefinition"] = definition
        if not args.dry_run:
            f.write_text(
                json.dumps(rebuilt, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
        print(f"{'(dry) ' if args.dry_run else ''}backfilled {f.relative_to(ROOT)}")
        done += 1
    print(f"\nbackfilled {done}, already had one: {skipped}")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    main()
