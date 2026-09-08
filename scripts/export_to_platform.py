"""Transform published B2C surveys into the CMI platform's ci_b2c_survey shape.

# change for b2c questionarie

The platform stores surveys as ONE ROW PER (region × market) in a table keyed by
a unique geo-prefixed snake_case ``market_slug``, with the whole payload in a
``jsonb`` column. This mirrors ``public.ci_b2b_survey`` exactly — same columns,
same nested JSON contract — so the B2C data drops into the identical pipeline.

Contract (verified against all 26 live ci_b2b_survey rows):

    survey (jsonb)
      note, module, sector, industry, grounded, web_researched,
      sample_size, analyst_name, analyst_title, generated_on,
      authority_sources[], data_status{}, segments[4]

    segments[i]  {id:int, title:str, questions:[...]}
    questions[j] {n:int, id:"Q1", text:str, type:str, options:[...],
                  sources:[], key_insight:str, confidence:str}
    options[k]   {label:str, pct:float, source:str}

Usage:
    python scripts/export_to_platform.py                 # write JSON only
    python scripts/export_to_platform.py --push          # + upsert into Neon
    python scripts/export_to_platform.py --slug smartwatches
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.question_plan import PUBLISH_REGION_SLUGS  # noqa: E402

OUT_DIR = ROOT / "output" / "platform_payloads"
TABLE = "public.ci_b2c_survey"

# Our answer format -> the platform's two accepted values.
_TYPE_MAP = {
    "multiple_choice": "multi_select",
    "single_choice": "single_select",
    "ranking": "single_select",
    "likert_5": "single_select",
    "likert_7": "single_select",
}

# Frontend section id -> the platform's segment ordinal.
_SEGMENT_ORDER = ["profile", "behavior", "preferences", "satisfaction"]


def snake(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")
    return re.sub(r"_+", "_", s)


def market_slug(segment: str, region_slug: str) -> str:
    """``europe_smartwatches_market`` — matches the live B2B convention."""
    base = snake(segment)
    if not base.endswith("_market"):
        base = f"{base}_market"
    return f"{snake(region_slug)}_{base}"


def market_name(segment: str, region_slug: str) -> str:
    label = {
        "global": "Global",
        "north-america": "North America",
        "europe": "Europe",
        "asia-pacific": "Asia Pacific",
        "latin-america": "Latin America",
        "middle-east-africa": "Middle East & Africa",
    }.get(region_slug, region_slug.replace("-", " ").title())
    seg = (segment or "").strip()
    if not seg.lower().endswith("market"):
        seg = f"{seg} Market"
    return f"{label} {seg}"


def _question_type(q: dict) -> str:
    if q.get("multiSelect"):
        return "multi_select"
    return _TYPE_MAP.get(q.get("type") or "", "single_select")


def _key_insight(q: dict) -> str:
    """Platform's key_insight = the headline finding, plus why it looks like that."""
    insight = (q.get("insight") or "").strip()
    note = (q.get("distributionNote") or "").strip()
    if insight and note:
        return f"{insight} {note}"
    return insight or note


def build_survey_json(payload: dict, region_slug: str) -> dict:
    """Our published region payload -> the platform's ``survey`` jsonb."""
    segment = payload.get("segment") or ""
    meta = payload.get("metadata") or {}
    definition = payload.get("surveyDefinition") or {}
    sections = ((payload.get("frontend") or {}).get("sections") or [])

    by_id = {s.get("id"): s for s in sections}
    ordered = [by_id[s] for s in _SEGMENT_ORDER if s in by_id]
    ordered += [s for s in sections if s.get("id") not in _SEGMENT_ORDER]

    segments = []
    qcount = 0
    for i, sec in enumerate(ordered, start=1):
        questions = []
        for q in sec.get("questions") or []:
            qcount += 1
            options = [
                {
                    "label": d.get("label"),
                    "pct": round(float(d.get("value") or 0), 2),
                    # Honest provenance — these are simulated, not fielded.
                    "source": "persona-weighted simulation (see data_status)",
                }
                for d in (q.get("data") or [])
            ]
            questions.append({
                "n": q.get("N") or 0,
                "id": f"Q{qcount}",
                "text": q.get("question") or "",
                "type": _question_type(q),
                "options": options,
                "sources": [],
                "key_insight": _key_insight(q),
                "confidence": "medium",
                # change for b2c questionarie — storyline provenance, additive
                # so the platform can ignore it without breaking.
                "story_beat": q.get("beatTitle") or "",
                "story_order": q.get("narrativeOrder") or q.get("qNum"),
            })
        segments.append({
            "id": i,
            "title": sec.get("label") or sec.get("id"),
            "questions": questions,
        })

    warnings = meta.get("warnings") or []
    return {
        "note": (
            "SURVEY INSTRUMENT — questions, ordering and methodology are "
            "research-grade. Percentage distributions are SIMULATED from a "
            "persona-weighted synthetic panel built on CMI report evidence and "
            "web research; they are NOT responses from real people. Replace with "
            "primary data after fielding this instrument."
        ),
        "module": "Consumer Survey (B2C)",
        "sector": snake(definition.get("category") or segment),
        "industry": market_name(segment, region_slug),
        "grounded": not bool(meta.get("grounding_thin")),
        "web_researched": True,
        "sample_size": meta.get("sample_size") or 2500,
        "analyst_name": "CMI Survey Agent",
        "analyst_title": "Automated consumer-insights pipeline",
        "generated_on": (meta.get("current_date") or str(date.today()))[:10],
        "authority_sources": [],
        "data_status": {
            "sample_size": f"{meta.get('sample_size') or 2500} simulated respondents (not yet fielded)",
            "market_facts": "verified — CMI proprietary database plus live web research",
            "distributions": "simulated estimates — pending primary survey fieldwork",
        },
        # The questions themselves — same 4-segment contract as ci_b2b_survey.
        "segments": segments,
        # change for b2c questionarie — additive B2C-only context
        "survey_definition": definition,
        "target_customer": payload.get("targetCustomer") or {},
        "storyline": meta.get("storyline") or {},
        "region_slug": region_slug,
        "total_questions": qcount,
        "warnings": warnings,
    }


def collect(surveys_dir: Path, only_slug: str | None = None) -> list[dict]:
    rows = []
    for folder in sorted(p for p in surveys_dir.iterdir() if p.is_dir()):
        if only_slug and folder.name != only_slug:
            continue
        for region in PUBLISH_REGION_SLUGS:
            f = folder / f"{region}.json"
            if not f.is_file():
                continue
            payload = json.loads(f.read_text(encoding="utf-8"))
            segment = payload.get("segment") or folder.name
            rows.append({
                "market_slug": market_slug(segment, region),
                "market_name": market_name(segment, region),
                "survey": build_survey_json(payload, region),
                "generated_on": ((payload.get("metadata") or {}).get("current_date")
                                 or str(date.today()))[:10],
                "_source": str(f.relative_to(ROOT)),
            })
    return rows


def upsert(rows: list[dict]) -> int:
    """Idempotent upsert on the unique market_slug — safe to re-run."""
    from psycopg2.extras import Json

    from src import db

    conn = db.get_connection()
    n = 0
    try:
        with conn:
            with conn.cursor() as cur:
                for r in rows:
                    cur.execute(
                        f"""
                        INSERT INTO {TABLE}
                            (market_slug, market_name, survey, generated_on)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (market_slug) DO UPDATE SET
                            market_name  = EXCLUDED.market_name,
                            survey       = EXCLUDED.survey,
                            generated_on = EXCLUDED.generated_on,
                            updated_at   = now()
                        """,
                        (r["market_slug"], r["market_name"],
                         Json(r["survey"]), r["generated_on"]),
                    )
                    n += 1
    finally:
        conn.close()
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true", help="upsert into ci_b2c_survey")
    ap.add_argument("--slug", default=None, help="only this survey folder")
    ap.add_argument("--surveys-dir", default=None)
    args = ap.parse_args()

    surveys_dir = Path(args.surveys_dir) if args.surveys_dir else (
        ROOT / "published_surveys"
    )
    if not surveys_dir.is_dir():
        print(f"no surveys directory: {surveys_dir}")
        raise SystemExit(1)

    rows = collect(surveys_dir, args.slug)
    if not rows:
        print("no published surveys found")
        raise SystemExit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for r in rows:
        out = OUT_DIR / f"{r['market_slug']}.json"
        out.write_text(
            json.dumps(
                {k: v for k, v in r.items() if not k.startswith("_")},
                indent=2, ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    bundle = OUT_DIR / "_all_rows.json"
    bundle.write_text(
        json.dumps([{k: v for k, v in r.items() if not k.startswith("_")}
                    for r in rows], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"{len(rows)} row(s) -> {OUT_DIR}")
    for r in rows:
        q = r["survey"]["total_questions"]
        print(f"  {r['market_slug']:<48} {q:>3} questions   ({r['_source']})")
    print(f"\nbundle: {bundle}")

    if args.push:
        n = upsert(rows)
        print(f"\nupserted {n} row(s) into {TABLE}")
    else:
        print(f"\n(dry run — re-run with --push to upsert into {TABLE})")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    main()
