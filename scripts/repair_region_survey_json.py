"""Repair region survey JSON: one-region %, diversify charts (8 kinds).

# change for b2c questionarie

Usage:
  python scripts/repair_region_survey_json.py output/organic-milk_north-america.json
  python scripts/repair_region_survey_json.py --all-organic-milk
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.nodes.assembler import (  # noqa: E402
    _ALL_CHART_KINDS,
    _eligible_charts,
    _to_chart_kind,
)
from src.survey_io import (  # noqa: E402
    FRONTEND_SURVEYS_DIR,
    build_region_payload,
    publish_region_file,
    publish_to_frontend,
    slugify,
)

_TYPE_FROM_CHART = {
    "hbar-likert": "likert_5",
    "vbar-likert": "likert_5",
    "hbar-multi": "multiple_choice",
}


def _guess_type(q: dict) -> str:
    if q.get("multiSelect") or q.get("chart") == "hbar-multi":
        return "multiple_choice"
    if q.get("ordered") and q.get("chart") in ("hbar-likert", "vbar-likert"):
        return "likert_5"
    if q.get("chart") in ("hbar-likert", "vbar-likert"):
        return "likert_5"
    return "single_choice"


def diversify_frontend(sections: list, region_slug: str) -> list:
    usage = {k: 0 for k in _ALL_CHART_KINDS}
    out_sections = []
    for sec in sections:
        qs = []
        for q in sec.get("questions") or []:
            q = dict(q)
            qtype = _guess_type(q)
            n_opts = len(q.get("options") or q.get("data") or [])
            eligible = _eligible_charts(qtype, n_opts)
            pick = min(
                eligible,
                key=lambda c: (
                    usage.get(c, 0),
                    _ALL_CHART_KINDS.index(c) if c in _ALL_CHART_KINDS else 99,
                ),
            )
            usage[pick] = usage.get(pick, 0) + 1
            q["chart"] = pick
            # Keep only this region's percentages.
            data = q.get("data") or []
            dbr = q.get("dataByRegion") or {}
            if region_slug in dbr and dbr[region_slug]:
                data = dbr[region_slug]
            q["data"] = data
            q["dataByRegion"] = {region_slug: data}
            if q.get("multiSelect") is None and pick == "hbar-multi":
                q["multiSelect"] = True
            if pick in ("hbar-likert", "vbar-likert", "radar") or qtype in (
                "likert_5", "likert_7", "ranking",
            ):
                q["ordered"] = True
            # change for b2c questionarie — ensure key exists (filled in repair_file)
            if "distributionNote" not in q:
                q["distributionNote"] = ""
            qs.append(q)
        out_sections.append({**sec, "questions": qs})
    return out_sections


def _notes_from_tabs(survey: dict) -> dict[str, str]:
    """Map question id → distribution_note from rich survey.tabs."""
    # change for b2c questionarie
    out: dict[str, str] = {}
    for tab in ((survey.get("survey") or {}).get("tabs") or []):
        for q in tab.get("questions") or []:
            qid = q.get("id")
            note = (q.get("distribution_note") or "").strip()
            if qid and note:
                out[str(qid)] = note
    return out


def repair_file(path: Path) -> dict:
    survey = json.loads(path.read_text(encoding="utf-8"))
    region = (survey.get("survey") or {}).get("region") or ""
    from src.question_plan import match_geographic_region, region_to_slug
    geo = match_geographic_region(region)
    slug = region_to_slug(geo) if geo else "global"

    notes = _notes_from_tabs(survey)

    fe = survey.get("frontend") or {}
    sections = diversify_frontend(fe.get("sections") or [], slug)
    # change for b2c questionarie — copy tab notes onto frontend; leave insight as Top response
    n_notes = 0
    for sec in sections:
        for q in sec.get("questions") or []:
            note = notes.get(str(q.get("id") or ""), "")
            if note:
                q["distributionNote"] = note
                n_notes += 1
            elif not q.get("distributionNote"):
                q["distributionNote"] = ""
            # Keep insight as Top response matching current data
            data = q.get("data") or []
            if data:
                top = max(data, key=lambda d: d.get("value") or 0)
                q["insight"] = (
                    f"Top response: {top.get('label')} ({top.get('value')}%)."
                )
    survey["frontend"] = {"sections": sections}

    # Trim rich survey tabs data_by_region
    for tab in ((survey.get("survey") or {}).get("tabs") or []):
        for q in tab.get("questions") or []:
            dbr = q.get("data_by_region") or {}
            if slug in dbr:
                q["data_by_region"] = {slug: dbr[slug]}
            q["chart_type"] = {
                "hbar": "horizontal_bar",
                "hbar-multi": "horizontal_bar",
                "hbar-likert": "horizontal_bar",
                "vbar": "bar",
                "vbar-likert": "bar",
                "donut": "donut",
                "stacked": "stacked",
                "radar": "radar",
            }.get(
                next(
                    (
                        qq.get("chart")
                        for sec in sections
                        for qq in sec["questions"]
                        if qq.get("id") == q.get("id")
                    ),
                    "vbar",
                ),
                q.get("chart_type") or "bar",
            )

    charts = Counter(
        qq.get("chart")
        for sec in sections
        for qq in sec.get("questions") or []
    )
    meta = (survey.get("survey") or {}).setdefault("metadata", {})
    meta["chart_kinds_used"] = sorted(charts.keys())

    path.write_text(json.dumps(survey, indent=2, ensure_ascii=False), encoding="utf-8")

    # Republish frontend folder
    segment = (survey.get("survey") or {}).get("market_segment") or "survey"
    seg_slug = slugify(segment)
    payload = build_region_payload(
        survey, segment=segment, slug=seg_slug, region_slug=slug,
    )
    published = publish_region_file(seg_slug, slug, payload, segment=segment)
    publish_to_frontend(seg_slug, json.dumps(survey, indent=2, ensure_ascii=False))
    return {
        "path": str(path),
        "region": slug,
        "charts": dict(charts),
        "published": published,
        "n_questions": sum(len(s.get("questions") or []) for s in sections),
        "n_distribution_notes": n_notes,
    }


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] == "--all-organic-milk":
        paths = sorted((ROOT / "output").glob("organic-milk_*.json"))
    else:
        paths = [Path(a) for a in args]
    for p in paths:
        if not p.is_file():
            print("skip missing", p)
            continue
        info = repair_file(p)
        print("repaired", info)


if __name__ == "__main__":
    main()
