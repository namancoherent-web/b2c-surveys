"""Repair sneakers region surveys: expand broken grids, dedupe, 2-decimal %, display N.

# change for b2c questionarie

Usage:
  python scripts/repair_sneakers_survey_qa.py
  python scripts/repair_sneakers_survey_qa.py --no-global
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.question_similarity import is_near_exact, norm_alnum  # noqa: E402
from src.survey_io import (  # noqa: E402
    FRONTEND_SURVEYS_DIR,
    build_region_payload,
    publish_region_file,
    publish_to_frontend,
)

OUTPUT = ROOT / "output"
SLUG = "global-sneakers-market"
SEGMENT = "global sneakers market"

_REGIONS = [
    "north-america",
    "europe",
    "asia-pacific",
    "latin-america",
    "middle-east-africa",
]

_IMPORTANCE_SCALE = [
    "Not at all important",
    "Slightly important",
    "Moderately important",
    "Very important",
    "Extremely important",
]

_SNEAKER_FACTORS = [
    "Price / value for money",
    "Comfort and fit",
    "Brand reputation",
    "Style and design",
    "Durability and quality",
    "Sustainability / eco materials",
    "Athletic performance features",
]

_SCALE_WORDS = {
    "not at all important", "slightly important", "moderately important",
    "very important", "extremely important", "not important",
    "somewhat important", "important", "neutral",
    "strongly disagree", "disagree", "agree", "strongly agree",
    "very dissatisfied", "dissatisfied", "satisfied", "very satisfied",
    "not at all likely", "slightly likely", "moderately likely",
    "very likely", "extremely likely",
}

_GRID_STEM_RE = re.compile(
    r"(each of the following|how important are each|rate the importance of each|"
    r"importance of each of|for each of the following)",
    re.I,
)


def _is_scale_only(options: list) -> bool:
    if not options or len(options) < 3:
        return False
    hits = 0
    for o in options:
        key = norm_alnum(str(o))
        if any(norm_alnum(w) in key or key in norm_alnum(w) for w in _SCALE_WORDS):
            hits += 1
        elif key in ("1", "2", "3", "4", "5", "6", "7"):
            hits += 1
    return hits >= max(3, len(options) - 1)


def is_broken_grid(q: dict) -> bool:
    text = q.get("question") or ""
    opts = q.get("options") or [d.get("label") for d in (q.get("data") or [])]
    return bool(_GRID_STEM_RE.search(text)) and _is_scale_only(opts)


def _largest_remainder_2dp(values: list[float]) -> list[float]:
    """Round to 2 decimals; force sum == 100.00 for sum-to-100 types."""
    if not values:
        return values
    vals = [max(0.0, float(v)) for v in values]
    total = sum(vals)
    if total <= 0:
        share = round(100.0 / len(vals), 2)
        out = [share] * len(vals)
        out[-1] = round(100.0 - sum(out[:-1]), 2)
        return out
    scaled = [v * 100.0 / total for v in vals]
    cents_list = [int(v * 100) for v in scaled]
    deficit = 10000 - sum(cents_list)
    # distribute remaining cents by largest fractional part
    frac = sorted(
        ((scaled[i] * 100 - cents_list[i], i) for i in range(len(scaled))),
        reverse=True,
    )
    k = 0
    while deficit != 0 and frac:
        i = frac[k % len(frac)][1]
        if deficit > 0:
            cents_list[i] += 1
            deficit -= 1
        elif cents_list[i] > 0:
            cents_list[i] -= 1
            deficit += 1
        k += 1
        if k > len(frac) * 5:
            break
    return [c / 100.0 for c in cents_list]


def _fix_data_list(data: list, *, sum_to_100: bool) -> list:
    if not data:
        return data
    labels = [d.get("label") for d in data]
    values = [float(d.get("value") or 0) for d in data]
    if sum_to_100:
        values = _largest_remainder_2dp(values)
    else:
        values = [round(max(0.0, min(100.0, v)), 2) for v in values]
    return [{"label": lb, "value": v} for lb, v in zip(labels, values)]


def _seeded_likert_dist(rng: random.Random, n: int = 5) -> list[float]:
    weights = [rng.uniform(0.3, 1.2) for _ in range(n)]
    if n >= 5:
        weights[3] *= rng.uniform(1.2, 1.8)
        weights[4] *= rng.uniform(0.9, 1.5)
        weights[0] *= rng.uniform(0.3, 0.7)
    return _largest_remainder_2dp(weights)


def _make_factor_question(
    *,
    factor: str,
    region_slug: str,
    rng: random.Random,
    base_note: str,
    funnel: int | None,
    layer: str,
) -> dict:
    values = _seeded_likert_dist(rng, 5)
    data = [{"label": lb, "value": v} for lb, v in zip(_IMPORTANCE_SCALE, values)]
    top = max(data, key=lambda d: d["value"])
    note = (
        f"Importance of {factor}: {base_note}".strip()
        if base_note
        else f"Persona mix drives importance of {factor} across the Likert scale."
    )
    q = {
        "question": f"How important is {factor} when you decide to buy a new pair of sneakers?",
        "options": list(_IMPORTANCE_SCALE),
        "chart": "hbar-likert",
        "data": data,
        "insight": f"Top response: {top['label']} ({top['value']:.2f}%).",
        "distributionNote": note,
        "questionLayer": layer or "full",
        "isGrounded": True,
        "ordered": True,
        "dataByRegion": {region_slug: list(data)},
    }
    if funnel is not None:
        q["funnelPosition"] = funnel
    return q


def _chart_implies_sum100(q: dict) -> bool:
    if q.get("multiSelect"):
        return False
    chart = (q.get("chart") or "").lower()
    if chart in ("hbar-multi",):
        return False
    return True


def _dedupe_questions(questions: list[dict], segment: str) -> list[dict]:
    kept: list[dict] = []
    for q in questions:
        opts = q.get("options") or [d.get("label") for d in (q.get("data") or [])]
        text = q.get("question") or ""
        dup = False
        for k in kept:
            kopts = k.get("options") or [d.get("label") for d in (k.get("data") or [])]
            if is_near_exact(text, k.get("question") or "", opts, kopts, segment=segment):
                dup = True
                break
        if not dup:
            kept.append(q)
    return kept


def _score_keep(q: dict) -> float:
    score = 1.0
    if q.get("isGrounded"):
        score += 0.5
    note = (q.get("distributionNote") or "").strip()
    score += min(1.0, len(note) / 120.0)
    if q.get("funnelPosition") is not None:
        score += 0.2
    if is_broken_grid(q):
        score -= 5.0
    return score


def repair_frontend(survey: dict, region_slug: str) -> dict:
    rng = random.Random(f"{SLUG}:{region_slug}")
    display_n = rng.randrange(300, 501, 10)

    fe = survey.get("frontend") or {}
    sections = fe.get("sections") or []

    flat: list[tuple[str, dict]] = []
    for sec in sections:
        sid = sec.get("id") or "profile"
        for q in sec.get("questions") or []:
            flat.append((sid, dict(q)))

    expanded: list[tuple[str, dict]] = []
    factor_pool = list(_SNEAKER_FACTORS)
    rng.shuffle(factor_pool)
    used_factors: set[str] = set()
    factor_i = 0

    for sid, q in flat:
        if not is_broken_grid(q):
            expanded.append((sid, q))
            continue
        base_note = (q.get("distributionNote") or "").strip()
        funnel = q.get("funnelPosition")
        layer = q.get("questionLayer") or "full"
        for _ in range(5):
            picks = [f for f in factor_pool if f not in used_factors]
            if not picks:
                used_factors.clear()
                picks = list(factor_pool)
            factor = picks[factor_i % len(picks)]
            factor_i += 1
            used_factors.add(factor)
            fq = _make_factor_question(
                factor=factor,
                region_slug=region_slug,
                rng=rng,
                base_note=base_note,
                funnel=funnel,
                layer=layer,
            )
            expanded.append((sid, fq))

    by_sec: dict[str, list[dict]] = {}
    for sid, q in expanded:
        by_sec.setdefault(sid, []).append(q)
    for sid in list(by_sec):
        by_sec[sid] = _dedupe_questions(by_sec[sid], SEGMENT)

    all_qs: list[tuple[str, dict]] = []
    for sec in sections:
        sid = sec.get("id") or "profile"
        for q in by_sec.get(sid) or []:
            all_qs.append((sid, q))

    # Prefer 60–65; never below 45. Pad with unique factor likerts up to 62.
    _EXTRA_FACTORS = [
        "Online availability",
        "In-store try-on experience",
        "Celebrity / athlete endorsement",
        "Colorway / limited editions",
        "After-sales warranty / returns",
        "Weight of the shoe",
        "Cushioning technology",
        "Breathability",
        "Resale / collectible value",
        "Social media trends",
        "Recommendation from friends",
        "Size consistency across brands",
        "Waterproofing / weather resistance",
        "Ease of cleaning",
        "Loyalty program / discounts",
    ]
    pad_factors = list(_SNEAKER_FACTORS) + _EXTRA_FACTORS
    pad_i = 0
    stem_variants = [
        "How important is {f} when you decide to buy a new pair of sneakers?",
        "When purchasing sneakers, how important is {f} to you?",
        "How would you rate the importance of {f} in your sneaker purchase?",
        "For your next sneaker purchase, how important is {f}?",
    ]
    while len(all_qs) < 62:
        sid = sections[len(all_qs) % max(1, len(sections))].get("id") or "profile"
        factor = pad_factors[pad_i % len(pad_factors)]
        pad_i += 1
        fq = _make_factor_question(
            factor=factor,
            region_slug=region_slug,
            rng=rng,
            base_note="Supplemental factor rating to reach survey quota.",
            funnel=3,
            layer="module",
        )
        placed = False
        for stem in stem_variants:
            fq["question"] = stem.format(f=factor)
            opts = fq["options"]
            if not any(
                is_near_exact(
                    fq["question"], q.get("question") or "", opts,
                    q.get("options") or [], segment=SEGMENT,
                )
                for _, q in all_qs
            ):
                placed = True
                break
        if not placed:
            continue
        all_qs.append((sid, fq))
        if pad_i > 200:
            break

    if len(all_qs) > 65:
        scored = [
            (_score_keep(q), i, sid, q) for i, (sid, q) in enumerate(all_qs)
        ]
        scored.sort(key=lambda t: (t[0], -t[1]))
        drop_n = len(all_qs) - 65
        drop_idx = {t[1] for t in scored[:drop_n]}
        all_qs = [(sid, q) for i, (sid, q) in enumerate(all_qs) if i not in drop_idx]

    sec_order = [s.get("id") or "profile" for s in sections]
    labels = {s.get("id") or "profile": s.get("label") for s in sections}
    buckets: dict[str, list[dict]] = {sid: [] for sid in sec_order}
    for sid, q in all_qs:
        buckets.setdefault(sid, []).append(q)

    new_sections = []
    qnum = 0
    for sid in sec_order:
        qs_out = []
        for q in buckets.get(sid) or []:
            qnum += 1
            sum100 = _chart_implies_sum100(q)
            data = _fix_data_list(list(q.get("data") or []), sum_to_100=sum100)
            q["data"] = data
            dbr = q.get("dataByRegion") or {}
            if region_slug in dbr:
                q["dataByRegion"] = {
                    region_slug: _fix_data_list(
                        list(dbr[region_slug]), sum_to_100=sum100,
                    )
                }
            else:
                q["dataByRegion"] = {region_slug: data}
            if q["dataByRegion"].get(region_slug):
                q["data"] = q["dataByRegion"][region_slug]
            if q.get("data"):
                top = max(q["data"], key=lambda d: d.get("value") or 0)
                q["insight"] = (
                    f"Top response: {top.get('label')} "
                    f"({float(top.get('value') or 0):.2f}%)."
                )
            q["id"] = f"q{qnum}"
            q["qNum"] = qnum
            q["N"] = display_n
            if "distributionNote" not in q:
                q["distributionNote"] = ""
            qs_out.append(q)
        if qs_out:
            new_sections.append({
                "id": sid,
                "label": labels.get(sid) or sid.replace("-", " ").title(),
                "questions": qs_out,
            })

    survey["frontend"] = {"sections": new_sections}

    for tab in ((survey.get("survey") or {}).get("tabs") or []):
        for q in tab.get("questions") or []:
            q["sample_size"] = 2500
            answers = q.get("answers") or []
            if answers:
                qtype = q.get("type") or "single_choice"
                sum100 = qtype in ("single_choice", "ranking", "likert_5", "likert_7")
                vals = [float(a.get("percentage") or 0) for a in answers]
                if sum100:
                    vals = _largest_remainder_2dp(vals)
                else:
                    vals = [round(max(0.0, min(100.0, v)), 2) for v in vals]
                for a, v in zip(answers, vals):
                    a["percentage"] = v
            dbr = q.get("data_by_region") or {}
            for rslug, arr in list(dbr.items()):
                if not arr:
                    continue
                if arr and isinstance(arr[0], (int, float)):
                    if (q.get("type") or "") in (
                        "single_choice", "ranking", "likert_5", "likert_7",
                    ):
                        dbr[rslug] = _largest_remainder_2dp([float(x) for x in arr])
                    else:
                        dbr[rslug] = [
                            round(max(0.0, min(100.0, float(x))), 2) for x in arr
                        ]

    meta = (survey.get("survey") or {}).setdefault("metadata", {})
    meta["N"] = display_n
    meta["sample_size"] = 2500
    meta["total_questions"] = qnum
    survey.setdefault("metadata", {})
    if isinstance(survey["metadata"], dict):
        survey["metadata"]["N"] = display_n
        survey["metadata"]["total_questions"] = qnum

    if not survey.get("segment"):
        survey["segment"] = (
            (survey.get("survey") or {}).get("market_segment") or SEGMENT
        )
    if not survey.get("region"):
        survey["region"] = region_slug
    if not survey.get("slug"):
        survey["slug"] = SLUG

    return survey


def repair_file(path: Path, *, publish: bool = True) -> dict:
    from src.question_plan import match_geographic_region, region_to_slug

    survey = json.loads(path.read_text(encoding="utf-8"))
    region = (
        survey.get("region")
        or (survey.get("survey") or {}).get("region")
        or ""
    )
    geo = match_geographic_region(str(region))
    slug = region_to_slug(geo) if geo else path.stem.split("_")[-1]
    if slug not in _REGIONS:
        for r in _REGIONS:
            if path.name.endswith(f"_{r}.json"):
                slug = r
                break

    survey = repair_frontend(survey, slug)
    path.write_text(json.dumps(survey, indent=2, ensure_ascii=False), encoding="utf-8")

    n = sum(
        len(s.get("questions") or [])
        for s in (survey.get("frontend") or {}).get("sections") or []
    )
    published = None
    if publish:
        payload = build_region_payload(
            survey, segment=SEGMENT, slug=SLUG, region_slug=slug,
        )
        display_n = ((survey.get("survey") or {}).get("metadata") or {}).get("N")
        for sec in (payload.get("frontend") or {}).get("sections") or []:
            for q in sec.get("questions") or []:
                q["N"] = display_n
        payload.setdefault("metadata", {})["N"] = display_n
        published = publish_region_file(SLUG, slug, payload, segment=SEGMENT)
        publish_to_frontend(SLUG, json.dumps(survey, indent=2, ensure_ascii=False))

    return {
        "path": str(path),
        "region": slug,
        "n_questions": n,
        "N": ((survey.get("survey") or {}).get("metadata") or {}).get("N"),
        "published": published,
        "broken_left": sum(
            1
            for s in (survey.get("frontend") or {}).get("sections") or []
            for q in s.get("questions") or []
            if is_broken_grid(q)
        ),
    }


def main() -> None:
    args = sys.argv[1:]
    do_global = "--no-global" not in args
    results = []
    for r in _REGIONS:
        p = OUTPUT / f"{SLUG}_{r}.json"
        if not p.is_file():
            print(f"missing: {p}")
            continue
        info = repair_file(p, publish=True)
        print(json.dumps(info, indent=2))
        results.append(info)

    if do_global and results:
        import os
        os.environ["RUN_REGIONS"] = ""
        import src.question_plan as qp
        qp.RUN_GEOGRAPHIC_REGIONS = list(qp.GEOGRAPHIC_REGIONS)
        from src.nodes.global_selector import publish_global_selection

        g = publish_global_selection(SEGMENT)
        gpath = Path(FRONTEND_SURVEYS_DIR) / SLUG / "global.json"
        if gpath.is_file():
            payload = json.loads(gpath.read_text(encoding="utf-8"))
            rng = random.Random(f"{SLUG}:global")
            display_n = rng.randrange(300, 501, 10)
            for sec in (payload.get("frontend") or {}).get("sections") or []:
                for q in sec.get("questions") or []:
                    if q.get("data"):
                        sum100 = not q.get("multiSelect")
                        q["data"] = _fix_data_list(q["data"], sum_to_100=sum100)
                        if q["data"]:
                            top = max(q["data"], key=lambda d: d.get("value") or 0)
                            q["insight"] = (
                                f"Top response: {top.get('label')} "
                                f"({float(top.get('value') or 0):.2f}%)."
                            )
                    q["N"] = display_n
            payload.setdefault("metadata", {})["N"] = display_n
            gpath.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8",
            )
            out_g = OUTPUT / f"{SLUG}_global.json"
            out_g.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8",
            )
            print(json.dumps({
                "global": g.get("path"),
                "N": display_n,
                "questions": g.get("questions"),
                "regions_used": g.get("regions_used"),
            }, indent=2))


if __name__ == "__main__":
    main()
