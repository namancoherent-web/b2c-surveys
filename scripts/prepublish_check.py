"""Pre-publish acceptance check for a B2C survey JSON (spec v3 Part E).

# change for b2c questionarie -- CHECK (spec: b2c_survey_question_structure
# _v3.txt Part E + survey_wording_and_brand_policy_update.txt Part D,
# 2026-09-11)

Part E of the spec is a checklist a human is meant to run before publishing.
A checklist that lives only in a document gets skipped, so it lives here as
code and runs against a delivered file:

    python scripts/prepublish_check.py "output/<file>.json"

Exit code 0 = every box ticked. Exit code 1 = at least one failure, with the
failing item named. The checks reuse the SAME validator functions the
pipeline gates on, so this cannot drift away from what the pipeline enforces.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.nodes.validator_critic import (  # noqa: E402
    REQUIRED_SLOTS,
    find_brand_issues,
    find_construct_duplicates,
    find_drifted_theme_questions,
    find_grammar_issues,
    find_missing_required_slots,
    find_offtheme_questions,
    find_parallelism_issues,
    find_register_issues,
    slot_satisfied_by,
)
from src.question_plan import QUESTIONS_PER_SECTION  # noqa: E402

# Section titles are market-specific by design (the RAR's THEMES are what is
# fixed, not its literal labels), so map by position: the four behavioural
# sections always appear in canonical order, with Profiling last.
_CANON_ORDER = [
    "consumer_profile",
    "buying_behavior",
    "preferences_expectations",
    "satisfaction_future_intent",
]


def _load(path: str) -> dict:
    with open(path, encoding="utf-8-sig") as fh:
        return json.load(fh)


def _flatten(doc: dict) -> tuple[list, dict, dict]:
    """-> (answered-shaped list, {canon: [texts]}, {canon: title})."""
    answered, by_canon, titles = [], {}, {}
    behavioural = [s for s in doc.get("segments") or []
                   if (s.get("title") or "").strip().lower() != "profiling"]
    for i, seg in enumerate(behavioural):
        if i >= len(_CANON_ORDER):
            break
        canon = _CANON_ORDER[i]
        titles[canon] = seg.get("title") or ""
        by_canon[canon] = []
        for q in seg.get("questions") or []:
            qq = dict(q)
            qq["tab"] = canon
            qq["question_layer"] = "core"
            answered.append(qq)
            by_canon[canon].append(q.get("text") or "")
    return answered, by_canon, titles


def run(path: str) -> int:
    doc = _load(path)
    answered, by_canon, _titles = _flatten(doc)
    segments = doc.get("segments") or []
    failures: list[str] = []
    ok: list[str] = []

    def check(label: str, passed: bool, detail: str = "") -> None:
        (ok if passed else failures).append(label + (f" — {detail}" if detail and not passed else ""))

    # [ ] Counts exact: 5 / 7 / 6 / 5 plus 2 profiling
    counts = {}
    for i, seg in enumerate([s for s in segments
                             if (s.get("title") or "").strip().lower() != "profiling"]):
        if i < len(_CANON_ORDER):
            counts[_CANON_ORDER[i]] = len(seg.get("questions") or [])
    bad_counts = [
        f"{c}={counts.get(c)}!={QUESTIONS_PER_SECTION[c]}"
        for c in _CANON_ORDER
        if counts.get(c) != QUESTIONS_PER_SECTION[c]
    ]
    prof = next((s for s in segments
                 if (s.get("title") or "").strip().lower() == "profiling"), None)
    n_prof = len(prof.get("questions") or []) if prof else 0
    check("Counts exact 5/7/6/5 + 2 profiling",
          not bad_counts and n_prof == 2,
          "; ".join(bad_counts) + (f"; profiling={n_prof}" if n_prof != 2 else ""))

    # [ ] All 23 slots present, each in its own section
    missing = find_missing_required_slots(answered, {c: c for c in _CANON_ORDER})
    n_missing = sum(len(v) for v in missing.values())
    total_slots = sum(len(v) for v in REQUIRED_SLOTS.values())
    check(f"All {total_slots} slots present",
          n_missing == 0,
          "; ".join(f"{c}:{l}" for c, gaps in missing.items() for _k, l in gaps))

    # [ ] Nothing off-theme / duplicated / drifted
    off = find_offtheme_questions(answered, {c: c for c in _CANON_ORDER})
    check("No off-theme questions", not off,
          "; ".join(t[:45] for v in off.values() for t, _n in v))
    check("No construct duplicates", not find_construct_duplicates(answered))
    drift = find_drifted_theme_questions(answered, {c: c for c in _CANON_ORDER})
    check("No drifted/near-duplicate themes", not drift,
          "; ".join(f"{k}" for v in drift.values() for k, _t, _r in v))

    # [ ] Register: contractions, "I " options, dashes, lengths, formality
    reg = find_register_issues(answered)
    check("Written register clean", not reg, f"{len(reg)} issue(s): " + "; ".join(reg[:3]))
    par = find_parallelism_issues(answered)
    check("Options share one grammatical shape", not par, "; ".join(par[:2]))

    # [ ] Every question is a grammatically correct English question
    gram = find_grammar_issues(answered)
    check("Question grammar correct", not gram, "; ".join(gram[:3]))

    # [ ] Brands only in permitted slots, on the allowlist, with catch-all
    from src import brand_allowlist as ba

    country = (doc.get("surveyScope") or {}).get("region") or ""
    category = (doc.get("surveyScope") or {}).get("category") or ""
    allowed = ba.brand_names(country, category, allow_search=False) if country else []
    brand_issues = find_brand_issues(answered, allowed_brands=allowed,
                                     section_ids={c: c for c in _CANON_ORDER})
    check("Brand policy clean", not brand_issues, "; ".join(brand_issues[:3]))

    # [ ] brand_verified written into the output
    check("brandVerified present in surveyScope",
          "brandVerified" in (doc.get("surveyScope") or {}))

    # [ ] Every question's percentages sum to 100.0
    bad_sums = []
    for seg in segments:
        for q in seg.get("questions") or []:
            if q.get("type") in ("single_select", "rating_0_10"):
                tot = sum(float(o.get("pct") or 0) for o in q.get("options") or [])
                if abs(tot - 100.0) > 0.15:
                    bad_sums.append(f"{q.get('id')}={tot}")
    check("Single-select/rating sum to 100", not bad_sums, "; ".join(bad_sums[:4]))

    # [ ] Advocacy question has exactly 11 options, 0 through 10
    adv = [q for seg in segments for q in seg.get("questions") or []
           if "recommend" in (q.get("text") or "").lower()]
    adv_ok = bool(adv) and all(
        len(q.get("options") or []) == 11
        and str((q["options"][0].get("label") or "")).strip().startswith("0")
        and str((q["options"][-1].get("label") or "")).strip().startswith("10")
        for q in adv
    )
    check("Advocacy is an 11-option 0-10 scale", adv_ok)

    # [ ] Gender question offers "Prefer not to say"
    gender_ok = any(
        "gender" in (q.get("text") or "").lower()
        and any("prefer not to say" in str(o.get("label") or "").lower()
                for o in q.get("options") or [])
        for seg in segments for q in seg.get("questions") or []
    )
    check("Gender offers 'Prefer not to say'", gender_ok)

    # [ ] Template order within every section
    order_bad = []
    for canon, texts in by_canon.items():
        slots = REQUIRED_SLOTS.get(canon, ())
        ranks = [
            next((i for i, (k, _l, p) in enumerate(slots)
                  if slot_satisfied_by(t, k, p)), len(slots))
            for t in texts
        ]
        if ranks != sorted(ranks):
            order_bad.append(canon)
    check("Template order within each section", not order_bad, ", ".join(order_bad))

    print(f"\nPRE-PUBLISH CHECK — {Path(path).name}")
    print("=" * 64)
    for line in ok:
        print(f"  [x] {line}")
    for line in failures:
        print(f"  [ ] FAIL: {line}")
    print("=" * 64)
    print(f"  {len(ok)} passed, {len(failures)} failed\n")
    return 1 if failures else 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/prepublish_check.py <survey.json>")
        raise SystemExit(2)
    raise SystemExit(run(sys.argv[1]))
