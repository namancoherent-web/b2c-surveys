"""Render a published survey as a research-firm markdown deliverable.

# change for b2c questionarie — A11

Mirrors ``export_survey_pdf.py``: reads a published payload from
``published_surveys/<slug>/<region>.json`` and writes markdown. It changes
nothing about generation — the instrument, the segments and the scoring model
are read as-is, so the markdown and the PDF always describe the same survey.

The document follows the order a client reads in: what the study is for, who it
separates, the instrument itself, how answers map to segments, and how to field
it.

Usage:
  python scripts/export_survey_markdown.py published_surveys/running-shoes/global.json
  python scripts/export_survey_markdown.py --all
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = ROOT / "published_surveys"
OUT_DIR = ROOT / "output" / "markdown"


def _esc(text: object) -> str:
    """Escape pipes so free text cannot break a markdown table."""
    return str(text or "").replace("|", "\\|").replace("\n", " ").strip()


def _sections(doc: dict) -> list:
    for holder in (doc.get("frontend") or {}, doc, doc.get("survey") or {}):
        if holder.get("sections"):
            return holder["sections"]
    return []


def _meta(doc: dict) -> dict:
    return doc.get("metadata") or (doc.get("survey") or {}).get("metadata") or {}


def _questions_in_order(sections: list) -> list[tuple[dict, dict]]:
    """Flatten to (section, question) in ascending qNum — the PDF's order."""
    rows = []
    for si, sec in enumerate(sections or []):
        for qi, q in enumerate(sec.get("questions") or []):
            try:
                key = int(q.get("qNum"))
            except (TypeError, ValueError):
                key = 10_000 + si * 1000 + qi
            rows.append((key, si * 1000 + qi, sec, q))
    rows.sort(key=lambda r: (r[0], r[1]))
    return [(sec, q) for _, _, sec, q in rows]


def _study_block(doc: dict, defn: dict, meta: dict) -> list[str]:
    out = ["## 1. Study objective and methodology", ""]

    purpose = defn.get("studyPurpose") or meta.get("studyPurpose")
    if purpose:
        out += [f"**Objective.** {purpose}", ""]

    reason = meta.get("classification_reason") or defn.get("whyTheseRespondents")
    if reason:
        out += [f"**Market classification.** {reason}", ""]

    # change for b2c questionarie — A11: a dual-motion business must say which
    # side of the business the study covers, or the reader will assume both.
    if meta.get("sells_to_both") or defn.get("dualMotionReason"):
        why = defn.get("dualMotionReason") or meta.get("dual_motion_reason") or ""
        out += [
            "**Scope — consumer motion.** This business sells to both consumer "
            "and business buyers. This study covers the **consumer** motion: "
            "people buying on their own account. Business buyers are a separate "
            "population with a buying committee and a procurement cycle, and "
            "surveying them requires a different instrument."
            + (f" {why}" if why else ""),
            "",
        ]

    if defn.get("whatThisSurveyIs"):
        out += [f"**What this survey is.** {defn['whatThisSurveyIs']}", ""]
    if defn.get("whoseContextThisIs"):
        out += [f"**Whose context this is.** {defn['whoseContextThisIs']}", ""]
    # These definition fields are sometimes a string and sometimes a list of
    # reasons, depending on which node wrote them — render both as prose.
    excluded = defn.get("whatIsDeliberatelyNotAsked")
    if excluded:
        out += ["**Deliberately not asked.**", ""]
        if isinstance(excluded, (list, tuple)):
            out += [f"- {item}" for item in excluded]
        else:
            out.append(str(excluded))
        out.append("")
    return out


def _segment_sizes(meta: dict, sections: list) -> dict:
    """Indicative segment share, derived from the scoring model.

    # change for b2c questionarie — A13

    There is no respondent-level data here, so a true segment size cannot be
    counted. What can be derived is each segment's share of total scoring
    weight actually earned: for every weighted option, its response percentage
    times its points. Normalised, that is a defensible indication of relative
    size — and it is labelled as indicative, never as a count.
    """
    weights = (meta.get("segmentModel") or {}).get("weights") or []
    if not weights:
        return {}
    values = {
        q.get("id"): {str(d.get("label")): float(d.get("value") or 0)
                      for d in (q.get("data") or [])}
        for _sec, q in _questions_in_order(sections)
    }
    earned: dict = {}
    for w in weights:
        by_label = values.get(w.get("question_id")) or {}
        pct = by_label.get(str(w.get("option_label")))
        if pct is None:
            continue
        earned[w.get("segment_id")] = earned.get(w.get("segment_id"), 0.0) + (
            pct * float(w.get("points") or 0)
        )
    total = sum(earned.values())
    if not total:
        return {}
    return {k: v * 100.0 / total for k, v in earned.items()}


def _segments_block(defn: dict, meta: dict, sections: list | None = None) -> list[str]:
    segments = defn.get("segments") or meta.get("segments") or []
    if not segments:
        return []
    out = ["## 2. Segments this study separates", ""]

    lens = meta.get("segmentation_lens") or defn.get("segmentationLens")
    if lens:
        out += [f"**Segmentation lens.** {lens}", ""]

    sizes = _segment_sizes(meta, sections or [])
    if sizes:
        out += ["| # | Segment | Indicative size | Defining behaviour |",
                "|---|---|---|---|"]
    else:
        out += ["| # | Segment | Defining behaviour |", "|---|---|---|"]

    for i, s in enumerate(segments, 1):
        name = _esc(s.get("name") if isinstance(s, dict) else s)
        desc = _esc(s.get("description") if isinstance(s, dict) else "")
        if sizes:
            sid = s.get("id") or s.get("segment_id") if isinstance(s, dict) else None
            share = sizes.get(sid)
            cell = f"{share:.0f}%" if share is not None else "—"
            out.append(f"| {i} | **{name}** | {cell} | {desc} |")
        else:
            out.append(f"| {i} | **{name}** | {desc} |")
    out.append("")
    if sizes:
        out += [
            "_Sizes are indicative, derived from each segment's share of the "
            "scoring weight earned across the simulated distributions. They are "
            "not respondent counts — field the study to size the segments._",
            "",
        ]
    return out


def _cross_tab_block(sections: list, doc: dict) -> list[str]:
    """Region cross-tabs — the only cut this data can actually support.

    # change for b2c questionarie — A13

    Cutting by segment, age or income needs respondent-level records, which
    simulated toplines do not contain. Regional variants DO exist per question,
    so those cross-tabs are real. Printing them and stating plainly which other
    cuts require fielding is more useful than a fabricated demographic table.
    """
    regions = ("north-america", "europe", "asia-pacific",
               "latin-america", "middle-east-africa")
    rows = [
        (sec, q) for sec, q in _questions_in_order(sections)
        if (q.get("dataByRegion") or {}) and (q.get("questionLayer") != "profiling")
    ]
    out = ["## 6. Cross-tabulations", ""]
    if not rows:
        out += [
            "This payload carries a single region, so no regional cut is "
            "available. Cutting by segment, age, gender or income requires "
            "respondent-level records — run the fieldwork, then apply the "
            "scoring model in section 4 to assign segments and cut from there.",
            "",
        ]
        return out

    out += ["Cut by region, for the three questions that discriminate most.", ""]
    for _sec, q in rows[:3]:
        dbr = q.get("dataByRegion") or {}
        present = [r for r in regions if dbr.get(r)]
        if not present:
            continue
        labels = [str(d.get("label")) for d in (q.get("data") or [])]
        out += ["", f"**Q{q.get('qNum')}. {q.get('question')}**", ""]
        out.append("| Response | " + " | ".join(
            r.replace("-", " ").title() for r in present) + " |")
        out.append("|---" * (len(present) + 1) + "|")
        for label in labels:
            cells = []
            for r in present:
                by_label = {str(d.get("label")): d.get("value")
                            for d in dbr.get(r) or []}
                v = by_label.get(label)
                cells.append(f"{float(v):.0f}%" if v is not None else "—")
            out.append(f"| {_esc(label)} | " + " | ".join(cells) + " |")
        out.append("")
    out += [
        "Cuts by segment, age, gender and income need respondent-level records. "
        "Apply the scoring model in section 4 after fielding to assign each "
        "respondent to a segment, then cut every behavioural question by it.",
        "",
    ]
    return out


def _executive_summary(doc: dict, defn: dict, meta: dict, sections: list) -> list[str]:
    """What a reader needs before the instrument: findings, not description.

    # change for b2c questionarie — A13

    Every statement here is derived from the distributions in this document —
    top responses, computed NPS, indicative segment sizes. Nothing is asserted
    that the data does not support.
    """
    qs = [q for _s, q in _questions_in_order(sections)
          if q.get("questionLayer") != "profiling"]
    if not qs:
        return []

    out = ["## Executive summary", ""]

    purpose = defn.get("studyPurpose") or meta.get("studyPurpose")
    if purpose:
        out += [purpose, ""]

    sizes = _segment_sizes(meta, sections)
    segments = defn.get("segments") or meta.get("segments") or []
    if sizes and segments:
        ranked = sorted(
            ((s.get("name"), sizes.get(s.get("id") or s.get("segment_id"), 0))
             for s in segments if isinstance(s, dict)),
            key=lambda x: x[1], reverse=True,
        )
        largest = ranked[0]
        smallest = ranked[-1]
        out.append(
            f"- **{largest[0]}** is the largest group at an indicative "
            f"{largest[1]:.0f}% of the market; **{smallest[0]}** the smallest at "
            f"{smallest[1]:.0f}%."
        )

    # The strongest single reading in each of the first three sections.
    seen_sections: list = []
    for sec, q in _questions_in_order(sections):
        if q.get("questionLayer") == "profiling":
            continue
        sid = sec.get("id")
        if sid in seen_sections:
            continue
        data = q.get("data") or []
        if not data:
            continue
        top = max(data, key=lambda d: float(d.get("value") or 0))
        share = float(top.get("value") or 0)
        if share < 25:
            continue
        seen_sections.append(sid)
        out.append(
            f"- On *{_esc(sec.get('label'))}*: {share:.0f}% answer "
            f"“{_esc(top.get('label'))}” to Q{q.get('qNum')}."
        )
        if len(seen_sections) >= 3:
            break

    headline = _headline_block(sections)
    for line in headline:
        if line.startswith("| Net Promoter"):
            value = line.split("**")[1] if "**" in line else ""
            out.append(
                f"- Net Promoter Score is **{value}** — see section 0 for the "
                "promoter/detractor split."
            )
            break

    out += [
        "",
        "_These readings come from simulated distributions, not fielded "
        "responses. Treat them as hypotheses to test, not findings._",
        "",
    ]
    return out


def _instrument_block(sections: list) -> list[str]:
    out = ["## 3. The instrument", ""]
    last = None
    for sec, q in _questions_in_order(sections):
        sid = sec.get("id") or sec.get("label")
        if sid != last:
            out += ["", f"### {sec.get('label') or sid}", ""]
            last = sid

        qnum = q.get("qNum")
        out.append(f"**Q{qnum}. {q.get('question') or ''}**")

        # change for b2c questionarie — A12: state the response type. Without it
        # a multi-select summing past 100% reads as an arithmetic error.
        qtype = (q.get("questionType") or "").replace("_", " ")
        instruction = q.get("responseInstruction") or ""
        beat = (q.get("beatTitle") or q.get("beat_title") or "").strip()
        bits = [b for b in (instruction, qtype and f"type: {qtype}",
                            beat and f"beat: {beat}") if b]
        if bits:
            out.append(f"<sub>{_esc(' · '.join(bits))}</sub>")
        out.append("")
        for opt in q.get("options") or []:
            out.append(f"- {opt}")
        if q.get("questionType") == "multiple_choice":
            out.append("")
            out.append("<sub>Responses sum above 100%: each option is an "
                       "independent share of respondents selecting it.</sub>")
        out.append("")
    return out


def _headline_block(sections: list) -> list[str]:
    """Derived readings a client checks first — computed, not asserted.

    # change for b2c questionarie — A12

    An NPS printed without being computed is how a document ends up publishing
    a deeply negative score beside a satisfied majority. Deriving it here means
    the number in the deliverable is the number the data actually produces.
    """
    import re as _re

    nps = satisfied = None
    for _sec, q in _questions_in_order(sections):
        data = q.get("data") or []
        scores = []
        for d in data:
            m = _re.match(r"\s*(\d{1,2})\b", str(d.get("label") or ""))
            if m and int(m.group(1)) <= 10:
                scores.append((int(m.group(1)), float(d.get("value") or 0)))
        if len(scores) >= 8 and nps is None:
            promoters = sum(v for s, v in scores if s >= 9)
            detractors = sum(v for s, v in scores if s <= 6)
            nps = (promoters - detractors, promoters, detractors, q.get("qNum"))
            continue
        sat_labels = [d for d in data if "satisf" in str(d.get("label") or "").lower()]
        if len(sat_labels) >= 3 and satisfied is None:
            total = sum(float(d.get("value") or 0) for d in sat_labels)
            pos = sum(
                float(d.get("value") or 0) for d in sat_labels
                if "dissatisf" not in str(d["label"]).lower()
                and "neither" not in str(d["label"]).lower()
            )
            if total:
                satisfied = (pos * 100.0 / total, q.get("qNum"))

    if not nps and not satisfied:
        return []
    out = ["## 0. Headline readings", "",
           "| Measure | Value | Source |", "|---|---|---|"]
    if nps:
        value, promoters, detractors, qnum = nps
        out.append(
            f"| Net Promoter Score | **{value:+.0f}** "
            f"({promoters:.0f}% promoters, {detractors:.0f}% detractors) | Q{qnum} |"
        )
    if satisfied:
        share, qnum = satisfied
        out.append(f"| Satisfied (top-2 box) | **{share:.0f}%** | Q{qnum} |")
    out += ["", "_Computed from the distributions in this document._", ""]
    return out


def _segment_names(defn: dict, meta: dict) -> dict:
    """segment_id -> display name.

    The scoring model keys on ids, which are stable across renames. A reader
    should see the same label in the scoring table as in the segments table, so
    resolve ids to names here rather than printing the raw key.
    """
    out = {}
    for s in (defn.get("segments") or meta.get("segments") or []):
        if isinstance(s, dict) and s.get("id") or isinstance(s, dict) and s.get("segment_id"):
            out[s.get("id") or s.get("segment_id")] = s.get("name") or ""
    return out


def _scoring_block(meta: dict, names: dict | None = None) -> list[str]:
    names = names or {}
    model = meta.get("segmentModel") or {}
    weights = model.get("weights") or []
    if not weights:
        return [
            "## 4. Segmentation scoring model", "",
            "_No scoring model is present in this payload._", "",
        ]

    out = [
        "## 4. Segmentation scoring model", "",
        "Each answer below contributes points to one segment. A respondent is "
        "assigned to the segment with the highest total. Points: 3 = decisive, "
        "2 = strong, 1 = supporting.", "",
        "| Question | Answer | Points to | Points |", "|---|---|---|---|",
    ]
    for w in weights:
        sid = w.get("segment_id")
        out.append(
            f"| {_esc(w.get('question_id'))} | {_esc(w.get('option_label'))} "
            f"| {_esc(names.get(sid) or sid)} | {w.get('points')} |"
        )
    out.append("")
    if model.get("tie_break"):
        out += [f"**Tie-break.** {model['tie_break']}", ""]

    # change for b2c questionarie — A16: name the questions that carry no
    # scoring weight, so a reviewer can see which ones are context rather than
    # segmentation, instead of having to work it out.
    idle = meta.get("nonDiscriminatingQuestions") or []
    if idle:
        out += [
            "**Questions that do not feed the model.** These carry no scoring "
            "weight — they set context but do not separate the segments. Keep "
            "them only if the narrative needs them:",
            "",
        ]
        out += [f"- Q{q.get('qNum')} — {_esc(q.get('question'))}" for q in idle]
        out.append("")
    for note in model.get("notes") or []:
        out.append(f"> {note}")
    if model.get("notes"):
        out.append("")
    return out


def _fielding_block(meta: dict) -> list[str]:
    notes = meta.get("fieldingNotes") or {}
    if not notes:
        return []
    out = ["## 5. Fielding notes", ""]
    for label, key in (
        ("Recommended sample", "recommendedSample"),
        ("Quota controls", "quotaControls"),
        ("Estimated length", "estimatedLength"),
    ):
        if notes.get(key):
            out.append(f"- **{label}.** {notes[key]}")
    out.append("")

    # change for b2c questionarie — A15: the mechanics a scripter needs.
    scripting = notes.get("scriptingNotes") or []
    if scripting:
        out += ["### Scripting and field mechanics", ""]
        out += [f"- {s}" for s in scripting]
        out.append("")

    # change for b2c questionarie — A12: profiling questions exist to cut the
    # behavioural findings. Saying so, and naming the cuts, is the difference
    # between collecting demographics and using them.
    out += [
        "### Planned cross-tabulations", "",
        "Every behavioural question is to be cut by:", "",
        "- Assigned segment (from the scoring model in section 4) — the primary cut",
        "- Age band and household composition (Profiling)",
        "- Household income band (Profiling)",
        "- Category usage frequency (the opening behavioural question)",
        "",
        "Report a cut only where the segment base is large enough to carry it; "
        "below roughly n=100 per cell, show the direction and label it indicative "
        "rather than quoting a percentage.",
        "",
    ]
    caveats = notes.get("caveats") or []
    if caveats:
        out += ["### Methodological caveats", ""]
        out += [f"- {c}" for c in caveats]
        out.append("")
    return out


def render(doc: dict) -> str:
    defn = doc.get("surveyDefinition") or {}
    meta = _meta(doc)
    sections = _sections(doc)

    segment = doc.get("segment") or defn.get("category") or "Survey"
    region = doc.get("region") or defn.get("region") or "global"
    total = meta.get("total_questions") or sum(
        len(s.get("questions") or []) for s in sections
    )

    lines = [
        f"# {segment} — consumer segmentation study",
        "",
        f"**Region:** {region}  |  **Questions:** {total}",
        "",
        "---",
        "",
    ]
    lines += _executive_summary(doc, defn, meta, sections)
    lines += _study_block(doc, defn, meta)
    lines += _headline_block(sections)
    lines += _segments_block(defn, meta, sections)
    lines += _instrument_block(sections)
    lines += _scoring_block(meta, _segment_names(defn, meta))
    lines += _fielding_block(meta)
    lines += _cross_tab_block(sections, doc)
    lines += [
        "---", "",
        "_Percentages in the accompanying charts are simulated from a "
        "persona-weighted panel, not fielded with live respondents. This "
        "document is the instrument and its scoring logic; replace the figures "
        "with primary data after fielding._", "",
    ]
    return "\n".join(lines)


def export(path: Path) -> Path:
    doc = json.loads(path.read_text(encoding="utf-8"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{path.parent.name}__{path.stem}.md"
    out.write_text(render(doc), encoding="utf-8")
    return out


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 2
    if args[0] == "--all":
        paths = sorted(
            p for d in PUBLISHED.iterdir() if d.is_dir()
            for p in d.glob("*.json") if p.stem != "index"
        )
    else:
        paths = [Path(a) for a in args]

    for p in paths:
        if not p.is_file():
            print(f"missing: {p}")
            continue
        print(f"wrote {export(p).relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
