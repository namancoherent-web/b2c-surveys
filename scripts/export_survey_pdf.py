"""Export region survey JSON → PDF (question + options + chart + distributionNote).

# change for b2c questionarie

Fixes prior distortion by sizing chart images from true figure aspect ratio
(no forced 6.4×2.85 stretch) and giving horizontal bars enough height.

Each question gets its own display N in [300, 500] (multiple of 10). Questions are emitted
in ascending qNum order to match the JSON sequence.

Usage:
  python scripts/export_survey_pdf.py --sneakers
  python scripts/export_survey_pdf.py --avocado
  python scripts/export_survey_pdf.py output/global-sneakers-market_europe.json
"""
from __future__ import annotations

import html
import io
import json
import math
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import inch  # noqa: E402
from reportlab.platypus import (  # noqa: E402
    HRFlowable,
    Image,
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"

_COLORS = [
    "#1B3A5F", "#2E6B9E", "#4A90C4", "#E8A838", "#3D9B6E",
    "#C45C3E", "#7B6B9E", "#5A8F7B", "#B85C38", "#6B8E23",
]
_LIKERT = ["#1B7A4A", "#5CB87A", "#E8C547", "#E07A3D", "#C0392B"]

# PDF content width (A4 − margins)
_CONTENT_W = 6.7 * inch


def _esc(s: str) -> str:
    return html.escape(str(s or ""), quote=False)


def _truncate(s: str, n: int = 42) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _wrap_label(s: str, width: int = 28) -> str:
    """Soft-wrap long option labels for axis ticks."""
    s = (s or "").strip()
    if len(s) <= width:
        return s
    words = s.split()
    lines: list[str] = []
    cur = ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if len(trial) <= width:
            cur = trial
        else:
            if cur:
                lines.append(cur)
            cur = w if len(w) <= width else w[: width - 1] + "…"
    if cur:
        lines.append(cur)
    return "\n".join(lines[:3])


def _labels_values(q: dict) -> tuple[list[str], list[float]]:
    data = q.get("data") or []
    if data:
        labels = [str(d.get("label") or "") for d in data]
        values = [float(d.get("value") or 0) for d in data]
        return labels, values
    opts = q.get("options") or []
    return [str(o) for o in opts], [0.0] * len(opts)


def _fig_to_png(fig) -> tuple[bytes, float, float]:
    """Save figure; return (png_bytes, width_inches, height_inches) from bbox."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor="white", pad_inches=0.15)
    plt.close(fig)
    buf.seek(0)
    png = buf.read()
    # Read pixel size → inches at 140 dpi
    from PIL import Image as PILImage

    im = PILImage.open(io.BytesIO(png))
    w_in = im.size[0] / 140.0
    h_in = im.size[1] / 140.0
    return png, w_in, h_in


def _render_chart(q: dict) -> tuple[bytes, float, float]:
    """Return PNG bytes and natural size in inches (no forced aspect)."""
    chart = (q.get("chart") or "vbar").lower()
    labels, values = _labels_values(q)
    if not labels:
        labels, values = ["(no data)"], [0.0]

    if chart == "pie":
        chart = "donut"
    # change for b2c questionarie — the web component has richer chart kinds
    # than the PDF renderer; fold each onto its nearest printable equivalent
    # so a new ChartKind can never blank a chart in the export.
    chart = {
        "diverging": "stacked",
        "lollipop": "hbar",
        "gauge": "donut",
        "waffle": "donut",
    }.get(chart, chart)
    if chart == "radar" and len(labels) < 3:
        chart = "vbar"

    n = len(labels)

    if chart == "donut":
        fig = plt.figure(figsize=(7.0, 3.4), dpi=140)
        fig.patch.set_facecolor("white")
        ax = fig.add_subplot(111)
        cols = [_COLORS[i % len(_COLORS)] for i in range(n)]
        wedges, *_ = ax.pie(
            values or [1],
            labels=None,
            colors=cols,
            startangle=90,
            wedgeprops=dict(width=0.42, edgecolor="white", linewidth=1.2),
        )
        ax.legend(
            wedges,
            [f"{_truncate(l, 40)}  {v:g}%" for l, v in zip(labels, values)],
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            fontsize=8,
            frameon=False,
            labelspacing=0.6,
        )
        ax.set_aspect("equal")
        fig.subplots_adjust(left=0.02, right=0.58, top=0.95, bottom=0.05)

    elif chart == "stacked":
        fig = plt.figure(figsize=(7.0, 2.6 + 0.35 * math.ceil(n / 2)), dpi=140)
        fig.patch.set_facecolor("white")
        ax = fig.add_subplot(111)
        left = 0.0
        for i, (lab, val) in enumerate(zip(labels, values)):
            ax.barh(
                [0], [val], left=left, height=0.5,
                color=_COLORS[i % len(_COLORS)],
                edgecolor="white",
                label=f"{_truncate(lab, 32)} ({val:g}%)",
            )
            if val >= 8:
                ax.text(
                    left + val / 2, 0, f"{val:g}%",
                    ha="center", va="center", color="white", fontsize=8, fontweight="bold",
                )
            left += val
        ax.set_yticks([])
        ax.set_xlim(0, max(100.0, left or 100.0))
        ax.set_xlabel("Percent (%)", fontsize=9)
        ax.legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.28),
            ncol=min(3, n), fontsize=7, frameon=False,
        )
        fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.32)

    elif chart == "radar":
        fig = plt.figure(figsize=(5.5, 5.0), dpi=140)
        fig.patch.set_facecolor("white")
        ax = fig.add_subplot(111, polar=True)
        angles = np.linspace(0, 2 * math.pi, n, endpoint=False).tolist()
        vals = list(values) + [values[0]]
        angles_c = angles + [angles[0]]
        ax.plot(angles_c, vals, color=_COLORS[0], linewidth=1.8)
        ax.fill(angles_c, vals, color=_COLORS[0], alpha=0.28)
        ax.set_xticks(angles)
        ax.set_xticklabels([_truncate(l, 20) for l in labels], fontsize=7)
        ax.set_ylim(0, max(100.0, (max(values) * 1.15) if values else 100.0))
        fig.subplots_adjust(left=0.12, right=0.88, top=0.88, bottom=0.12)

    elif chart in ("hbar", "hbar-multi", "hbar-likert"):
        # Height scales with option count — avoids squashed bars
        fig_h = max(2.4, 0.42 * n + 1.0)
        fig = plt.figure(figsize=(7.0, fig_h), dpi=140)
        fig.patch.set_facecolor("white")
        ax = fig.add_subplot(111)
        y = np.arange(n)
        cols = (
            (_LIKERT * ((n // len(_LIKERT)) + 1))[:n]
            if chart == "hbar-likert"
            else [_COLORS[i % len(_COLORS)] for i in range(n)]
        )
        ax.barh(y, values, color=cols, height=0.7, edgecolor="white", linewidth=0.4)
        ax.set_yticks(y)
        ax.set_yticklabels([_wrap_label(l, 34) for l in labels], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Percent (%)", fontsize=9)
        xmax = max(100.0, (max(values) * 1.18) if values else 100.0)
        ax.set_xlim(0, xmax)
        for yi, v in zip(y, values):
            ax.text(min(v + xmax * 0.012, xmax * 0.98), yi, f"{v:g}%", va="center", fontsize=8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        # Left margin for wrapped y-labels
        fig.subplots_adjust(left=0.32, right=0.96, top=0.96, bottom=0.12)

    else:  # vbar / vbar-likert
        fig = plt.figure(figsize=(7.0, 3.6), dpi=140)
        fig.patch.set_facecolor("white")
        ax = fig.add_subplot(111)
        x = np.arange(n)
        cols = (
            (_LIKERT * ((n // len(_LIKERT)) + 1))[:n]
            if chart == "vbar-likert"
            else [_COLORS[i % len(_COLORS)] for i in range(n)]
        )
        ax.bar(x, values, color=cols, width=0.72, edgecolor="white", linewidth=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [_wrap_label(l, 14) for l in labels], fontsize=7, rotation=20, ha="right",
        )
        ax.set_ylabel("Percent (%)", fontsize=9)
        ymax = max(100.0, (max(values) * 1.18) if values else 100.0)
        ax.set_ylim(0, ymax)
        for xi, v in zip(x, values):
            ax.text(xi, v + ymax * 0.015, f"{v:g}%", ha="center", va="bottom", fontsize=7)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        fig.subplots_adjust(left=0.10, right=0.98, top=0.94, bottom=0.28)

    return _fig_to_png(fig)


def _pdf_image(png: bytes, nat_w: float, nat_h: float) -> Image:
    """Fit image to content width without distorting aspect ratio."""
    max_w = _CONTENT_W
    max_h = 5.2 * inch  # keep a single chart on one page when possible
    scale = min(max_w / (nat_w * inch), max_h / (nat_h * inch), 1.0)
    # nat_* are inches already
    w = nat_w * inch * scale
    h = nat_h * inch * scale
    # If still too wide (nat in inches already), scale to max_w
    if w > max_w:
        h = h * (max_w / w)
        w = max_w
    if h > max_h:
        w = w * (max_h / h)
        h = max_h
    return Image(io.BytesIO(png), width=w, height=h)


def _format_tc_value(value) -> str:
    """Render a targetCustomer field value for PDF paragraphs."""
    # change for b2c questionarie
    if value is None:
        return "—"
    if isinstance(value, dict):
        if not value:
            return "—"
        parts = []
        for k, v in value.items():
            if isinstance(v, (list, tuple)):
                v_s = ", ".join(str(x) for x in v) if v else "—"
            else:
                v_s = str(v) if v not in (None, "") else "—"
            parts.append(f"{k}: {v_s}")
        return "; ".join(parts)
    if isinstance(value, (list, tuple)):
        if not value:
            return "—"
        return "; ".join(str(x) for x in value)
    s = str(value).strip()
    return s if s else "—"


def _target_customer_block(survey: dict, styles: dict) -> list:
    """Build PDF flowables for targetCustomer (all keys except note).

    # change for b2c questionarie
    """
    tc = survey.get("targetCustomer") or (
        (survey.get("survey") or {}).get("targetCustomer")
    )
    if not isinstance(tc, dict) or not tc:
        return []

    h1 = styles["h1"]
    small = styles["small"]
    body = styles["tc_body"]
    key_style = styles["tc_key"]

    block: list = []
    block.append(Paragraph("Target customer", h1))
    for key, value in tc.items():
        if key == "note":
            continue
        label = str(key)
        # Prefer readable labels for camelCase keys
        spaced = "".join(
            f" {ch}" if ch.isupper() and i else ch for i, ch in enumerate(label)
        ).strip()
        block.append(Paragraph(f"<b>{_esc(spaced)}</b>", key_style))
        block.append(Paragraph(_esc(_format_tc_value(value)), body))
    block.append(Spacer(1, 8))
    block.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#CCCCCC")))
    block.append(Spacer(1, 6))
    return block


def _assign_per_question_n(survey: dict, *, seed: str) -> None:
    """Give every question the survey's single respondent base.

    # change for b2c questionarie — A16

    This used to assign a DISTINCT random N per question and write it back into
    the source JSON, so exporting a PDF silently rewrote the published data.
    That produced bases moving 310 -> 390 -> 470 between consecutive questions
    with no routing to explain it, and it overwrote the single base the
    generator had already decided.

    The instrument has no skip logic — every respondent sees every question —
    so one base is the only defensible reading. The base already on the file is
    authoritative; a fresh one is drawn only when the file carries none.
    """
    sections = (survey.get("frontend") or {}).get("sections") or []
    questions = [q for sec in sections for q in (sec.get("questions") or [])]

    base = survey.get("metadata", {}).get("N")
    if base is None:
        base = next((q.get("N") for q in questions if q.get("N") is not None), None)
    if base is None:
        base = random.Random(seed).randrange(300, 501, 10)
    base = int(base)

    for q in questions:
        q["N"] = base
    (survey.get("survey") or {}).setdefault("metadata", {})["N"] = base
    survey.setdefault("metadata", {})["N"] = base


# change for b2c questionarie — human-readable section headings for the PDF.
# frontend.sections carries an id but no title, so fall back to a canonical map
# and finally to a title-cased id rather than printing a raw slug.
_SECTION_TITLES = {
    # frontend.sections uses the SHORT legacy ids ("profile", "behavior"),
    # not the canonical ones, so both spellings must resolve or the PDF prints
    # no heading at all and every question runs together.
    "profile": "Consumer Profile, Ownership & Usage Behaviour",
    "behavior": "Purchase Journey & Decision Drivers",
    "behaviour": "Purchase Journey & Decision Drivers",
    "preferences": "Brand / Product Experience & Satisfaction",
    "satisfaction": "Unmet Needs, Switching & Future Purchase Intent",
    "consumer_profile": "Consumer Profile, Ownership & Usage Behaviour",
    "running_activity_context": "Consumer Profile, Ownership & Usage Behaviour",
    "buying_behavior": "Purchase Journey & Decision Drivers",
    "purchase_journey_drivers": "Purchase Journey & Decision Drivers",
    "preferences_expectations": "Product Requirements & Trade-offs",
    "product_requirements_tradeoffs": "Product Requirements & Trade-offs",
    "satisfaction_future_intent": "Satisfaction, Switching & Future Intent",
    "satisfaction_replacement_intent": "Satisfaction, Switching & Future Intent",
    "profiling": "Profiling & Classification",
}
# The generated section ids vary by category (the planner names them), so match
# on the canonical stem when the exact id is not in the table above.
_SECTION_TITLE_STEMS = (
    ("profil", "Profiling & Classification"),
    ("satisfaction", "Satisfaction, Switching & Future Intent"),
    ("fit_", "Fit, Performance & Product Requirements"),
    ("requirement", "Product Requirements & Trade-offs"),
    ("purchase", "Purchase Journey & Decision Drivers"),
    ("buying", "Purchase Journey & Decision Drivers"),
    ("usage", "Consumer Profile, Ownership & Usage Behaviour"),
    ("activity", "Consumer Profile, Ownership & Usage Behaviour"),
)


def _section_title(sec: dict, sec_id: str, blueprint_labels: dict | None = None) -> str:
    """Best available human label for a section.

    Prefers the label the pipeline actually chose for this market (the matched
    CMI framework's category name), so a food study says "Consumption, Usage &
    Occasion Behavior" rather than the Consumer Goods wording.
    """
    for key in ("title", "label", "name"):
        val = (sec.get(key) or "").strip()
        if val:
            return val
    if blueprint_labels and sec_id in blueprint_labels:
        return blueprint_labels[sec_id]
    if sec_id in _SECTION_TITLES:
        return _SECTION_TITLES[sec_id]
    low = sec_id.lower()
    for stem, title in _SECTION_TITLE_STEMS:
        if stem in low:
            return title
    return sec_id.replace("_", " ").title()


def _iter_questions_in_json_order(sections: list) -> list[tuple[dict, dict]]:
    """Flatten questions in ascending qNum order (matches JSON numbering).

    # change for b2c questionarie — PDF sequence must follow JSON qNum order.
    """
    rows: list[tuple[int, int, dict, dict]] = []
    for sec_i, sec in enumerate(sections or []):
        for q_i, q in enumerate(sec.get("questions") or []):
            qnum = q.get("qNum")
            try:
                key = int(qnum)
            except (TypeError, ValueError):
                key = 10_000 + sec_i * 1000 + q_i
            rows.append((key, sec_i * 1000 + q_i, sec, q))
    rows.sort(key=lambda r: (r[0], r[1]))
    return [(sec, q) for _, _, sec, q in rows]


def _survey_metadata(survey: dict) -> dict:
    """Metadata block — region payloads carry it top-level, rich files nest it."""
    return (
        survey.get("metadata")
        or (survey.get("survey") or {}).get("metadata")
        or {}
    )


def _definition_block(survey: dict, st: dict) -> list:
    """Render 'About this survey' — the context a reader needs before question 1.

    # change for b2c questionarie
    """
    d = (
        survey.get("surveyDefinition")
        or (survey.get("survey") or {}).get("surveyDefinition")
        or {}
    )
    if not d:
        return []

    flow: list = [Paragraph("About this survey", st["h1"])]

    for label, key in (
        ("What this survey is", "whatThisSurveyIs"),
        ("Whose context this is", "whoseContextThisIs"),
        ("Why these respondents", "whyTheseRespondents"),
        ("How to read the numbers", "howToReadTheNumbers"),
    ):
        val = (d.get(key) or "").strip()
        if val:
            flow.append(Paragraph(_esc(label), st["tc_key"]))
            flow.append(Paragraph(_esc(val), st["tc_body"]))

    not_asked = d.get("whatIsDeliberatelyNotAsked") or []
    if not_asked:
        flow.append(Paragraph("What is deliberately not asked", st["tc_key"]))
        for item in not_asked:
            flow.append(Paragraph(f"• {_esc(str(item))}", st["tc_body"]))

    structure = d.get("structure") or {}
    secs = structure.get("sections") or []
    if secs:
        flow.append(Paragraph(
            f"Structure — {structure.get('totalQuestions', '?')} questions", st["tc_key"],
        ))
        for s in secs:
            flow.append(Paragraph(
                f"• <b>{_esc(str(s.get('label')))}</b> "
                f"({s.get('questions', 0)} questions) — {_esc(str(s.get('covers') or ''))}",
                st["tc_body"],
            ))

    flow.append(Spacer(1, 8))
    flow.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#CCCCCC")))
    flow.append(Spacer(1, 6))
    return flow


def _storyline_block(survey: dict, st: dict) -> list:
    """Render the planned narrative arc this survey follows.

    # change for b2c questionarie — Phase A8. Shows the storyline the planner
    # produced for this segment/region, so the question order in the pages that
    # follow can be read against its plan.
    """
    meta = _survey_metadata(survey)
    storyline = meta.get("storyline") or {}
    if not storyline:
        return []

    labels = {
        "profile": "Consumer Profile",
        "behavior": "Buying Behavior",
        "preferences": "Preferences & Expectations",
        "satisfaction": "Satisfaction & Future Intent",
    }
    flow: list = [Paragraph("Survey storyline", st["h1"])]

    summary = (meta.get("storyline_summary") or "").strip()
    if summary:
        flow.append(Paragraph(_esc(summary), st["tc_body"]))
    source = meta.get("storyline_source")
    if source:
        flow.append(Paragraph(
            f"Arc source: <b>{_esc(str(source))}</b> "
            "(<i>planner</i> = adapted to this category and region; "
            "<i>default</i> = canonical arc)",
            st["small"],
        ))

    for sid, beats in storyline.items():
        if not beats:
            continue
        flow.append(Paragraph(_esc(labels.get(sid, sid.title())), st["tc_key"]))
        line = "  →  ".join(
            f"{b.get('order')}. {_esc(str(b.get('title') or b.get('beatId')))}"
            for b in beats
        )
        flow.append(Paragraph(line, st["tc_body"]))

    notes = meta.get("storyline_notes") or []
    if notes:
        flow.append(Paragraph("How the arc was adapted", st["tc_key"]))
        for n in notes[:6]:
            flow.append(Paragraph(f"• {_esc(str(n))}", st["tc_body"]))

    flow.append(Spacer(1, 8))
    flow.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#CCCCCC")))
    flow.append(Spacer(1, 6))
    return flow


def export_pdf(survey_path: Path, out_path: Path | None = None) -> Path:
    survey = json.loads(survey_path.read_text(encoding="utf-8"))
    # change for b2c questionarie — assign per-question N before rendering
    _assign_per_question_n(survey, seed=f"pdf-n:{survey_path.stem}")
    # Persist N back into the source JSON so dashboard/PDF stay aligned.
    survey_path.write_text(
        json.dumps(survey, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    meta_seg = (
        survey.get("segment")
        or (survey.get("survey") or {}).get("market_segment")
        or "Survey"
    )
    region = (
        survey.get("region")
        or (survey.get("survey") or {}).get("region")
        or survey_path.stem
    )
    sections = ((survey.get("frontend") or {}).get("sections") or [])
    ordered = _iter_questions_in_json_order(sections)

    if out_path is None:
        out_path = OUTPUT / f"{survey_path.stem}.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "Title2", parent=styles["Title"], fontSize=15, leading=18,
        spaceAfter=4, textColor=colors.HexColor("#1B3A5F"),
    )
    h1 = ParagraphStyle(
        "Sec", parent=styles["Heading1"], fontSize=12, leading=15,
        spaceBefore=12, spaceAfter=6, textColor=colors.HexColor("#1B3A5F"),
    )
    qstyle = ParagraphStyle(
        "Q", parent=styles["Normal"], fontSize=10, leading=14,
        spaceBefore=2, spaceAfter=2, textColor=colors.HexColor("#1a1a1a"),
    )
    small = ParagraphStyle(
        "Small", parent=styles["Normal"], fontSize=8, leading=10,
        textColor=colors.HexColor("#666666"),
    )
    opt_style = ParagraphStyle(
        "Opt", parent=styles["Normal"], fontSize=9, leading=12,
        textColor=colors.HexColor("#333333"),
    )
    # change for b2c questionarie — distributionNote after chart
    note_style = ParagraphStyle(
        "DistNote", parent=styles["Normal"], fontSize=8.5, leading=11.5,
        spaceBefore=6, spaceAfter=2,
        textColor=colors.HexColor("#2C4A6E"),
        backColor=colors.HexColor("#F0F4F8"),
        borderPadding=6,
        leftIndent=4, rightIndent=4,
    )
    note_label = ParagraphStyle(
        "DistLabel", parent=styles["Normal"], fontSize=8, leading=10,
        spaceBefore=6, textColor=colors.HexColor("#1B3A5F"),
    )
    tc_key = ParagraphStyle(
        "TcKey", parent=styles["Normal"], fontSize=9, leading=11,
        spaceBefore=6, spaceAfter=1, textColor=colors.HexColor("#1B3A5F"),
    )
    tc_body = ParagraphStyle(
        "TcBody", parent=styles["Normal"], fontSize=9, leading=12,
        textColor=colors.HexColor("#333333"),
    )

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        leftMargin=0.55 * inch,
        rightMargin=0.55 * inch,
        topMargin=0.5 * inch,
        bottomMargin=0.5 * inch,
        title=f"{meta_seg} — {region}",
    )
    story: list = []
    story.append(Paragraph(_esc(str(meta_seg)), title_style))
    story.append(
        Paragraph(
            f"Region: <b>{_esc(str(region))}</b> &nbsp;|&nbsp; "
            "Questions · options · % charts · distribution notes &nbsp;|&nbsp; "
            "N varies per question (300–500, ×10)",
            small,
        )
    )
    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor("#CCCCCC")))

    # change for b2c questionarie — definition first, then who was interviewed
    story.extend(
        _definition_block(
            survey,
            {"h1": h1, "small": small, "tc_key": tc_key, "tc_body": tc_body},
        )
    )
    story.extend(
        _target_customer_block(
            survey,
            {
                "h1": h1,
                "small": small,
                "tc_key": tc_key,
                "tc_body": tc_body,
            },
        )
    )

    # change for b2c questionarie — Phase A8: show the arc before the questions
    story.extend(
        _storyline_block(
            survey,
            {"h1": h1, "small": small, "tc_key": tc_key, "tc_body": tc_body},
        )
    )

    last_sec_id = object()
    for sec, q in ordered:
        sec_id = sec.get("id") or sec.get("label")
        if sec_id != last_sec_id:
            # change for b2c questionarie — frontend.sections carries an id but
            # no label/title, so this printed the raw slug
            # ("running_activity_context"). Resolve a readable name instead.
            story.append(
                Paragraph(_esc(_section_title(sec, str(sec_id or ""))), h1)
            )
            last_sec_id = sec_id

        qnum = q.get("qNum")
        text = q.get("question") or ""
        # change for b2c questionarie — per-question display N
        q_n = q.get("N")
        if q_n is None:
            q_n = random.randrange(300, 501, 10)
            q["N"] = q_n

        story.append(Paragraph(f"<b>Q{qnum}.</b> {_esc(text)}", qstyle))
        # change for b2c questionarie — the question_layer values ("core",
        # "module", "profiling") are generator plumbing: core vs module is how
        # the region fan-out decides which questions are shared, and it means
        # nothing to someone reading the instrument. Printing it put a stray
        # "| core" under most questions. Only N belongs on this line.
        meta_line = f"N = <b>{_esc(str(q_n))}</b>"
        story.append(Paragraph(meta_line, small))

        labels, values = _labels_values(q)
        opts = q.get("options") or labels
        pct_map = {str(l): v for l, v in zip(labels, values)}
        items = []
        for o in opts:
            pct = pct_map.get(str(o))
            if pct is None:
                for l, v in zip(labels, values):
                    if str(l).strip().lower() == str(o).strip().lower():
                        pct = v
                        break
            if pct is not None:
                items.append(
                    ListItem(
                        Paragraph(f"{_esc(o)} — <b>{pct:g}%</b>", opt_style),
                        leftIndent=6,
                    )
                )
            else:
                items.append(
                    ListItem(Paragraph(_esc(str(o)), opt_style), leftIndent=6)
                )
        if items:
            story.append(Paragraph("<i>Options</i>", small))
            story.append(
                ListFlowable(items, bulletType="bullet", start="•", leftIndent=10)
            )

        try:
            png, nat_w, nat_h = _render_chart(q)
            story.append(Spacer(1, 6))
            story.append(_pdf_image(png, nat_w, nat_h))
        except Exception as exc:  # noqa: BLE001
            story.append(Paragraph(f"[chart error: {_esc(str(exc))}]", small))

        # change for b2c questionarie — distributionNote under each chart
        note = (q.get("distributionNote") or q.get("distribution_note") or "").strip()
        if note:
            story.append(Paragraph("<b>Distribution note</b>", note_label))
            story.append(Paragraph(_esc(note), note_style))
        else:
            story.append(Paragraph("<i>Distribution note: (none)</i>", small))

        story.append(Spacer(1, 8))
        story.append(
            HRFlowable(width="100%", thickness=0.4, color=colors.HexColor("#E0E0E0"))
        )
        story.append(Spacer(1, 6))

    doc.build(story)
    return out_path

_REGION_SLUGS = [
    "north-america",
    "europe",
    "asia-pacific",
    "latin-america",
    "middle-east-africa",
]


def _export_segment_regions(slug_prefix: str, out_subdir: str) -> list[Path]:
    """Export all 5 region PDFs for a segment into output/<out_subdir>/."""
    # change for b2c questionarie
    out_dir = OUTPUT / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for region in _REGION_SLUGS:
        src = OUTPUT / f"{slug_prefix}_{region}.json"
        if not src.is_file():
            print(f"missing: {src}")
            continue
        dest = out_dir / f"{slug_prefix}_{region}.pdf"
        path = export_pdf(src, dest)
        print(f"wrote {path}")
        written.append(path)
    return written


def export_sneakers() -> list[Path]:
    """Export all 5 region PDFs into output/sneaker/."""
    return _export_segment_regions("global-sneakers-market", "sneaker")


def export_avocado() -> list[Path]:
    """Export all 5 avocado-oil region PDFs into output/avocado-oil/."""
    # change for b2c questionarie
    written = _export_segment_regions("global-avocado-oil-market", "avocado-oil")
    # Mirror updated per-question N into the frontend surveys folder.
    try:
        import sys

        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from src.survey_io import publish_region_file

        for region in _REGION_SLUGS:
            src = OUTPUT / f"global-avocado-oil-market_{region}.json"
            if not src.is_file():
                continue
            payload = json.loads(src.read_text(encoding="utf-8"))
            path = publish_region_file(
                "global-avocado-oil-market",
                region,
                payload,
                segment="Avocado Oil",
            )
            if path:
                print(f"published {path}")
    except Exception as exc:  # noqa: BLE001
        print(f"publish skipped: {exc}")
    return written


def export_final_avocado() -> list[Path]:
    """Export every JSON in output/final_avacado_oil/ to a PDF beside it.

    # change for b2c questionarie
    """
    folder = OUTPUT / "final_avacado_oil"
    if not folder.is_dir():
        print(f"missing folder: {folder}")
        return []
    written: list[Path] = []
    for src in sorted(folder.glob("*.json")):
        dest = folder / f"{src.stem}.pdf"
        path = export_pdf(src, dest)
        print(f"wrote {path}")
        written.append(path)
    return written


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] in ("--sneakers", "--sneaker"):
        export_sneakers()
        return
    if args[0] in ("--avocado", "--avocado-oil"):
        export_avocado()
        return
    if args[0] in ("--final-avocado", "--final-avacado-oil", "--final_avacado_oil"):
        export_final_avocado()
        return
    if args[0] == "--organic-milk-ap-na":
        paths = [
            OUTPUT / "organic-milk_asia-pacific.json",
            OUTPUT / "organic-milk_north-america.json",
        ]
    else:
        paths = [Path(a) for a in args]

    for p in paths:
        if not p.is_file():
            print(f"missing: {p}")
            continue
        out = export_pdf(p)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
