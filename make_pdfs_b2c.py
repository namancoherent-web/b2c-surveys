"""
make_pdfs_b2c.py — PDF renderer for B2C survey JSON.

Reads the surveyScope/behaviouralPersonas/structure/segments delivery
format (B2C_MASTER_RULES_FINAL.txt / JSON_FORMAT_SPECIFICATION.txt).
Reuses make_pdfs.py's palette and chart flowables (built for the B2B
{"label","pct"} option shape, which this format already uses natively).

Usage:
    python make_pdfs_b2c.py "b2c-survey-agent-main/output/<file>.json"
"""
import glob
import json
import os
import sys

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

from make_pdfs import (
    ACCENT, BLUE, BODY, GREY, H_SUB, H_TITLE, INSIGHT, KV, LIGHT, NAVY, Q_TEXT,
    esc, make_chart, seg_banner,
)

WARNING_BODY = ParagraphStyle("WarningBody", parent=BODY, fontSize=9,
                              textColor=NAVY, leading=13, spaceAfter=2)
SECTION_LEDE = ParagraphStyle("SectionLede", parent=BODY, fontSize=9.5,
                              textColor=GREY, leading=13, spaceAfter=4)

# Spec: PDF RENDERING CONVENTIONS.
_RUNNING_HEADER = "COHERENT MARKET INSIGHTS — CONSUMER INTELLIGENCE (B2C)"
_COVER_KICKER = "CONSUMER INTELLIGENCE · B2C SEGMENTATION SURVEY"
_FOOTER_LEFT = "Example data — for survey-design review, not a fielded result."

# Spec question types -> instruction text + sum semantics.
_INSTRUCTION = {
    "single_select": "Select one.",
    "multi_select": "Select all that apply.",
    "rating_0_10": "Select one.",
}


def header_footer_b2c(canvas, doc, industry):
    canvas.saveState()
    canvas.setStrokeColor(LIGHT)
    canvas.setLineWidth(0.6)
    canvas.line(18 * mm, 285 * mm, 192 * mm, 285 * mm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(GREY)
    canvas.drawString(18 * mm, 287 * mm, _RUNNING_HEADER)
    canvas.drawRightString(192 * mm, 287 * mm, industry)
    canvas.line(18 * mm, 14 * mm, 192 * mm, 14 * mm)
    canvas.drawString(18 * mm, 9 * mm, _FOOTER_LEFT)
    canvas.drawRightString(192 * mm, 9 * mm, f"Page {doc.page}")
    canvas.restoreState()


def _chart_type_for(qtype: str, n_opts: int) -> str:
    if qtype == "multi_select":
        return "horizontal_bar"
    if qtype == "rating_0_10":
        return "vertical_bar"
    return "horizontal_bar" if n_opts > 5 else "donut"


def build(doc_json: dict, out_path: str):
    scope = doc_json.get("surveyScope") or {}
    personas = doc_json.get("behaviouralPersonas") or []
    structure = doc_json.get("structure") or {}
    segments = doc_json.get("segments") or []

    industry = scope.get("category") or scope.get("title") or "Market"
    region = scope.get("region") or ""

    doc = SimpleDocTemplate(
        out_path, pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=22 * mm, bottomMargin=18 * mm,
        title=f"{industry} — Consumer Survey (B2C)",
    )
    story = []

    # -- Cover: Title -----------------------------------------------------
    story.append(Spacer(1, 34 * mm))
    story.append(Paragraph(
        _COVER_KICKER,
        ParagraphStyle("kick", parent=H_SUB, fontName="Helvetica-Bold",
                       textColor=BLUE, fontSize=11)))
    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph(esc(scope.get("title") or industry), H_TITLE))
    story.append(Spacer(1, 5 * mm))

    # -- Cover: Survey Scope heading + definition paragraph ---------------
    if scope.get("definition"):
        story.append(Paragraph("Survey Scope", ParagraphStyle(
            "ScopeH", parent=H_SUB, fontName="Helvetica-Bold", textColor=NAVY, fontSize=11)))
        story.append(Spacer(1, 2 * mm))
        story.append(Paragraph(esc(scope["definition"]), BODY))
        story.append(Spacer(1, 6 * mm))

    # -- Cover: Category / Region / Audience Type table (no Study Purpose) -
    meta_rows = [
        ["Category", esc(industry)],
        ["Region", esc(region) or "—"],
        ["Audience Type", esc(scope.get("audienceType") or "B2C Consumer")],
    ]
    mt = Table(meta_rows, colWidths=[42 * mm, 132 * mm])
    mt.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("TEXTCOLOR", (0, 0), (0, -1), NAVY),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, LIGHT),
    ]))
    story.append(mt)
    story.append(Spacer(1, 6 * mm))

    # -- Cover: Customers We Surveyed --------------------------------------
    if scope.get("customersWeSurveyed"):
        story.append(Paragraph("Customers We Surveyed", ParagraphStyle(
            "CwsH", parent=H_SUB, fontName="Helvetica-Bold", textColor=NAVY, fontSize=11)))
        story.append(Spacer(1, 2 * mm))
        for c in scope["customersWeSurveyed"]:
            story.append(Paragraph(f"• {esc(c)}", BODY))
        story.append(Spacer(1, 6 * mm))

    # -- Cover: What This Survey Deliberately Does Not Ask -----------------
    if scope.get("whatIsNotAsked"):
        story.append(Paragraph("What This Survey Deliberately Does Not Ask", ParagraphStyle(
            "WinaH", parent=H_SUB, fontName="Helvetica-Bold", textColor=NAVY, fontSize=11)))
        story.append(Spacer(1, 2 * mm))
        for w in scope["whatIsNotAsked"]:
            story.append(Paragraph(f"• {esc(w)}", BODY))
        story.append(Spacer(1, 6 * mm))

    # -- Cover: How to Read the Numbers ------------------------------------
    if scope.get("howToReadTheNumbers"):
        story.append(Paragraph("How to Read the Numbers", ParagraphStyle(
            "HtrH", parent=H_SUB, fontName="Helvetica-Bold", textColor=NAVY, fontSize=11)))
        story.append(Spacer(1, 2 * mm))
        story.append(Paragraph(esc(scope["howToReadTheNumbers"]), BODY))

    # -- Behavioural Personas page -----------------------------------------
    if personas:
        story.append(PageBreak())
        story.append(seg_banner("Behavioural Personas"))
        story.append(Spacer(1, 4 * mm))
        rows = [
            [Paragraph(esc(p.get("name", "")), KV),
             Paragraph(esc(p.get("description", "")), BODY)]
            for p in personas
        ]
        pt = Table(rows, colWidths=[42 * mm, 132 * mm])
        pt.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LINEBELOW", (0, 0), (-1, -2), 0.35, LIGHT),
        ]))
        story.append(pt)

    # -- Each segment, in order ---------------------------------------------
    for seg in segments:
        story.append(PageBreak())
        story.append(seg_banner(esc(seg.get("title") or "")))
        story.append(Spacer(1, 4 * mm))
        if seg.get("focus"):
            story.append(Paragraph(esc(seg["focus"]), SECTION_LEDE))
            story.append(Spacer(1, 2 * mm))
        for q in seg.get("questions") or []:
            qtype = q.get("type") or "single_select"
            options = q.get("options") or []
            instruction = _INSTRUCTION.get(qtype, "Select one.")
            block = [Paragraph(
                f"{q.get('id', '')}. {esc(q.get('text', ''))}  "
                f"<font size='7' color='#5A6473'>[{esc(instruction)}]</font>",
                Q_TEXT,
            )]
            block.append(Spacer(1, 1.5 * mm))
            if options:
                chart_type = _chart_type_for(qtype, len(options))
                chart_q = {"options": options, "chart_type": chart_type}
                block.append(make_chart(chart_q, 0))
            if q.get("key_insight"):
                block.append(Paragraph(f"Key Insights: {esc(q['key_insight'])}", INSIGHT))
            block.append(Spacer(1, 4 * mm))
            story.append(KeepTogether(block))

    # -- Optional: Survey Structure summary page -----------------------------
    struct_sections = structure.get("sections") or []
    if struct_sections:
        story.append(PageBreak())
        story.append(seg_banner("Survey Structure"))
        story.append(Spacer(1, 4 * mm))
        story.append(Paragraph(
            f"Total questions: {structure.get('totalQuestions', 0)}",
            ParagraphStyle("TotQ", parent=BODY, fontName="Helvetica-Bold", textColor=NAVY),
        ))
        story.append(Spacer(1, 3 * mm))
        rows = [[Paragraph(esc(s.get("label", "")), KV),
                 Paragraph(f"{s.get('questions', 0)} questions", KV)]
                for s in struct_sections]
        struct_t = Table(rows, colWidths=[142 * mm, 32 * mm])
        struct_t.setStyle(TableStyle([
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LINEBELOW", (0, 0), (-1, -2), 0.3, LIGHT),
        ]))
        story.append(struct_t)

    doc.build(
        story,
        onFirstPage=lambda c, d: header_footer_b2c(c, d, industry),
        onLaterPages=lambda c, d: header_footer_b2c(c, d, industry),
    )


def main():
    if len(sys.argv) < 2:
        print("Usage: python make_pdfs_b2c.py <path-to-b2c-survey.json>")
        return
    target = sys.argv[1]
    files = [target] if os.path.isfile(target) else sorted(glob.glob(os.path.join(target, "*.json")))
    if not files:
        print(f"No JSON found at {target}")
        return
    for f in files:
        with open(f, encoding="utf-8-sig") as fh:
            data = json.load(fh)
        out = f.rsplit(".json", 1)[0] + ".pdf"
        build(data, out)
        print(f"  wrote {out}")
    print(f"\nDone. {len(files)} PDF(s) written.")


if __name__ == "__main__":
    main()
