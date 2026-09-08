"""Trim published surveys so the UI shows ONLY questions present in the PDFs.

# change for b2c questionarie

The PDFs are the reviewed artefact. Any question that has since been swapped in
(e.g. by a re-run of the A6 global selection) is not in the PDF a reviewer read,
so it should not appear in the UI either. This reads each PDF's actual text and
drops any published question whose wording is not found there.

Only removes; never adds. Questions the PDF has but the JSON does not are
reported, not restored — the JSON is treated as authoritative for removals that
were made deliberately.

Usage:
    python scripts/sync_ui_to_pdfs.py --dry-run
    python scripts/sync_ui_to_pdfs.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pypdf import PdfReader  # noqa: E402

PDF_DIR = ROOT / "output" / "pdfs"


def norm(text: str) -> str:
    """Squash to comparable form — PDFs re-wrap lines and lose exact spacing."""
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def pdf_text(path: Path) -> str:
    reader = PdfReader(str(path))
    return norm(" ".join((p.extract_text() or "") for p in reader.pages))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "published_surveys"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    files = [f for f in sorted(Path(args.src).glob("*/*.json")) if f.name != "index.json"]
    total_dropped = 0
    missing_pdf = []

    for f in files:
        pdf = PDF_DIR / f"{f.parent.name}__{f.stem}.pdf"
        if not pdf.is_file():
            missing_pdf.append(f"{f.parent.name}/{f.stem}")
            continue

        haystack = pdf_text(pdf)
        payload = json.loads(f.read_text(encoding="utf-8"))
        sections = ((payload.get("frontend") or {}).get("sections") or [])
        dropped = []

        for section in sections:
            keep = []
            for q in section.get("questions") or []:
                # Match on a distinctive slice — the PDF wraps long stems.
                probe = norm(q.get("question") or "")[:70]
                if probe and probe in haystack:
                    keep.append(q)
                else:
                    dropped.append((q.get("qNum"), q.get("question") or ""))
            section["questions"] = keep

        if not dropped:
            continue

        n = 0
        for section in sections:
            for i, q in enumerate(section["questions"], start=1):
                n += 1
                q["qNum"] = n
                q["id"] = f"q{n}"
                q["narrativeOrder"] = i
                q["funnelPosition"] = i

        meta = payload.setdefault("metadata", {})
        meta["total_questions"] = n
        definition = payload.get("surveyDefinition") or {}
        if definition.get("structure"):
            definition["structure"]["totalQuestions"] = n
            for s in definition["structure"].get("sections") or []:
                m = next((x for x in sections if x.get("id") == s.get("id")), None)
                if m:
                    s["questions"] = len(m["questions"])

        total_dropped += len(dropped)
        print(f"{'(dry) ' if args.dry_run else ''}{f.parent.name}/{f.stem}  -> {n} questions")
        for num, text in dropped:
            print(f"    - dropped Q{num} (not in PDF): {text[:70]}")
        if not args.dry_run:
            f.write_text(
                json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

    print(f"\n{total_dropped} question(s) removed across {len(files)} file(s)")
    if missing_pdf:
        print(f"no PDF for: {', '.join(missing_pdf)}")
    if args.dry_run:
        print("(dry run — nothing written)")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    main()
