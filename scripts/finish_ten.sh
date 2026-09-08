#!/usr/bin/env bash
# Post-processing for the 10-market run. Waits for the generator, then:
#   1. repairs any segment names the naming rules did not catch at write time
#      (a market that started seconds after a rule landed imports the old module)
#   2. exports the global PDF + markdown deliverable for each market
#   3. runs the pre-output validation gate over every published file
set -u
cd "$(dirname "$0")/.." || exit 1

while ! grep -q '^end:' logs/ten_summary.txt 2>/dev/null; do sleep 30; done

MARKETS=(athletic-apparel pet-food home-coffee-machines skincare-serums
         electric-vehicles ride-hailing-apps mobile-gaming bottled-water
         mattresses online-grocery-delivery)

echo "=== repairing segment names ==="
PYTHONPATH=. ./.venv/Scripts/python.exe scripts/repair_segment_names.py
PYTHONPATH=. ./.venv/Scripts/python.exe scripts/repair_display_base.py

echo "=== exporting global PDFs ==="
mkdir -p output/pdfs output/markdown
for m in "${MARKETS[@]}"; do
  f="published_surveys/${m}/global.json"
  [ -f "$f" ] || { echo "  MISSING ${m}"; continue; }
  ./.venv/Scripts/python.exe scripts/export_survey_pdf.py "$f" >/dev/null 2>&1
  [ -f output/global.pdf ] && mv -f output/global.pdf "output/pdfs/${m}__global.pdf" \
    && echo "  wrote ${m}__global.pdf"
  PYTHONPATH=. ./.venv/Scripts/python.exe scripts/export_survey_markdown.py "$f" >/dev/null 2>&1
done

echo "=== validation gate ==="
PYTHONPATH=. ./.venv/Scripts/python.exe scripts/validate_survey.py \
  published_surveys/*/global.json 2>&1 | tail -25
echo "FINISHED: $(date)"
