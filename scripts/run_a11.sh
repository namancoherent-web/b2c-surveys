#!/usr/bin/env bash
# A11 regeneration — clarity-tuned questions, segmentation lens, and a scoring
# model whose weights point at published question ids.
#
# Run from a script file rather than an inline `bash -c`: two inline attempts
# were killed mid-pipeline, and a detached script has been reliable here.
set -u
cd "$(dirname "$0")/.." || exit 1

MARKETS=("Video Streaming Subscriptions" "Running Shoes")
SUMMARY="logs/a11_summary.txt"
: > "$SUMMARY"

echo "start: $(date)" | tee -a "$SUMMARY"
for seg in "${MARKETS[@]}"; do
  slug=$(echo "$seg" | tr '[:upper:] ' '[:lower:]-')
  log="logs/a11_${slug}.log"
  echo "=== START ${seg} :: $(date +%H:%M:%S) ===" | tee -a "$SUMMARY"

  PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -u run.py "$seg" > "$log" 2>&1
  code=$?

  echo "=== DONE  ${seg} :: exit=${code} ===" | tee -a "$SUMMARY"
  grep -E 'Questions:|Run failed|REJECTED' "$log" | tail -2 | tee -a "$SUMMARY"
done
echo "end: $(date)" | tee -a "$SUMMARY"
