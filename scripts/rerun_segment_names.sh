#!/usr/bin/env bash
# Second pass — Video Streaming and Running Shoes were planned before the
# segment-naming rule (storyline_v6) landed, so their blueprints still carry
# the old marketing-register names. Organic Milk and Smartwatches pick v6 up
# on their first run and are not repeated here.
#
# Waits for the first batch to finish so the two runs never share an egress IP.
set -u
cd "$(dirname "$0")/.." || exit 1

SUMMARY="logs/rerun_sections_summary.txt"
while ! grep -q '^end:' "$SUMMARY" 2>/dev/null; do sleep 30; done

MARKETS=("Video Streaming Subscriptions" "Running Shoes")
OUT="logs/rerun_segnames_summary.txt"
: > "$OUT"

echo "start: $(date)" | tee -a "$OUT"
for seg in "${MARKETS[@]}"; do
  slug=$(echo "$seg" | tr '[:upper:] ' '[:lower:]-')
  log="logs/rerun_segnames_${slug}.log"
  echo "=== START ${seg} :: $(date +%H:%M:%S) ===" | tee -a "$OUT"
  start=$(date +%s)

  PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -u run.py "$seg" > "$log" 2>&1
  code=$?

  echo "=== DONE  ${seg} :: exit=${code} :: $(( $(date +%s) - start ))s ===" | tee -a "$OUT"
  grep -E 'Questions:|Run failed|REJECTED' "$log" | tail -2 | tee -a "$OUT"
done
echo "end: $(date)" | tee -a "$OUT"
