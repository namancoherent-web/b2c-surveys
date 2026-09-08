#!/usr/bin/env bash
# Re-run every market with adaptive sections, one region each (RUN_REGIONS),
# so each produces a global.json. Sequential: parallel jobs share one DDG
# egress IP and get throttled, which guts the evidence pool.
set -u
cd "$(dirname "$0")/.." || exit 1

MARKETS=(
  "Video Streaming Subscriptions"
  "Running Shoes"
  "Organic Milk"
  "Smartwatches"
)

mkdir -p logs
SUMMARY="logs/rerun_sections_summary.txt"
: > "$SUMMARY"

echo "start: $(date)" | tee -a "$SUMMARY"
for seg in "${MARKETS[@]}"; do
  slug=$(echo "$seg" | tr '[:upper:] ' '[:lower:]-')
  log="logs/rerun_${slug}.log"
  echo "=== START ${seg} :: $(date +%H:%M:%S) ===" | tee -a "$SUMMARY"
  start=$(date +%s)

  PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -u run.py "$seg" > "$log" 2>&1
  code=$?

  elapsed=$(( $(date +%s) - start ))
  secs=$(grep -m1 '✓ story_planner' "$log" 2>/dev/null || echo '')
  echo "=== DONE  ${seg} :: exit=${code} :: ${elapsed}s ===" | tee -a "$SUMMARY"
  [ -n "$secs" ] && echo "    ${secs}" | tee -a "$SUMMARY"
  grep -E 'Questions:|Run failed|REJECTED' "$log" | tail -2 | tee -a "$SUMMARY"
  echo | tee -a "$SUMMARY"
done
echo "end: $(date)" | tee -a "$SUMMARY"
