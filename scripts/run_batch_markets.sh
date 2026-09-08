#!/usr/bin/env bash
# Sequential multi-market run. Sequential on purpose: parallel jobs share one
# DuckDuckGo egress IP and get blocked, which silently guts the evidence pool.
set -u
cd "$(dirname "$0")/.." || exit 1

MARKETS=(
  "Video Streaming Subscriptions"
  "Running Shoes"
  "Organic Milk"
)

mkdir -p logs
STAMP=$(date +%Y%m%d_%H%M)
SUMMARY="logs/batch_${STAMP}_summary.txt"

echo "batch start: $(date)" | tee "$SUMMARY"
echo "markets: ${#MARKETS[@]}  regions: all 5 + global" | tee -a "$SUMMARY"
echo | tee -a "$SUMMARY"

for seg in "${MARKETS[@]}"; do
  slug=$(echo "$seg" | tr '[:upper:] ' '[:lower:]-')
  log="logs/run_${slug}.log"
  echo "=== START ${seg} :: $(date +%H:%M:%S) ===" | tee -a "$SUMMARY"
  start=$(date +%s)

  PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -u run.py "$seg" > "$log" 2>&1
  code=$?

  elapsed=$(( $(date +%s) - start ))
  qs=$(grep -c '^✓ question_architect' "$log" 2>/dev/null || echo 0)
  final=$(grep -E 'Questions:|Total questions:' "$log" | tail -1)
  echo "=== DONE  ${seg} :: exit=${code} :: ${elapsed}s :: ${final} ===" | tee -a "$SUMMARY"
  grep -E '^✗|REJECTED|Run failed' "$log" | head -3 | tee -a "$SUMMARY"
  echo | tee -a "$SUMMARY"
done

echo "batch end: $(date)" | tee -a "$SUMMARY"
echo "--- artifacts ---" | tee -a "$SUMMARY"
find published_surveys -name '*.json' | sort | tee -a "$SUMMARY"
