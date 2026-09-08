#!/usr/bin/env bash
# Definitive 5-market run. Every market uses the SAME code — earlier passes had
# fixes landing mid-run, which made results incomparable across markets.
#
# Fix set: storyline_v8 / questionnaire_v14
#   age+income claim made true      persona archetypes scrubbed
#   question types + sum labels     no "not a user" options
#   NPS recomputed vs satisfaction  price coverage enforced
#   intent-based redundancy         unfieldable grid guard
#   per-category must-cover topics
set -u
cd "$(dirname "$0")/.." || exit 1

MARKETS=(
  "Video Streaming Subscriptions"
  "Running Shoes"
  "Organic Milk"
  "Smartwatches"
  "Meal Kit Delivery Services"
)

SUMMARY="logs/final5_summary.txt"
: > "$SUMMARY"
echo "start: $(date)" | tee -a "$SUMMARY"
for seg in "${MARKETS[@]}"; do
  slug=$(echo "$seg" | tr '[:upper:] ' '[:lower:]-')
  echo "=== START ${seg} :: $(date +%H:%M:%S) ===" | tee -a "$SUMMARY"
  PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -u run.py "$seg" \
    > "logs/final5_${slug}.log" 2>&1
  code=$?
  echo "=== DONE  ${seg} :: exit=${code} ===" | tee -a "$SUMMARY"
  grep -E 'Questions:|Run failed|REJECTED' "logs/final5_${slug}.log" | tail -1 | tee -a "$SUMMARY"
done
echo "end: $(date)" | tee -a "$SUMMARY"
