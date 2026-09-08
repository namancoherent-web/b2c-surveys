#!/usr/bin/env bash
# Final A12 pass across five markets, with the complete fix set:
#   - age/income claim made true          - persona archetypes scrubbed
#   - question types + response labels    - no "not a user" options
#   - NPS recomputed vs satisfaction      - price coverage enforced
#   - intent-based redundancy detection   - unfieldable grid guard
#
# Five categories chosen to exercise different shapes: a subscription service,
# a durable, a consumable, a device, and a recurring delivery service — the
# storyline and segmentation lens should differ across them.
set -u
cd "$(dirname "$0")/.." || exit 1

# Wait for any in-flight run so two jobs never share the search egress IP.
# (no in-flight run to wait for)

MARKETS=(
  "Video Streaming Subscriptions"
  "Running Shoes"
  "Organic Milk"
  "Smartwatches"
  "Meal Kit Delivery Services"
)

SUMMARY="logs/a12_summary.txt"
: > "$SUMMARY"
echo "start: $(date)" | tee -a "$SUMMARY"
for seg in "${MARKETS[@]}"; do
  slug=$(echo "$seg" | tr '[:upper:] ' '[:lower:]-')
  echo "=== START ${seg} :: $(date +%H:%M:%S) ===" | tee -a "$SUMMARY"
  PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -u run.py "$seg" \
    > "logs/a12_${slug}.log" 2>&1
  echo "=== DONE  ${seg} :: exit=$? ===" | tee -a "$SUMMARY"
  grep -E 'Questions:|Run failed|REJECTED' "logs/a12_${slug}.log" | tail -1 | tee -a "$SUMMARY"
done
echo "end: $(date)" | tee -a "$SUMMARY"
