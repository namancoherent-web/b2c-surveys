#!/usr/bin/env bash
# Ten fresh B2C markets, deliberately spread across category shapes so the
# planner's lens choice, storyline and must-cover topics have to genuinely
# adapt rather than reuse a template:
#
#   consumable / FMCG      Pet Food, Bottled Water
#   beauty / regimen       Skincare Serums
#   appliance / durable    Home Coffee Machines, Mattresses
#   apparel / fashion      Athletic Apparel
#   big-ticket considered  Electric Vehicles
#   service app            Ride-Hailing Apps, Online Grocery Delivery
#   digital entertainment  Mobile Gaming
#
# Sequential: parallel jobs share one search egress IP and get throttled,
# which guts the evidence pool.
set -u
cd "$(dirname "$0")/.." || exit 1

MARKETS=(
  "Athletic Apparel"
  "Pet Food"
  "Home Coffee Machines"
  "Skincare Serums"
  "Electric Vehicles"
  "Ride-Hailing Apps"
  "Mobile Gaming"
  "Bottled Water"
  "Mattresses"
  "Online Grocery Delivery"
)

SUMMARY="logs/ten_summary.txt"
: > "$SUMMARY"
echo "start: $(date)" | tee -a "$SUMMARY"
i=0
for seg in "${MARKETS[@]}"; do
  i=$((i+1))
  slug=$(echo "$seg" | tr '[:upper:] ' '[:lower:]-')
  echo "=== START ${i}/10 ${seg} :: $(date +%H:%M:%S) ===" | tee -a "$SUMMARY"
  PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -u run.py "$seg" \
    > "logs/ten_${slug}.log" 2>&1
  code=$?
  echo "=== DONE  ${i}/10 ${seg} :: exit=${code} ===" | tee -a "$SUMMARY"
  grep -E 'Questions:|Run failed|REJECTED|NEEDS CLARIFICATION' "logs/ten_${slug}.log" \
    | tail -1 | tee -a "$SUMMARY"
done
echo "end: $(date)" | tee -a "$SUMMARY"
