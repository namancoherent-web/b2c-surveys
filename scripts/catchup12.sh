#!/usr/bin/env bash
# Markets 1-2 (Video Streaming, Running Shoes) started before two late patches
# landed: must_cover_topics reaching metadata, and the global rebuild carrying
# the response-type fields. Re-run just those two after the batch so all five
# markets are on identical code. Markets 3-5 already have everything.
#
# Cheap: the storyline and questionnaire caches are warm, so this mostly
# re-publishes rather than re-generating.
set -u
cd "$(dirname "$0")/.." || exit 1
while ! grep -q '^end:' logs/final5_summary.txt 2>/dev/null; do sleep 30; done

for seg in "Video Streaming Subscriptions" "Running Shoes"; do
  slug=$(echo "$seg" | tr '[:upper:] ' '[:lower:]-')
  echo "=== CATCHUP ${seg} :: $(date +%H:%M:%S) ===" >> logs/final5_summary.txt
  PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -u run.py "$seg" \
    > "logs/final5_${slug}.log" 2>&1
  echo "=== CATCHUP DONE ${seg} :: exit=$? ===" >> logs/final5_summary.txt
done
echo "allfive: $(date)" >> logs/final5_summary.txt
