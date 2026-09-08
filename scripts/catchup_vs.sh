#!/usr/bin/env bash
# Video Streaming ran first, before must_cover_topics reached metadata and
# before the global rebuild carried the response-type fields. Re-run it alone
# once the batch ends so all five markets are on identical code.
set -u
cd "$(dirname "$0")/.." || exit 1
while ! grep -q '^end:' logs/final5_summary.txt 2>/dev/null; do sleep 30; done
echo "=== CATCHUP Video Streaming :: $(date +%H:%M:%S) ===" >> logs/final5_summary.txt
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe -u run.py "Video Streaming Subscriptions" \
  > logs/final5_video-streaming-subscriptions.log 2>&1
echo "=== CATCHUP DONE :: exit=$? ===" >> logs/final5_summary.txt
echo "allfive: $(date)" >> logs/final5_summary.txt
