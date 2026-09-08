"""E2E: Organic Milk — Asia Pacific + North America + global (DDG/Jina on)."""
from __future__ import annotations

import asyncio
import json
import sys

from src.tasks import (
    classify_segment_async,
    run_global_selection,
    run_one_region_async,
)


def _progress(node: str, update: dict) -> None:
    try:
        from run import _progress_line
        print(_progress_line(node, update), flush=True)
    except Exception:  # noqa: BLE001
        print(f"[ok] {node}", flush=True)


async def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    segment = "Organic Milk"
    # change for b2c questionarie — classify once, then subset region jobs
    print(f"\n-- Classify (B2C gate) for {segment!r} --", flush=True)
    clf = await classify_segment_async(segment, on_progress=_progress)
    if clf.get("status") == "rejected":
        print(json.dumps(clf, indent=2), flush=True)
        return
    classification = clf["classification"]

    regions = ["Asia Pacific", "North America"]
    results = []
    for i, r in enumerate(regions, 1):
        print(f"\n-- Region job {i}/{len(regions)}: {r} --", flush=True)
        results.append(
            await run_one_region_async(
                segment, r, classification, on_progress=_progress,
            )
        )
        print(
            json.dumps(
                {
                    k: results[-1].get(k)
                    for k in (
                        "region",
                        "region_slug",
                        "questions",
                        "core_questions",
                        "module_questions",
                        "published",
                        "status",
                    )
                },
                indent=2,
            ),
            flush=True,
        )

    print(f"\n-- Global selection (A6) for {segment!r} --", flush=True)
    g = run_global_selection(segment)
    print(json.dumps(g, indent=2, default=str), flush=True)
    slug = g.get("slug") or "organic-milk"
    print("\nDONE", flush=True)
    print("Preview:", flush=True)
    for rs in ("asia-pacific", "north-america", "global"):
        print(f"  http://localhost:3000/survey/{slug}/{rs}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
