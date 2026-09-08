"""Copy published surveys into the Next.js app's public/ dir + build a manifest.

# change for b2c questionarie

Files under ``public/`` are served at the site root, so

    frontend-v2/apps/web/public/surveys/smartwatches/europe.json
        -> GET /surveys/smartwatches/europe.json

The manifest (``public/surveys/index.json``) is what the /surveys gallery page
lists, so the UI never has to guess which markets or regions exist.

Usage:
    python scripts/publish_to_web_public.py
    python scripts/publish_to_web_public.py --web-root <path-to>/frontend-v2/apps/web
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.question_plan import PUBLISH_REGION_SLUGS  # noqa: E402

DEFAULT_WEB_ROOT = Path(
    r"C:\Users\vimarsh.CMI\cmi-platform-ai\frontend-v2\apps\web"
)

_REGION_LABELS = {
    "global": "Global",
    "north-america": "North America",
    "europe": "Europe",
    "asia-pacific": "Asia Pacific",
    "latin-america": "Latin America",
    "middle-east-africa": "Middle East & Africa",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--web-root", default=str(DEFAULT_WEB_ROOT))
    ap.add_argument("--src", default=str(ROOT / "published_surveys"))
    args = ap.parse_args()

    src = Path(args.src)
    web_root = Path(args.web_root)
    if not src.is_dir():
        print(f"no published surveys at {src}")
        raise SystemExit(1)
    if not (web_root / "public").is_dir():
        print(f"not a Next.js web root (no public/): {web_root}")
        raise SystemExit(1)

    dest_root = web_root / "public" / "surveys"
    dest_root.mkdir(parents=True, exist_ok=True)

    surveys = []
    for folder in sorted(p for p in src.iterdir() if p.is_dir()):
        regions = []
        dest = dest_root / folder.name
        dest.mkdir(parents=True, exist_ok=True)
        segment = folder.name
        total_by_region: dict[str, int] = {}

        for region in PUBLISH_REGION_SLUGS:
            f = folder / f"{region}.json"
            if not f.is_file():
                continue
            payload = json.loads(f.read_text(encoding="utf-8"))
            segment = payload.get("segment") or segment
            n_q = sum(
                len(s.get("questions") or [])
                for s in ((payload.get("frontend") or {}).get("sections") or [])
            )
            total_by_region[region] = n_q
            shutil.copy2(f, dest / f"{region}.json")
            regions.append({
                "slug": region,
                "label": _REGION_LABELS.get(region, region),
                "questions": n_q,
                "path": f"/surveys/{folder.name}/{region}.json",
            })

        if not regions:
            continue

        # Pull headline context from whichever region we have.
        first = json.loads(
            (dest / f"{regions[0]['slug']}.json").read_text(encoding="utf-8")
        )
        definition = first.get("surveyDefinition") or {}
        surveys.append({
            "slug": folder.name,
            "segment": segment,
            "category": definition.get("category") or segment,
            "audienceType": definition.get("audienceType") or "B2C consumer",
            "summary": definition.get("whatThisSurveyIs") or "",
            "regions": regions,
            "totalQuestions": sum(total_by_region.values()),
        })

    manifest = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "count": len(surveys),
        "surveys": surveys,
    }
    (dest_root / "index.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    print(f"published {len(surveys)} survey(s) -> {dest_root}")
    for s in surveys:
        rs = ", ".join(r["slug"] for r in s["regions"])
        print(f"  {s['slug']:<34} {len(s['regions'])} region(s): {rs}")
    print(f"\nmanifest: {dest_root / 'index.json'}")
    print("browse at: http://localhost:3000/surveys")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    main()
