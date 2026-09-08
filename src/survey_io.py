"""Survey output paths, slugging, and frontend publish helpers.

# change for b2c questionarie — Phase A7 per-region folder + legacy flat file

Layout under FRONTEND_SURVEYS_DIR:
    <slug>.json                         # legacy (survey-preview) — kept for now
    <slug>/index.json                   # {segment, slug, regions, generatedAt}
    <slug>/global.json
    <slug>/north-america.json
    ...
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from typing import Any

from src.question_plan import PUBLISH_REGION_SLUGS

OUTPUT_DIR = "output"

# Where the cmi-platform frontend serves generated surveys from.
FRONTEND_SURVEYS_DIR = os.getenv(
    "FRONTEND_SURVEYS_DIR",
    os.path.join("..", "cmi-platform-ai", "frontend-v2", "apps", "web", "public", "surveys"),
)


def slugify(text: str) -> str:
    """Lowercase, non-alphanumerics → hyphens, collapse repeats, strip ends."""
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower())
    s = re.sub(r"-+", "-", s).strip("-")
    return s or "survey"


def unique_output_path(slug: str) -> str:
    """Return a non-colliding output path; add a timestamp (then counter) on repeat."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base = os.path.join(OUTPUT_DIR, f"{slug}.json")
    if not os.path.exists(base):
        return base
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M")
    candidate = os.path.join(OUTPUT_DIR, f"{slug}_{stamp}.json")
    i = 1
    while os.path.exists(candidate):
        candidate = os.path.join(OUTPUT_DIR, f"{slug}_{stamp}_{i}.json")
        i += 1
    return candidate


def atomic_write_text(path: str, text: str) -> None:
    """Write via temp file + replace so readers never see a partial JSON."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    directory = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _insight_from_data(data: list) -> str:
    if not data:
        return ""
    top = max(data, key=lambda d: d.get("value") or 0)
    return f"Top response: {top.get('label')} ({float(top.get('value') or 0):.2f}%)."


# change for b2c questionarie -- CHECK (B2C_MASTER_RULES_FINAL /
# JSON_FORMAT_SPECIFICATION): the delivered survey shape is now
# surveyScope/behaviouralPersonas/structure/segments, one region per file
# (no more dataByRegion multi-region blob to project a single region out
# of -- each generation job already produces exactly the one region/country
# it was run for, so there is nothing left for this function to pick out).
def build_region_payload(
    survey: dict,
    *,
    segment: str,
    slug: str,
    region_slug: str,
) -> dict[str, Any]:
    """One file under public/surveys/<slug>/<region>.json."""
    return {
        "segment": segment,
        "slug": slug,
        "region": region_slug,
        "surveyScope": survey.get("surveyScope") or {},
        "behaviouralPersonas": survey.get("behaviouralPersonas") or [],
        "structure": survey.get("structure") or {},
        "segments": survey.get("segments") or [],
    }


def _frontend_parent_ready() -> bool:
    parent = os.path.dirname(os.path.abspath(FRONTEND_SURVEYS_DIR))
    return os.path.isdir(parent)


def write_index_json(
    folder: str,
    *,
    segment: str,
    slug: str,
    regions: list[str],
    generated_at: str | None = None,
) -> str:
    """Atomically write/replace index.json listing ready regions."""
    payload = {
        "segment": segment,
        "slug": slug,
        "regions": list(regions),
        "generatedAt": generated_at
        or datetime.now(timezone.utc).isoformat(),
    }
    path = os.path.join(folder, "index.json")
    atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False))
    return path


def update_index_add_region(
    folder: str,
    *,
    segment: str,
    slug: str,
    region_slug: str,
) -> str:
    """Read-modify-write index.json to add one region (idempotent)."""
    path = os.path.join(folder, "index.json")
    regions: list[str] = []
    generated_at = datetime.now(timezone.utc).isoformat()
    if os.path.isfile(path):
        try:
            prev = json.loads(open(path, encoding="utf-8").read())
            regions = list(prev.get("regions") or [])
            generated_at = prev.get("generatedAt") or generated_at
            segment = prev.get("segment") or segment
            slug = prev.get("slug") or slug
        except (json.JSONDecodeError, OSError):
            pass
    if region_slug not in regions:
        # Keep canonical order from PUBLISH_REGION_SLUGS when possible.
        regions.append(region_slug)
        order = {s: i for i, s in enumerate(PUBLISH_REGION_SLUGS)}
        regions = sorted(regions, key=lambda r: order.get(r, 999))
    return write_index_json(
        folder, segment=segment, slug=slug, regions=regions, generated_at=generated_at,
    )


def publish_region_file(
    slug: str,
    region_slug: str,
    payload: dict,
    *,
    segment: str,
) -> str | None:
    """Write one <slug>/<region>.json and update index.json. Returns path or None."""
    if not _frontend_parent_ready():
        return None
    folder = os.path.join(FRONTEND_SURVEYS_DIR, slug)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{region_slug}.json")
    atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False))
    update_index_add_region(folder, segment=segment, slug=slug, region_slug=region_slug)
    return path


def publish_to_frontend(slug: str, json_text: str) -> str | None:
    """Legacy: copy the full survey JSON to public/surveys/<slug>.json.

    Kept for /survey-preview/[slug] until Phase B5.
    """
    if not _frontend_parent_ready():
        return None
    os.makedirs(FRONTEND_SURVEYS_DIR, exist_ok=True)
    path = os.path.join(FRONTEND_SURVEYS_DIR, f"{slug}.json")
    atomic_write_text(path, json_text)
    return path


def publish_survey_bundle(
    slug: str,
    survey: dict,
    *,
    segment: str,
    json_text: str | None = None,
) -> dict[str, Any]:
    """Publish legacy flat file + per-region folder + index.

    # change for b2c questionarie — Phase A7
    Regions may still share the same questions until A4b/A5; percentages
    already differ via dataByRegion projection.
    """
    result: dict[str, Any] = {
        "legacy": None,
        "folder": None,
        "index": None,
        "regions": {},
    }
    if not _frontend_parent_ready():
        return result

    text = json_text if json_text is not None else json.dumps(
        survey, indent=2, ensure_ascii=False,
    )
    result["legacy"] = publish_to_frontend(slug, text)

    folder = os.path.join(FRONTEND_SURVEYS_DIR, slug)
    os.makedirs(folder, exist_ok=True)
    result["folder"] = folder

    ready: list[str] = []
    for region_slug in PUBLISH_REGION_SLUGS:
        payload = build_region_payload(
            survey, segment=segment, slug=slug, region_slug=region_slug,
        )
        path = os.path.join(folder, f"{region_slug}.json")
        atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False))
        result["regions"][region_slug] = path
        ready.append(region_slug)

    result["index"] = write_index_json(
        folder,
        segment=segment,
        slug=slug,
        regions=ready,
    )
    return result
