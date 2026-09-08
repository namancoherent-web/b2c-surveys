"""Phase C — slug audit: agent PUBLISH_REGION_SLUGS == frontend REGIONS.

# change for b2c questionarie

Parses cmi-platform-ai geographies.ts and asserts byte-identical region slugs
against survey_agent question_plan.PUBLISH_REGION_SLUGS.
"""
from __future__ import annotations

import re
from pathlib import Path

from src.question_plan import PUBLISH_REGION_SLUGS, REGION_BLEND_WEIGHTS, REGION_SLUGS

# survey_agent/ -> coherent_b2c_survey_question_2/
_REPO_ROOT = Path(__file__).resolve().parents[2]
_GEO_TS = (
    _REPO_ROOT
    / "cmi-platform-ai"
    / "frontend-v2"
    / "apps"
    / "web"
    / "src"
    / "lib"
    / "geographies.ts"
)

_CANONICAL = [
    "global",
    "north-america",
    "europe",
    "asia-pacific",
    "latin-america",
    "middle-east-africa",
]


def _frontend_region_slugs(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    # Prefer SURVEY_REGION_SLUGS if present; else parse REGIONS array slugs
    # in order until COUNTRIES section.
    m = re.search(
        r"export const REGIONS: Geo\[\] = \[([\s\S]*?)\];",
        text,
    )
    if not m:
        raise AssertionError(f"Could not find REGIONS in {path}")
    block = m.group(1)
    return re.findall(r'slug:\s*"([^"]+)"', block)


def test_agent_matches_canonical():
    assert list(PUBLISH_REGION_SLUGS) == _CANONICAL
    assert list(REGION_SLUGS.values()) == _CANONICAL
    print("agent canonical OK", PUBLISH_REGION_SLUGS)


def test_blend_keys_are_publish_slugs():
    geo = [s for s in PUBLISH_REGION_SLUGS if s != "global"]
    assert set(REGION_BLEND_WEIGHTS.keys()) == set(geo)
    print("blend weight keys OK")


def test_frontend_matches_agent():
    assert _GEO_TS.is_file(), f"missing frontend geographies: {_GEO_TS}"
    fe = _frontend_region_slugs(_GEO_TS)
    assert fe == list(PUBLISH_REGION_SLUGS), (
        f"slug mismatch:\n  agent={list(PUBLISH_REGION_SLUGS)}\n  frontend={fe}"
    )
    # Explicit anti-drift checks (historical footguns).
    assert "mea" not in fe
    assert "middle-east-africa" in fe
    assert "asia-pacific" in fe
    print("frontend==agent OK", fe)


if __name__ == "__main__":
    test_agent_matches_canonical()
    test_blend_keys_are_publish_slugs()
    test_frontend_matches_agent()
    print("C slug audit PASSED")
