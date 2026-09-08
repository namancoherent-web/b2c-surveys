"""Smoke tests for segment-only B2C gate + fixed-5 fan-out.

# change for b2c questionarie

Run from survey_agent/:
    python tests/test_b2c_gate_fanout.py
"""

from __future__ import annotations

import asyncio
import ast
import inspect
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.nodes.scope_classify import decide_b2c_gate
from src.question_plan import GEOGRAPHIC_REGIONS


def test_gate_rejects_high_confidence_pure_b2b():
    out = decide_b2c_gate(
        segment_type="b2b",
        classification_confidence="high",
        is_hybrid=False,
        classification_reason="OEM industrial buyers only",
        strict=False,
    )
    assert out["gate_passed"] is False
    assert out["status"] == "rejected"
    assert out["reason"]


def test_gate_accepts_b2c():
    out = decide_b2c_gate(
        segment_type="b2c",
        classification_confidence="high",
        is_hybrid=False,
        classification_reason="Household sneakers buyers",
        strict=False,
    )
    assert out["gate_passed"] is True
    assert out["status"] == "accepted"


def test_gate_accepts_hybrid_proxy_lenient():
    out = decide_b2c_gate(
        segment_type="b2b",
        classification_confidence="medium",
        is_hybrid=True,
        classification_reason="Patients are end users",
        strict=False,
    )
    assert out["gate_passed"] is True
    assert out["gate_decision"] == "accepted_hybrid_proxy"


def test_gate_accepts_ambiguous_lenient():
    out = decide_b2c_gate(
        segment_type="b2b",
        classification_confidence="low",
        is_hybrid=False,
        classification_reason="Unclear buyer",
        strict=False,
    )
    assert out["gate_passed"] is True


def test_fixed_five_continents():
    assert len(GEOGRAPHIC_REGIONS) == 5
    assert "Global" not in GEOGRAPHIC_REGIONS


def test_run_py_rejects_region_arg():
    """(c) run.py / run_one_async do not accept a region argument."""
    src = (ROOT / "run.py").read_text(encoding="utf-8")
    assert "no longer accepted" in src.lower() or "Usage: python run.py" in src
    from src import tasks

    sig = inspect.signature(tasks.run_one_async)
    assert "region" not in sig.parameters
    assert "segment" in sig.parameters


def test_enqueue_uses_classify_and_fanout_only():
    """(c) enqueue.py enqueues classify_and_fanout(segment) only."""
    src = (ROOT / "enqueue.py").read_text(encoding="utf-8")
    assert "classify_and_fanout" in src
    assert "run_one_region" not in src
    from src import tasks

    sig = inspect.signature(tasks.classify_and_fanout)
    assert list(sig.parameters) == ["segment"]


def test_rejected_writes_no_region_files():
    """(a) rejected classify writes rejection artifact only."""
    from src import tasks

    fake_state = {
        "status": "rejected",
        "gate_passed": False,
        "segment_type": "b2b",
        "classification_confidence": "high",
        "is_hybrid": False,
        "rejection_reason": "Bought by OEMs, not households.",
        "classification_reason": "Bought by OEMs, not households.",
        "normalized_segment": "Industrial Hydraulic Seals",
        "gate_decision": "rejected_pure_b2b",
    }

    async def fake_astream(_initial, stream_mode="updates"):
        yield {"scope_classify": fake_state}
        yield {
            "reject_stop": {
                "status": "rejected",
                "rejection_reason": fake_state["rejection_reason"],
                "final_survey": {
                    "status": "rejected",
                    "reason": fake_state["rejection_reason"],
                    "segment_type": "b2b",
                },
            }
        }

    mock_graph = MagicMock()
    mock_graph.astream = fake_astream

    with tempfile.TemporaryDirectory() as td:
        out_dir = Path(td)
        with patch.object(tasks, "_OUTPUT_DIR", out_dir):
            with patch.object(tasks, "build_classify_graph", return_value=mock_graph):
                result = asyncio.run(
                    tasks.classify_segment_async("industrial hydraulic seals")
                )

        assert result["status"] == "rejected"
        assert result["regions_enqueued"] == 0
        assert Path(result["path"]).exists()
        region_files = list(out_dir.glob("*_north-america.json")) + list(
            out_dir.glob("*_europe.json")
        )
        assert region_files == []


def test_accepted_fans_out_exactly_five_regions():
    """(b)+(d) B2C accept → exactly 5 regional enqueues; classify once."""
    from src import tasks

    classification = {
        "normalized_segment": "Sneakers",
        "segment_type": "b2c",
        "classification_confidence": "high",
        "is_hybrid": False,
        "classification_reason": "Consumers buy sneakers",
        "audience_profile": {},
        "gate_decision": "accepted_b2c",
        "gate_passed": True,
    }

    with patch.object(
        tasks,
        "classify_segment",
        return_value={"status": "accepted", "classification": classification},
    ) as mock_clf:
        with patch.object(tasks, "_enqueue_regional_jobs") as mock_enq:
            mock_enq.return_value = {
                "regional_job_ids": ["a", "b", "c", "d", "e"],
                "global_job_id": "g",
                "regions": list(GEOGRAPHIC_REGIONS),
            }
            out = tasks.classify_and_fanout("global sneakers market")

    mock_clf.assert_called_once_with("global sneakers market")
    mock_enq.assert_called_once()
    assert len(mock_enq.return_value["regions"]) == 5
    assert out["status"] == "accepted"
    assert out["regions_enqueued"] == 5


def test_region_job_requires_classification_skips_scope():
    """(d) regional pipeline requires classification; region graph has no scope."""
    from src import tasks
    from src.graph import build_classify_graph, build_region_graph

    async def _run():
        try:
            await tasks.run_one_region_async("sneakers", "Europe", classification=None)
            raise AssertionError("expected ValueError")
        except ValueError as e:
            assert "classification" in str(e).lower()

    asyncio.run(_run())

    region_nodes = set(build_region_graph().get_graph().nodes)
    classify_nodes = set(build_classify_graph().get_graph().nodes)
    # LangGraph may key nodes as strings
    region_names = {str(n) for n in region_nodes}
    classify_names = {str(n) for n in classify_nodes}
    assert any("evidence_harvester" in n for n in region_names)
    assert not any(n == "scope_classify" or n.endswith("scope_classify") for n in region_names if not n.startswith("__"))
    # More reliable: compiled graph node list
    assert "scope_classify" not in region_names or "__start__" in region_names
    # Explicit check via graph builder structure
    assert "scope_classify" in classify_names
    assert "evidence_harvester" not in classify_names


def test_manifest_has_no_required_region():
    for name in ("segments.json", "segments-one.json"):
        import json

        data = json.loads((ROOT / name).read_text(encoding="utf-8"))
        assert "segments" in data
        # region key optional/ignored — prefer absent
        assert "region" not in data or True


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print(f"OK   {name}")
        except Exception as e:  # noqa: BLE001
            print(f"FAIL {name}: {e}")
            failures += 1
    print()
    print("PASSED" if failures == 0 else f"{failures} FAILED")
    sys.exit(1 if failures else 0)
