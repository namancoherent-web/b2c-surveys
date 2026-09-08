"""Phase A4b/A5 smoke — region helpers, core/module split, publish one region."""
from src.question_plan import (
    CORE_PER_TAB,
    MODULE_PER_TAB,
    QUESTIONS_PER_TAB_TARGET,
    REGION_QUESTION_MODE,
    match_geographic_region,
    region_to_slug,
)
from src.nodes.scope_classify import scope_classify
from src.nodes.question_architect import _Q_SCHEMA_V


def test_mode_and_split():
    assert REGION_QUESTION_MODE == "core_plus_module"
    assert CORE_PER_TAB + MODULE_PER_TAB == QUESTIONS_PER_TAB_TARGET
    assert CORE_PER_TAB >= 6
    assert MODULE_PER_TAB >= 2
    assert match_geographic_region("Europe") == "Europe"
    assert match_geographic_region("europe") == "Europe"
    assert match_geographic_region("asia-pacific") == "Asia Pacific"
    assert match_geographic_region("Global") is None
    assert region_to_slug("Europe") == "europe"
    assert _Q_SCHEMA_V == "questionnaire_v6"
    print("mode/split OK", {"core": CORE_PER_TAB, "module": MODULE_PER_TAB})


def test_scope_pins_region_job():
    # change for b2c questionarie — region pin helpers still used by region jobs
    assert match_geographic_region("North America") == "North America"
    print("scope pin helpers OK")


if __name__ == "__main__":
    test_mode_and_split()
    test_scope_pins_region_job()
    from src.graph import build_classify_graph, build_graph, build_region_graph

    assert build_classify_graph() is not None
    assert build_region_graph() is not None
    assert build_graph() is not None
    print("graph builds OK (classify + region + full)")
    print("A5 smoke PASSED")
