"""Phase A6 smoke — select+blend global from synthetic regional files."""
import json
import os
import tempfile

# This smoke publishes its own two-region fixture, so a local RUN_REGIONS
# (used to narrow real runs) must not filter one of them out of the index.
os.environ["RUN_REGIONS"] = ""

from src.nodes.global_selector import (  # noqa: E402
    publish_global_selection,
    select_global_questionnaire,
)
from src.question_plan import (  # noqa: E402
    QUESTIONS_PER_TAB_TARGET,
    REGION_BLEND_WEIGHTS,
)


def _q(text, options, data, layer="core", grounded=False):
    out = {
        "id": "x",
        "qNum": 1,
        "question": text,
        "options": options,
        "chart": "vbar",
        "data": [{"label": o, "value": v} for o, v in zip(options, data)],
        "insight": "",
        "questionLayer": layer,
    }
    if grounded:
        out["isGrounded"] = True
    return out


def _payload(region, questions, section="profile"):
    return {
        "segment": "Air Fryer",
        "slug": "air-fryer",
        "region": region,
        "frontend": {
            "sections": [{
                "id": section,
                "label": "Consumer Profile",
                "questions": questions,
            }],
        },
        "metadata": {},
    }


def test_blend_and_select():
    # Shared core across 2 regions; unique modules.
    core_text = "How often do you use an air fryer?"
    opts = ["Daily", "Weekly", "Rarely"]
    eu = _payload("europe", [
        _q(core_text, opts, [20, 50, 30], "core", grounded=True),
        _q("Which EU energy label matters most?", ["A", "B", "C"], [40, 40, 20], "module"),
    ])
    na = _payload("north-america", [
        _q(core_text, opts, [45, 30, 25], "core", grounded=True),
        _q("Do you prefer air fryer brands sold at big-box stores?",
           ["Yes", "No", "Not sure"], [50, 30, 20], "module"),
    ])

    selected = select_global_questionnaire(
        {"europe": eu, "north-america": na},
        segment="Air Fryer",
    )
    secs = selected["frontend"]["sections"]
    assert secs, "expected at least one section"
    qs = secs[0]["questions"]
    texts = [q["question"] for q in qs]
    assert core_text in texts
    # No invented questions
    for t in texts:
        assert t in (
            core_text,
            "Which EU energy label matters most?",
            "Do you prefer air fryer brands sold at big-box stores?",
        )
    core = next(q for q in qs if q["question"] == core_text)
    assert set(core["sourceRegions"]) == {"europe", "north-america"}
    # Blended Daily is between 20 and 45 (weighted), not a fresh simulation.
    daily = next(d["value"] for d in core["data"] if d["label"] == "Daily")
    assert 20 < daily < 45, daily
    assert abs(sum(d["value"] for d in core["data"]) - 100) < 0.2
    assert selected["selection"]["method"] == "select_and_blend"
    print("select+blend OK", {"n": len(qs), "daily": daily})


def test_publish_global():
    with tempfile.TemporaryDirectory() as tmp:
        surveys = os.path.join(tmp, "surveys")
        slug = "air-fryer"
        folder = os.path.join(surveys, slug)
        os.makedirs(folder)
        for region, daily in (("europe", 20), ("north-america", 45)):
            p = _payload(region, [
                _q("How often do you use an air fryer?",
                   ["Daily", "Weekly", "Rarely"],
                   [daily, 50 if region == "europe" else 30, 30 if region == "europe" else 25],
                   "core"),
            ])
            with open(os.path.join(folder, f"{region}.json"), "w", encoding="utf-8") as fh:
                json.dump(p, fh)

        info = publish_global_selection(
            "Air Fryer", slug=slug, surveys_dir=surveys,
        )
        assert os.path.isfile(info["path"])
        assert os.path.isfile(info["index"])
        global_p = json.loads(open(info["path"], encoding="utf-8").read())
        assert global_p["region"] == "global"
        assert global_p["frontend"]["sections"][0]["questions"]
        index = json.loads(open(info["index"], encoding="utf-8").read())
        assert "global" in index["regions"]
        assert "europe" in index["regions"]
        print("publish global OK", info["questions"])


def test_weights_present():
    assert abs(sum(REGION_BLEND_WEIGHTS.values()) - 1.0) < 0.01
    # The per-tab target is configurable (QUESTIONS_PER_TAB_TARGET) and has since
    # been tuned down to 8; what this guards is that a tab is never so thin that
    # the storyline has no room to develop.
    assert QUESTIONS_PER_TAB_TARGET >= 4
    print("weights OK")


if __name__ == "__main__":
    test_weights_present()
    test_blend_and_select()
    test_publish_global()
    print("A6 smoke PASSED")
