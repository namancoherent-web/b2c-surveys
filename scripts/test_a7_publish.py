"""Phase A7 smoke — per-region publish folder + index + legacy."""
import json
import os
import tempfile

from src.question_plan import PUBLISH_REGION_SLUGS
from src import survey_io


def _mini_survey():
    return {
        "frontend": {
            "sections": [{
                "id": "profile",
                "label": "Consumer Profile",
                "questions": [{
                    "id": "q1",
                    "qNum": 1,
                    "question": "How often do you use an air fryer?",
                    "options": ["Daily", "Weekly", "Rarely"],
                    "chart": "vbar",
                    "data": [
                        {"label": "Daily", "value": 40},
                        {"label": "Weekly", "value": 35},
                        {"label": "Rarely", "value": 25},
                    ],
                    "insight": "Top response: Daily (40%).",
                    "dataByRegion": {
                        "global": [
                            {"label": "Daily", "value": 40},
                            {"label": "Weekly", "value": 35},
                            {"label": "Rarely", "value": 25},
                        ],
                        "europe": [
                            {"label": "Daily", "value": 20},
                            {"label": "Weekly", "value": 50},
                            {"label": "Rarely", "value": 30},
                        ],
                        "north-america": [
                            {"label": "Daily", "value": 45},
                            {"label": "Weekly", "value": 30},
                            {"label": "Rarely", "value": 25},
                        ],
                    },
                }],
            }],
        },
        "survey": {
            "market_segment": "Air Fryer",
            "metadata": {"total_questions": 1, "warnings": []},
        },
    }


def test_publish_bundle():
    survey = _mini_survey()
    with tempfile.TemporaryDirectory() as tmp:
        surveys_dir = os.path.join(tmp, "public", "surveys")
        os.makedirs(surveys_dir)
        # Parent of FRONTEND_SURVEYS_DIR must exist (survey_io checks this).
        survey_io.FRONTEND_SURVEYS_DIR = surveys_dir

        slug = "air-fryer"
        info = survey_io.publish_survey_bundle(
            slug, survey, segment="Air Fryer",
            json_text=json.dumps(survey),
        )
        assert info["legacy"] and os.path.isfile(info["legacy"])
        assert info["index"] and os.path.isfile(info["index"])
        assert set(info["regions"]) == set(PUBLISH_REGION_SLUGS)

        index = json.loads(open(info["index"], encoding="utf-8").read())
        assert index["slug"] == slug
        assert index["segment"] == "Air Fryer"
        assert index["regions"] == PUBLISH_REGION_SLUGS
        assert "generatedAt" in index

        europe = json.loads(open(info["regions"]["europe"], encoding="utf-8").read())
        assert europe["region"] == "europe"
        assert "dataByRegion" not in europe["frontend"]["sections"][0]["questions"][0]
        assert europe["frontend"]["sections"][0]["questions"][0]["data"][0]["value"] == 20

        global_p = json.loads(open(info["regions"]["global"], encoding="utf-8").read())
        assert global_p["frontend"]["sections"][0]["questions"][0]["data"][0]["value"] == 40

        # Legacy still has full payload with dataByRegion
        legacy = json.loads(open(info["legacy"], encoding="utf-8").read())
        assert "dataByRegion" in legacy["frontend"]["sections"][0]["questions"][0]

    print("A7 publish bundle OK")


def test_slug_list_matches_frontend():
    assert PUBLISH_REGION_SLUGS == [
        "global",
        "north-america",
        "europe",
        "asia-pacific",
        "latin-america",
        "middle-east-africa",
    ]
    print("A7 slug list OK")


if __name__ == "__main__":
    test_slug_list_matches_frontend()
    test_publish_bundle()
    print("A7 smoke PASSED")
