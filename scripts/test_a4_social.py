"""Phase A4 smoke — YouTube/Twitter/Reddit voice facts never carry numerics."""
from src import config
from src.evidence_utils import VOICE_ONLY_ORIGINS, build_voice_fact
from src.searchers.twitter_search import search_twitter, twitter_to_findings
from src.searchers.youtube_search import youtube_to_findings


def test_voice_fact_strips_numbers():
    f = build_voice_fact(
        fact="41% of people hate the smell and return rates are 25%",
        origin="youtube",
        source_ref="https://www.youtube.com/watch?v=abc",
        title="demo — comment",
        topic="complaints",
    )
    assert f is not None
    assert f["origin"] == "youtube"
    assert f["is_numeric"] is False
    assert f["value"] is None
    assert f["authority"] == "low"
    print("voice fact numeric strip OK")


def test_youtube_to_findings():
    rows = [{
        "text": "The basket is impossible to clean and the plastic smells weird after a week.",
        "video_title": "Air fryer review",
        "url": "https://www.youtube.com/watch?v=abc123",
        "published_at": "2026-01-01",
    }]
    findings = youtube_to_findings(rows, "air fryer complaints", "complaints")
    assert len(findings) == 1
    assert findings[0]["origin"] == "youtube"
    assert findings[0]["is_numeric"] is False
    assert findings[0]["value"] is None
    print("youtube_to_findings OK")


def test_twitter_default_off():
    assert config.TWITTER_ENABLED is False or isinstance(config.TWITTER_ENABLED, bool)
    # Default in .env.example is false; if user enabled locally, still must not crash
    hits = search_twitter("air fryer", limit=5)
    assert isinstance(hits, list)
    if not config.TWITTER_ENABLED:
        assert hits == []
        assert twitter_to_findings([], "x", "reviews") == []
    print("twitter default-off OK", {"enabled": config.TWITTER_ENABLED, "hits": len(hits)})


def test_voice_origins_set():
    assert VOICE_ONLY_ORIGINS == frozenset({"reddit", "youtube", "twitter"})
    print("VOICE_ONLY_ORIGINS OK")


if __name__ == "__main__":
    test_voice_fact_strips_numbers()
    test_youtube_to_findings()
    test_twitter_default_off()
    test_voice_origins_set()
    print("A4 smoke PASSED")
