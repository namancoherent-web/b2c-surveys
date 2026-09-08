"""Date helpers shared across nodes (e.g. for grounding queries in 'now')."""

from datetime import datetime, timezone


def current_date_context() -> dict:
    """Return the current UTC date in a few presentation-ready forms.

    Keys:
        iso:        "YYYY-MM-DD"
        year:       int
        month_year: "Month YYYY" (e.g. "June 2026")
    """
    now = datetime.now(timezone.utc)
    return {
        "iso": now.strftime("%Y-%m-%d"),
        "year": now.year,
        "month_year": now.strftime("%B %Y"),
    }
