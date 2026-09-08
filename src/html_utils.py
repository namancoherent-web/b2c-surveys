"""HTML cleanup helpers for parsing fetched web pages."""

import html
import re

from bs4 import BeautifulSoup


def clean_html(raw: str) -> str:
    """Strip tags and normalize whitespace from raw HTML.

    Removes script/style content, unescapes HTML entities, and collapses runs
    of whitespace into single spaces. Returns "" for falsy input.
    """
    if not raw:
        return ""

    # Fast path: no tags to strip. Avoids BeautifulSoup's
    # MarkupResemblesLocatorWarning on short plain strings (titles, values).
    if "<" not in raw:
        return re.sub(r"\s+", " ", html.unescape(raw)).strip()

    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator=" ")
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
