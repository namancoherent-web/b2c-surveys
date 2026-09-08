"""One-time Reddit cookie setup for Windows.

On Windows, ``rdt login`` often fails because reading Edge/Chrome cookies
requires Administrator privileges (or Chrome's encryption blocks extraction).

This script lets you paste the ``reddit_session`` cookie manually from Edge
DevTools — no admin needed.

Steps in Edge:
  1. Log into https://www.reddit.com
  2. Press F12 → Application → Cookies → https://www.reddit.com
  3. Click the ``reddit_session`` row and copy its Value
  4. Run this script and paste when prompted

Usage:
    .venv\\Scripts\\python.exe scripts\\setup_reddit_cookies.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path("secrets/reddit-cookies.json")


def main() -> None:
    print("Reddit manual cookie setup (Windows-friendly)\n")
    print("In Edge DevTools (F12):")
    print("  Application → Cookies → https://www.reddit.com → reddit_session\n")

    session = input("Paste reddit_session cookie value (or press Enter to cancel): ").strip()
    if not session:
        print("Cancelled.")
        sys.exit(0)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "cookies": {"reddit_session": session},
        "source": "manual-edge-export",
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved to {OUT.resolve()}")
    print("\nAdd to your .env:")
    print("REDDIT_CLI_COOKIE_PATH=./secrets/reddit-cookies.json")
    print("REDDIT_ENABLED=true")
    print("\nThen verify:")
    print("  .venv\\Scripts\\python.exe -m src.diagnose_sources")


if __name__ == "__main__":
    main()
