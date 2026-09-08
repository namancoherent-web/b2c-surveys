"""Verify container egress uses the host Proton VPN after a rotate.

Run from a worker container (or any host that should share VPN egress):

    docker compose exec worker python scripts/verify_vpn_egress.py

If the IP does NOT change after POST /rotate, VPN rotation has no effect on
container traffic — use the documented fallbacks (route Docker subnet through
Proton, or run the fetch layer on the host).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Running as scripts/verify_vpn_egress.py does not put /app on sys.path.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import requests

from src import vpn_client


def _public_ip(timeout: float = 10.0) -> str | None:
    try:
        r = requests.get("https://api.ipify.org", timeout=timeout)
        if r.ok:
            return r.text.strip()
    except Exception as exc:  # noqa: BLE001
        print(f"ipify failed: {exc}")
    return None


def main() -> int:
    print(f"VPN_ROTATE_ENABLED={os.getenv('VPN_ROTATE_ENABLED', 'true')}")
    print(f"VPN_ROTATE_URL={vpn_client.BASE}")
    alive = vpn_client.health()
    print(f"/health -> {'ok' if alive else 'DOWN (rotation will be skipped)'}")

    before = _public_ip()
    print(f"egress IP before rotate: {before}")
    if not before:
        print("FAIL: could not read public IP")
        return 1

    if not alive:
        print("SKIP rotate: tool not reachable — set VPN_ROTATE_ENABLED=false or start automation-tool")
        return 0

    print("POST /rotate (may wait up to ~2 min for Free-plan cooldown)…")
    ok = vpn_client.rotate("survey-agent/egress-check")
    print(f"rotate -> {'ok' if ok else 'failed'}")
    after = _public_ip()
    print(f"egress IP after rotate:  {after}")

    if not ok:
        print("FAIL: rotate call failed")
        return 1
    if after and after != before:
        print("PASS: container egress IP changed — VPN path is working")
        return 0
    print(
        "FAIL: IP unchanged — Docker egress is NOT through Proton.\n"
        "Fallbacks: (1) route the Docker subnet through Proton VPN, or\n"
        "           (2) run the fetch layer on the Windows host instead of in containers.\n"
        "Do not rely on rotation until this check passes."
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
