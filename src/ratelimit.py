"""Redis-backed shared token-bucket rate limiter.

Race-free across all RQ worker containers: refill + take is one atomic Lua call.
Fails open (local sleep) if Redis is down so CLI / single-process runs still work.
"""

from __future__ import annotations

import os
import time

import redis

_R = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))

# KEYS[1]=bucket key; ARGV: rate, burst, now(ms), requested(1)
_LUA = """
local key=KEYS[1]
local rate=tonumber(ARGV[1]); local burst=tonumber(ARGV[2])
local now=tonumber(ARGV[3]); local want=tonumber(ARGV[4])
local st=redis.call('HMGET', key, 'tokens', 'ts')
local tokens=tonumber(st[1]); local ts=tonumber(st[2])
if tokens==nil then tokens=burst; ts=now end
local delta=math.max(0, now-ts)/1000.0
tokens=math.min(burst, tokens + delta*rate)
local allowed=0
if tokens>=want then tokens=tokens-want; allowed=1 end
redis.call('HMSET', key, 'tokens', tokens, 'ts', now)
redis.call('PEXPIRE', key, 60000)
return allowed
"""
_SHA = None


def _script():
    global _SHA
    if _SHA is None:
        _SHA = _R.script_load(_LUA)
    return _SHA


def acquire(bucket: str, rate_per_sec: float, burst: float, timeout: float = 120.0) -> None:
    """Block until a token is available. FAIL OPEN if Redis is down (local sleep)."""
    deadline = time.time() + timeout
    while True:
        try:
            ok = _R.evalsha(
                _script(),
                1,
                f"rl:{bucket}",
                rate_per_sec,
                burst,
                int(time.time() * 1000),
                1,
            )
            if ok == 1:
                return
        except Exception:
            # Fail open: don't deadlock jobs when Redis is unavailable (e.g. CLI).
            time.sleep(1.0 / max(rate_per_sec, 0.1))
            return
        if time.time() > deadline:
            return  # don't deadlock a job
        time.sleep(0.05)
