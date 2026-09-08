"""Environment configuration.

Loads variables from a local ``.env`` (via python-dotenv) and exposes them as
module-level constants. Import-time validation fails fast with a clear message
if a required secret is missing, so misconfiguration surfaces at startup rather
than deep inside a node.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the survey_agent project root (parent of src/), not CWD.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


def _require(name: str) -> str:
    """Return the env var ``name`` or raise a clear startup error."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            f"Copy .env.example to .env and fill it in."
        )
    return value


NEON_CONNECTION_STRING: str = _require("NEON_CONNECTION_STRING")

# --- LLM provider -----------------------------------------------------------
# "deepseek" (recommended) | "openrouter" | "anthropic"
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "deepseek").lower()

# DeepSeek native API — https://platform.deepseek.com/api_keys
DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

# OpenRouter (optional). Get a key at https://openrouter.ai/keys.
OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL: str = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
# Used when LLM_PROVIDER=openrouter
LLM_MODEL: str = os.getenv("LLM_MODEL", "deepseek/deepseek-chat")

# Anthropic (optional alternative provider).
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL: str = os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

# Cap on output tokens (DeepSeek V3 max output ~8k); keeps requests within limits.
MAX_OUTPUT_TOKENS: int = int(os.getenv("MAX_OUTPUT_TOKENS", "8000"))

# Exa semantic search (SOURCE B primary). Optional — degrades to DDG if unset.
# Free tier: https://exa.ai
EXA_API_KEY: str = os.getenv("EXA_API_KEY", "")

# Reddit consumer-voice search (optional). Prefer OAuth; cookie/rdt is fallback.
REDDIT_CLI_COOKIE_PATH: str = os.getenv("REDDIT_CLI_COOKIE_PATH", "")
REDDIT_ENABLED: bool = os.getenv("REDDIT_ENABLED", "true").lower() in ("1", "true", "yes")
# change for b2c questionarie — official Reddit OAuth (script/web app)
REDDIT_CLIENT_ID: str = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET: str = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT: str = os.getenv(
    "REDDIT_USER_AGENT",
    "survey-agent/1.0 (B2C consumer-voice; contact: local)",
)

# change for b2c questionarie — YouTube Data API (comments = consumer language)
YOUTUBE_API_KEY: str = os.getenv("YOUTUBE_API_KEY", "")
YOUTUBE_ENABLED: bool = os.getenv("YOUTUBE_ENABLED", "true").lower() in ("1", "true", "yes")

# change for b2c questionarie — Twitter/X (costly/gated; OFF by default)
TWITTER_ENABLED: bool = os.getenv("TWITTER_ENABLED", "false").lower() in ("1", "true", "yes")
TWITTER_BEARER_TOKEN: str = os.getenv("TWITTER_BEARER_TOKEN", "")

# change for b2c questionarie — web fetch safety (avoid DDG/Jina blocks without VPN)
DDG_ENABLED: bool = os.getenv("DDG_ENABLED", "true").lower() in ("1", "true", "yes")
JINA_ENABLED: bool = os.getenv("JINA_ENABLED", "true").lower() in ("1", "true", "yes")

# Validate the active provider's credentials at startup (fail fast, clear error).
if LLM_PROVIDER == "anthropic":
    if not ANTHROPIC_API_KEY:
        raise RuntimeError(
            "LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is missing. Set it in .env."
        )
elif LLM_PROVIDER == "deepseek":
    if not DEEPSEEK_API_KEY:
        raise RuntimeError(
            "LLM_PROVIDER=deepseek but DEEPSEEK_API_KEY is missing. "
            "Get a key at https://platform.deepseek.com/api_keys and set DEEPSEEK_API_KEY in .env."
        )
elif LLM_PROVIDER == "openrouter":
    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "LLM_PROVIDER=openrouter but OPENROUTER_API_KEY is missing. "
            "Get a key at https://openrouter.ai/keys and set OPENROUTER_API_KEY in .env."
        )
else:
    raise RuntimeError(
        f"Unknown LLM_PROVIDER={LLM_PROVIDER!r}. Use deepseek | openrouter | anthropic."
    )
