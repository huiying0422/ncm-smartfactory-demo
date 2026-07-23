"""Configuration loader.

Loads environment variables from a local .env file (via python-dotenv)
and exposes typed accessors for the rest of the application.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (one level up from src/) if present.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_ENV_PATH = _PROJECT_ROOT / ".env"

# load_dotenv is a no-op if the file does not exist; existing env vars win
# unless override=False (default), which is what we want.
load_dotenv(dotenv_path=_ENV_PATH)


def get_pgurl() -> str:
    """Return the Postgres connection URL from the PGURL env var.

    Raises:
        RuntimeError: if PGURL is not set.
    """
    pgurl = os.environ.get("PGURL")
    if not pgurl:
        raise RuntimeError(
            "PGURL is not set. Add it to your .env file, e.g.\n"
            "  PGURL=postgresql://user:pass@host/db?sslmode=require"
        )
    return pgurl
