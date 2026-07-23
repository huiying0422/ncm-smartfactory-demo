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
    """Return the Postgres connection URL.

    Resolution order:
        1. st.secrets["PGURL"]  (Streamlit / Streamlit Cloud)
        2. os.environ["PGURL"]  (environment variable)
        3. PGURL from .env      (loaded via python-dotenv at import time)

    Raises:
        RuntimeError: if PGURL is not found in any of the three sources.
    """
    # 1. Streamlit secrets. Wrapped so scripts/tests without streamlit installed
    #    (or with no secrets file) still work.
    try:
        import streamlit as st

        if "PGURL" in st.secrets:
            pgurl = st.secrets["PGURL"]
            if pgurl:
                return pgurl
    except Exception:
        pass

    # 2. Environment variable, and 3. .env (loaded into os.environ at import).
    pgurl = os.environ.get("PGURL")
    if not pgurl:
        raise RuntimeError(
            "PGURL is not set. Provide it via one of:\n"
            "  1. .streamlit/secrets.toml  ->  PGURL = \"postgresql://...\"\n"
            "  2. environment variable     ->  export PGURL=postgresql://...\n"
            "  3. .env file                ->  PGURL=postgresql://user:pass@host/db"
        )
    return pgurl
