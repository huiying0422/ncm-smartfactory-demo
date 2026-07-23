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
    # 1. Streamlit Cloud secrets
    try:
        import streamlit as st
        if "PGURL" in st.secrets:
            return st.secrets["PGURL"]
    except Exception:
        pass  # not running under streamlit, or no secrets file

    # 2. Environment variable / .env
    pgurl = os.environ.get("PGURL")
    if pgurl:
        return pgurl

    raise RuntimeError(
        "PGURL is not set. Provide it via st.secrets, an environment "
        "variable, or a .env file."
    )
