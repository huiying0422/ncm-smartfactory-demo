"""Unit tests for src.config (no database required)."""

import importlib

import pytest


def test_get_pgurl_returns_value(monkeypatch):
    monkeypatch.setenv("PGURL", "postgresql://u:p@host/db")
    import src.config as config
    importlib.reload(config)
    assert config.get_pgurl() == "postgresql://u:p@host/db"


def test_get_pgurl_raises_when_unset(monkeypatch):
    monkeypatch.delenv("PGURL", raising=False)
    import src.config as config
    importlib.reload(config)
    # Ensure a stray .env doesn't repopulate it after reload.
    monkeypatch.delenv("PGURL", raising=False)
    with pytest.raises(RuntimeError):
        config.get_pgurl()


def test_get_pgurl_prefers_streamlit_secrets(monkeypatch):
    # st.secrets should win over os.environ.
    import streamlit as st
    monkeypatch.setattr(
        st, "secrets", {"PGURL": "postgresql://from-secrets"}, raising=False
    )
    monkeypatch.setenv("PGURL", "postgresql://from-env")
    import src.config as config
    importlib.reload(config)
    assert config.get_pgurl() == "postgresql://from-secrets"


def test_get_pgurl_falls_back_to_env_when_secrets_missing(monkeypatch):
    # If streamlit secrets access raises (no secrets.toml), fall back to env.
    import streamlit as st

    class _Boom:
        def __contains__(self, key):
            raise RuntimeError("no secrets file")

    monkeypatch.setattr(st, "secrets", _Boom(), raising=False)
    monkeypatch.setenv("PGURL", "postgresql://from-env")
    import src.config as config
    importlib.reload(config)
    assert config.get_pgurl() == "postgresql://from-env"
