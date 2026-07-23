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
