"""Unit tests for src.db.

Pure-unit tests mock psycopg2/pandas so they run without a database.
An optional live test runs against the real DB when PGURL is reachable.
"""

import os
from contextlib import contextmanager

import pandas as pd
import pytest

import src.db as db

EXPECTED_TABLES = {
    "batch_master",
    "batch_results",
    "tag_history",
    "alarm_history",
    "capa",
    "poms_route",
    "sap_material",
    "equipment_master",
}


def test_get_conn_commits_and_closes(monkeypatch):
    events = []

    class FakeConn:
        def commit(self):
            events.append("commit")

        def rollback(self):
            events.append("rollback")

        def close(self):
            events.append("close")

    monkeypatch.setattr(db, "get_pgurl", lambda: "postgresql://x")
    monkeypatch.setattr(db.psycopg2, "connect", lambda url: FakeConn())

    with db.get_conn() as conn:
        assert isinstance(conn, FakeConn)

    assert events == ["commit", "close"]


def test_get_conn_rolls_back_on_error(monkeypatch):
    events = []

    class FakeConn:
        def commit(self):
            events.append("commit")

        def rollback(self):
            events.append("rollback")

        def close(self):
            events.append("close")

    monkeypatch.setattr(db, "get_pgurl", lambda: "postgresql://x")
    monkeypatch.setattr(db.psycopg2, "connect", lambda url: FakeConn())

    with pytest.raises(ValueError):
        with db.get_conn():
            raise ValueError("boom")

    assert events == ["rollback", "close"]


def test_query_returns_dataframe(monkeypatch):
    expected = pd.DataFrame({"n": [42]})

    @contextmanager
    def fake_conn():
        yield object()

    monkeypatch.setattr(db, "get_conn", fake_conn)
    monkeypatch.setattr(
        db.pd, "read_sql_query", lambda sql, conn, params=None: expected
    )

    out = db.query("SELECT 1")
    pd.testing.assert_frame_equal(out, expected)


def test_list_tables(monkeypatch):
    monkeypatch.setattr(
        db,
        "query",
        lambda sql, params=None: pd.DataFrame(
            {"table_name": ["a", "b", "c"]}
        ),
    )
    assert db.list_tables() == ["a", "b", "c"]


# --------------------------------------------------------------- live test
def _db_available() -> bool:
    if not os.environ.get("PGURL"):
        return False
    try:
        db.query("SELECT 1")
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _db_available(), reason="database not reachable")
def test_live_expected_tables_present_and_populated():
    tables = set(db.list_tables())
    assert EXPECTED_TABLES.issubset(tables)
    for t in EXPECTED_TABLES:
        n = int(db.query(f'SELECT COUNT(*) AS n FROM "{t}"')["n"].iloc[0])
        assert n > 0, f"table {t} is empty"
