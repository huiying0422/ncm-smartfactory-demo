"""Contract tests for the Databricks notebook's output tables.

These verify the state of the database AFTER the notebook has run:
``batch_predictions`` and ``equipment_health`` each have 40 rows and no
null ``health_score``. They auto-skip if the DB (or the tables) are absent.
"""

import os

import pytest

import src.db as db


def _db_available() -> bool:
    if not os.environ.get("PGURL"):
        return False
    try:
        db.query("SELECT 1")
        return True
    except Exception:
        return False


def _table_exists(name: str) -> bool:
    return name in set(db.list_tables())


pytestmark = pytest.mark.skipif(
    not _db_available(), reason="database not reachable"
)


@pytest.mark.skipif(
    not (_db_available() and _table_exists("batch_predictions")),
    reason="batch_predictions not created yet (run the notebook first)",
)
def test_batch_predictions_contract():
    n = int(db.query("SELECT COUNT(*) AS n FROM batch_predictions")["n"].iloc[0])
    assert n == 40
    nulls = int(
        db.query(
            "SELECT COUNT(*) AS n FROM batch_predictions "
            "WHERE health_score IS NULL"
        )["n"].iloc[0]
    )
    assert nulls == 0


@pytest.mark.skipif(
    not (_db_available() and _table_exists("equipment_health")),
    reason="equipment_health not created yet (run the notebook first)",
)
def test_equipment_health_contract():
    n = int(db.query("SELECT COUNT(*) AS n FROM equipment_health")["n"].iloc[0])
    assert n == 40
    nulls = int(
        db.query(
            "SELECT COUNT(*) AS n FROM equipment_health "
            "WHERE health_score IS NULL"
        )["n"].iloc[0]
    )
    assert nulls == 0
