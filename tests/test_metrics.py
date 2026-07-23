"""Tests for src.metrics.

Pure-logic tests run without a DB; a live section asserts the three-tier
early-warning story (SPC 16 / conventional 21 / failure 35) against the DB.
"""

import os

import pandas as pd
import pytest

import src.db as db
from src.metrics import (
    conventional_alarm_batch,
    early_warning_batches,
    quality_failure_batch,
    ridge_coefficients,
    spc_alarm_batch,
)


# ---------------------------------------------------------------- pure logic
def test_spc_alarm_needs_two_consecutive():
    df = pd.DataFrame({
        "batch_seq": [1, 2, 3, 4, 5, 6],
        # single spike at 3, sustained run starts at 5
        "health_alarm": [False, False, True, False, True, True],
    })
    assert spc_alarm_batch(df) == 5


def test_spc_alarm_none_when_no_run():
    df = pd.DataFrame({
        "batch_seq": [1, 2, 3],
        "health_alarm": [True, False, True],
    })
    assert spc_alarm_batch(df) is None


def test_conventional_alarm_filters_source_and_priority():
    df = pd.DataFrame({
        "source": ["ML-401/Vib", "FD-301/DP/Cake", "FD-301/AGT/Torque"],
        "priority": [2, 2, 3],
        "batch_seq": [5, 21, 10],
    })
    # ML row excluded (wrong source); torque row excluded (priority 3)
    assert conventional_alarm_batch(df) == 21


def test_quality_failure_batch():
    df = pd.DataFrame({
        "batch_seq": [33, 34, 35, 36],
        "actual_residual_ppm": [400.0, 480.0, 528.9, 503.5],
    })
    assert quality_failure_batch(df) == 35


def test_early_warning():
    assert early_warning_batches(16, 35) == 19
    assert early_warning_batches(None, 35) is None


# ---------------------------------------------------------------- live DB
def _db_ready() -> bool:
    if not os.environ.get("PGURL"):
        return False
    try:
        return {"batch_predictions", "batch_features"}.issubset(
            set(db.list_tables())
        )
    except Exception:
        return False


live = pytest.mark.skipif(not _db_ready(), reason="db/tables not available")


@pytest.fixture(scope="module")
def pred():
    return db.query("SELECT * FROM batch_predictions ORDER BY batch_seq")


@pytest.fixture(scope="module")
def alarms_seq():
    return db.query(
        "SELECT a.source, a.priority, bp.batch_seq "
        "FROM alarm_history a "
        "JOIN batch_predictions bp ON bp.batch_id = a.batch_id"
    )


@live
def test_live_spc_alarm_batch_16(pred):
    assert spc_alarm_batch(pred) == 16


@live
def test_live_conventional_alarm_batch_21(alarms_seq):
    assert conventional_alarm_batch(alarms_seq) == 21


@live
def test_live_quality_failure_batch_35(pred):
    assert quality_failure_batch(pred) == 35


@live
def test_live_early_warning_19(pred, alarms_seq):
    spc = spc_alarm_batch(pred)
    fail = quality_failure_batch(pred)
    assert early_warning_batches(spc, fail) == 19


@live
def test_live_ridge_coefficients_shape():
    feats = db.query("SELECT * FROM batch_features ORDER BY batch_seq")
    coefs = ridge_coefficients(feats)
    assert len(coefs) == 5
    # sorted by absolute value descending
    assert list(coefs.index) == list(coefs.abs().sort_values(ascending=False).index)
