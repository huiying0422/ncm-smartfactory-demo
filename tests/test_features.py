"""Tests for src.features (require a reachable, populated database)."""

import os

import pytest

import src.db as db
from src.features import build_batch_features, write_batch_features


def _db_available() -> bool:
    if not os.environ.get("PGURL"):
        return False
    try:
        db.query("SELECT 1")
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _db_available(), reason="database not reachable"
)


@pytest.fixture(scope="module")
def features():
    return build_batch_features()


def test_exactly_40_rows_no_duplicate_batch_ids(features):
    assert len(features) == 40
    assert features["batch_id"].is_unique


def test_vacuum_trends_with_batch_seq(features):
    rho = features["fd_vacuum_ultimate"].corr(
        features["batch_seq"], method="spearman"
    )
    assert rho > 0.7, f"fd_vacuum_ultimate vs batch_seq Spearman={rho:.3f}"


def test_filt_time_trends_with_batch_seq(features):
    rho = features["fd_filt_time_min"].corr(
        features["batch_seq"], method="spearman"
    )
    assert rho > 0.7, f"fd_filt_time_min vs batch_seq Spearman={rho:.3f}"


def test_no_alarms_before_batch_seq_15(features):
    early = features[features["batch_seq"] <= 15]
    assert (early["alarm_count"] == 0).all()


def test_no_column_entirely_nan(features):
    all_nan = [c for c in features.columns if features[c].isna().all()]
    assert not all_nan, f"columns entirely NaN: {all_nan}"


def test_write_batch_features_roundtrip(features):
    write_batch_features(features)
    n = int(db.query("SELECT COUNT(*) AS n FROM batch_features")["n"].iloc[0])
    assert n == 40
