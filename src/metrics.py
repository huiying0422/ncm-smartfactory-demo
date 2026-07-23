"""Pure, testable metric helpers for the dashboard.

These functions take DataFrames and return plain numbers/objects so they can be
unit-tested without Streamlit. The Streamlit app (`app.py`) wires them to the DB.
"""

from typing import Optional

import numpy as np
import pandas as pd

# FD-301 equipment sensor features (same five used by the health model).
FD_SENSOR_FEATURES = [
    "fd_vacuum_ultimate",
    "fd_pulldown_min",
    "fd_filt_time_min",
    "fd_cake_dp_max",
    "fd_blade_torque_mean",
]

QUALITY_SPEC_PPM = 500.0
QUALITY_TRAIN_MAX_SEQ = 30
RUN_RULE = 2  # consecutive out-of-limit points required for an SPC alarm


def spc_alarm_batch(pred_df: pd.DataFrame, run_rule: int = RUN_RULE) -> Optional[int]:
    """First batch_seq starting a run of ``run_rule`` consecutive health_alarm=True.

    Returns None if no such run exists.
    """
    d = pred_df.sort_values("batch_seq").reset_index(drop=True)
    above = d["health_alarm"].astype(bool).to_numpy()
    for i in range(len(above) - run_rule + 1):
        if above[i:i + run_rule].all():
            return int(d["batch_seq"].iloc[i])
    return None


def conventional_alarm_batch(
    alarms_with_seq: pd.DataFrame,
    source_prefix: str = "FD-301",
    max_priority: int = 2,
) -> Optional[int]:
    """First batch_seq with an ``source_prefix`` alarm of priority <= max_priority.

    ``alarms_with_seq`` must have columns: source, priority, batch_seq.
    """
    m = alarms_with_seq[
        alarms_with_seq["source"].str.startswith(source_prefix)
        & (alarms_with_seq["priority"] <= max_priority)
    ]
    if m.empty:
        return None
    return int(m["batch_seq"].min())


def quality_failure_batch(
    pred_df: pd.DataFrame,
    spec_ppm: float = QUALITY_SPEC_PPM,
    col: str = "actual_residual_ppm",
) -> Optional[int]:
    """First batch_seq whose residual solvent exceeds ``spec_ppm``."""
    m = pred_df[pred_df[col] > spec_ppm]
    if m.empty:
        return None
    return int(m["batch_seq"].min())


def early_warning_batches(spc: Optional[int], failure: Optional[int]) -> Optional[int]:
    """Batches of lead time between the SPC alarm and the quality failure."""
    if spc is None or failure is None:
        return None
    return failure - spc


def ridge_coefficients(
    features_df: pd.DataFrame,
    features=FD_SENSOR_FEATURES,
    target: str = "residual_solv_ppm",
    train_max_seq: int = QUALITY_TRAIN_MAX_SEQ,
    alpha: float = 1.0,
) -> pd.Series:
    """Standardized Ridge coefficients (features scaled inside the pipeline).

    Trained on batches 1..train_max_seq. Returns a Series indexed by feature,
    sorted by absolute value descending.
    """
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    train = features_df[features_df["batch_seq"] <= train_max_seq]
    X = train[features].astype(float)
    y = train[target].astype(float)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=alpha)),
    ]).fit(X, y)

    coefs = pd.Series(
        model.named_steps["ridge"].coef_, index=features, name="coefficient"
    )
    return coefs.reindex(coefs.abs().sort_values(ascending=False).index)
