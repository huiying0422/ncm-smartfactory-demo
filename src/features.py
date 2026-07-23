"""Feature engineering.

Turns raw ``tag_history`` (one row per tag sample) into a modeling table
with exactly one row per ``batch_id``.

Public API:
    build_batch_features() -> pd.DataFrame
    write_batch_features(df=None, table="batch_features") -> int  # rows written
"""

from typing import Optional

import numpy as np
import pandas as pd

from src.db import get_conn, query

# ---------------------------------------------------------------- tag names
FD = "API01/FD-301"
VAC = f"{FD}/PT/Vacuum"
DP = f"{FD}/DP/Cake"
FD_TORQUE = f"{FD}/AGT/Torque"
WASH = f"{FD}/FT/WashSolvent"
LOD = f"{FD}/LOD"
FILT_TIME = f"{FD}/FiltTime"

R101_TORQUE = "API01/R-101/AGT/Torque"
CR201_TORQUE = "API01/CR-201/AGT/Torque"
ML401_VIB = "API01/ML-401/Vib"
ML401_AMPS = "API01/ML-401/MTR/Amps"
ML401_THROUGHPUT = "API01/ML-401/Throughput"

RELEVANT_TAGS = [
    VAC, DP, FD_TORQUE, WASH, LOD, FILT_TIME,
    R101_TORQUE, CR201_TORQUE,
    ML401_VIB, ML401_AMPS, ML401_THROUGHPUT,
]

# Vacuum threshold (mbar) used to detect end of pull-down.
VAC_PULLDOWN_THRESHOLD = 100.0
# WashSolvent flow (units) above which the wash is considered active.
WASH_ACTIVE_THRESHOLD = 50.0
# Fraction of a batch's vacuum samples treated as the "ultimate" tail.
VAC_TAIL_FRACTION = 0.60
# Window (minutes) for the terminal LOD slope.
LOD_SLOPE_WINDOW_MIN = 60.0


# ---------------------------------------------------------------- helpers
def _minutes(ts: pd.Series, origin: pd.Timestamp) -> pd.Series:
    """Convert a timestamp series to minutes elapsed from ``origin``."""
    return (ts - origin).dt.total_seconds() / 60.0


def _series(g: pd.DataFrame, tag: str) -> pd.DataFrame:
    """Return the (ts-sorted) samples for a single tag within a batch group."""
    s = g[g["tagpath"] == tag]
    return s.sort_values("ts")


def _slope_per_min(ts: pd.Series, value: pd.Series) -> float:
    """Least-squares slope of value vs. time-in-minutes. NaN if < 2 points."""
    if len(value) < 2:
        return np.nan
    x = _minutes(ts, ts.iloc[0]).to_numpy(dtype=float)
    y = value.to_numpy(dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    if mask.sum() < 2 or np.ptp(x[mask]) == 0:
        return np.nan
    return float(np.polyfit(x[mask], y[mask], 1)[0])


def _fd_features(g: pd.DataFrame) -> dict:
    """Compute all FD-301 features for one batch group."""
    out = {}

    # --- vacuum: ultimate (tail median) + pull-down time
    vac = _series(g, VAC)
    if len(vac):
        n = len(vac)
        tail_start = int(np.floor(n * (1.0 - VAC_TAIL_FRACTION)))
        out["fd_vacuum_ultimate"] = float(vac["value"].iloc[tail_start:].median())

        first_ts = vac["ts"].iloc[0]
        below = vac[vac["value"] < VAC_PULLDOWN_THRESHOLD]
        if len(below):
            out["fd_pulldown_min"] = float(
                (below["ts"].iloc[0] - first_ts).total_seconds() / 60.0
            )
        else:
            out["fd_pulldown_min"] = np.nan
    else:
        out["fd_vacuum_ultimate"] = np.nan
        out["fd_pulldown_min"] = np.nan

    # --- filtration time (single-point tag)
    filt = _series(g, FILT_TIME)
    out["fd_filt_time_min"] = (
        float(filt["value"].iloc[-1]) if len(filt) else np.nan
    )

    # --- cake differential pressure (max)
    dp = _series(g, DP)
    out["fd_cake_dp_max"] = float(dp["value"].max()) if len(dp) else np.nan

    # --- blade torque (mean)
    tq = _series(g, FD_TORQUE)
    out["fd_blade_torque_mean"] = float(tq["value"].mean()) if len(tq) else np.nan

    # --- dry duration: last active-wash sample -> last FD-301 sample
    wash = _series(g, WASH)
    fd_rows = g[g["tagpath"].str.startswith(FD)]
    if len(wash) and len(fd_rows):
        active = wash[wash["value"] > WASH_ACTIVE_THRESHOLD]
        if len(active):
            last_wash_ts = active["ts"].iloc[-1]
            last_fd_ts = fd_rows["ts"].max()
            out["fd_dry_duration_min"] = float(
                (last_fd_ts - last_wash_ts).total_seconds() / 60.0
            )
        else:
            out["fd_dry_duration_min"] = np.nan
    else:
        out["fd_dry_duration_min"] = np.nan

    # --- LOD: final value + terminal slope
    lod = _series(g, LOD)
    if len(lod):
        out["fd_lod_final"] = float(lod["value"].iloc[-1])
        last_ts = lod["ts"].iloc[-1]
        window = lod[lod["ts"] >= last_ts - pd.Timedelta(minutes=LOD_SLOPE_WINDOW_MIN)]
        out["fd_lod_slope_last60"] = _slope_per_min(window["ts"], window["value"])
    else:
        out["fd_lod_final"] = np.nan
        out["fd_lod_slope_last60"] = np.nan

    return out


def _mean_tag(g: pd.DataFrame, tag: str) -> float:
    s = _series(g, tag)
    return float(s["value"].mean()) if len(s) else np.nan


def _batch_features(g: pd.DataFrame) -> pd.Series:
    """Aggregate a single batch's tag rows into a feature row."""
    feats = _fd_features(g)
    feats["r101_agt_torque_mean"] = _mean_tag(g, R101_TORQUE)
    feats["cr201_agt_torque_mean"] = _mean_tag(g, CR201_TORQUE)
    feats["ml401_vib_mean"] = _mean_tag(g, ML401_VIB)
    feats["ml401_amps_mean"] = _mean_tag(g, ML401_AMPS)
    feats["ml401_throughput_mean"] = _mean_tag(g, ML401_THROUGHPUT)
    return pd.Series(feats)


# ---------------------------------------------------------------- main build
def build_batch_features() -> pd.DataFrame:
    """Return one feature row per batch_id (joined with results + master)."""
    master = query(
        "SELECT batch_id, disposition, operator, route, start_ts "
        "FROM batch_master"
    )
    results = query(
        "SELECT batch_id, yield_pct, purity_pct, d50_um, "
        "residual_solv_ppm, lod_pct FROM batch_results"
    )

    # --- tag-derived features
    placeholders = ",".join(["%s"] * len(RELEVANT_TAGS))
    tags = query(
        "SELECT batch_id, tagpath, ts, value FROM tag_history "
        f"WHERE tagpath IN ({placeholders})",
        params=RELEVANT_TAGS,
    )
    tags["ts"] = pd.to_datetime(tags["ts"])

    feat = (
        tags.groupby("batch_id", sort=False)
        .apply(_batch_features, include_groups=False)
        .reset_index()
    )

    # --- alarm context
    alarms = query(
        "SELECT batch_id, "
        "COUNT(*) AS alarm_count, "
        "COUNT(*) FILTER (WHERE priority = 1) AS alarm_count_p1 "
        "FROM alarm_history GROUP BY batch_id"
    )

    # --- assemble
    df = master.merge(feat, on="batch_id", how="left")
    df = df.merge(results, on="batch_id", how="left")
    df = df.merge(alarms, on="batch_id", how="left")

    # batches with no alarms -> 0 (not NaN)
    df["alarm_count"] = df["alarm_count"].fillna(0).astype(int)
    df["alarm_count_p1"] = df["alarm_count_p1"].fillna(0).astype(int)

    # sequence 1..N ordered by start_ts
    df["start_ts"] = pd.to_datetime(df["start_ts"])
    df = df.sort_values("start_ts").reset_index(drop=True)
    df["batch_seq"] = np.arange(1, len(df) + 1)

    return df


# ---------------------------------------------------------------- persistence
def _sql_type(dtype) -> str:
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "TIMESTAMP"
    if pd.api.types.is_integer_dtype(dtype):
        return "INTEGER"
    if pd.api.types.is_float_dtype(dtype):
        return "REAL"
    return "TEXT"


def _clean(v):
    if isinstance(v, (list, tuple, np.ndarray, pd.Series)):
        return v
    if pd.isna(v):
        return None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    return v


def write_batch_features(
    df: Optional[pd.DataFrame] = None, table: str = "batch_features"
) -> int:
    """Create-or-replace ``table`` in Postgres and write ``df`` to it.

    Returns the number of rows written.
    """
    from psycopg2.extras import execute_values

    if df is None:
        df = build_batch_features()

    cols = list(df.columns)
    col_defs = ", ".join(f'"{c}" {_sql_type(df[c].dtype)}' for c in cols)
    ddl = f'DROP TABLE IF EXISTS {table}; CREATE TABLE {table} ({col_defs});'

    col_list = ", ".join(f'"{c}"' for c in cols)
    records = [
        tuple(_clean(v) for v in row)
        for row in df.itertuples(index=False, name=None)
    ]

    with get_conn() as conn:
        cur = conn.cursor()
        cur.execute(ddl)
        execute_values(
            cur, f"INSERT INTO {table} ({col_list}) VALUES %s", records
        )
        cur.close()

    return len(records)
