# NCM Smart Factory Demo

> ## ⚠️ 100% SYNTHETIC DATA
> **All data in this project is synthetic, generated for demonstration purposes
> only. No AbbVie data was used, referenced, or derived** (see
> `data/generate_ncm_demo_data.py`).

Predictive-maintenance demo for a small-molecule API final step. Process line:

```
R-101 Reactor -> CR-201 Crystallizer -> FD-301 Nutsche Filter Dryer -> ML-401 Mill
```

Embedded storyline: the **FD-301** filter dryer's vacuum system degrades and the
filter cloth blinds over a 40-batch campaign, driving late-campaign
residual-solvent failures. The equipment sensors drift *long before* the quality
result does — that gap is the predictive-maintenance opportunity.

## Architecture

```
  Ignition / OT   ->   Postgres historian   ->   Databricks (ML)   ->   Streamlit
  (PLC + sensors)      (tag_history,             (PCA/T² health +       (this app,
                        batch_*, alarms)          Ridge quality)         5 tabs)
```

Sensors and alarms land in a Postgres historian; a Databricks notebook builds the
models and writes results back to Postgres; the Streamlit app reads those tables.

## Key finding — three-tier early warning

Everything below is **computed from the data, not hardcoded** (`src/metrics.py`):

| Signal | Batch |
|--------|-------|
| Multivariate SPC health model (Hotelling T², 2-point run rule) | **16** |
| Conventional alarm system (FD-301 source, priority ≤ 2) | **21** |
| First quality failure (residual solvent > 500 ppm) | **35** |

The multivariate model caught the degradation **5 batches earlier than the
existing alarm system** and **19 batches before the first out-of-spec batch** —
a comparison against the status quo, not a bare number.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Configure the DB connection (choose ONE):
#  a) local .env file:
echo 'PGURL=postgresql://user:pass@host/db?sslmode=require' > .env
#  b) Streamlit secrets (for the app / Streamlit Cloud):
cp .streamlit/secrets.toml.example .streamlit/secrets.toml   # then edit it
```

The app resolves the connection from `st.secrets["PGURL"]` first, then falls
back to `PGURL` in `.env` / the environment.

## Running the tests

```bash
python -m pytest -v
```

Tests that need the database auto-skip when `PGURL` is unset/unreachable, so the
suite still runs in CI without a live DB. With the populated DB reachable, the
full suite is **25 passed** (see "Unit Testing Results" below).

## Layout

| Path | Purpose |
|------|---------|
| `src/config.py` | Loads `PGURL` from `.env`; `get_pgurl()` |
| `src/db.py` | `get_conn()` ctx mgr, `query()` -> DataFrame, `list_tables()` |
| `src/features.py` | `build_batch_features()`, `write_batch_features()` |
| `src/metrics.py` | Pure, testable dashboard metrics (SPC/conventional/failure/lead time, Ridge coefs) |
| `app.py` | Streamlit dashboard (5 tabs) |
| `scripts/check_db.py` | Prints row counts of every table |
| `scripts/build_features.py` | Builds + persists `batch_features`, prints shape |
| `notebooks/databricks_degradation_model.py` | Databricks notebook: PCA/T² health + RF quality models; writes `batch_predictions`, `equipment_health` |
| `tests/` | pytest suite |

## Scripts

```bash
python scripts/check_db.py        # 8 tables, all non-zero
python scripts/build_features.py  # build_batch_features() -> (40, 26)
streamlit run app.py              # 5-tab dashboard (uses st.secrets["PGURL"] or .env)
```

## Feature table (`batch_features`)

One row per `batch_id` (40 rows, 26 columns).

- **FD-301 (hero equipment):** `fd_vacuum_ultimate`, `fd_pulldown_min`,
  `fd_filt_time_min`, `fd_cake_dp_max`, `fd_blade_torque_mean`,
  `fd_dry_duration_min`, `fd_lod_final`, `fd_lod_slope_last60`
- **Other units:** `r101_agt_torque_mean`, `cr201_agt_torque_mean`,
  `ml401_vib_mean`, `ml401_amps_mean`, `ml401_throughput_mean`
- **Context:** `alarm_count`, `alarm_count_p1`, `batch_seq`
- **Joined:** `yield_pct`, `purity_pct`, `d50_um`, `residual_solv_ppm`,
  `lod_pct` (results); `disposition`, `operator`, `route`, `start_ts` (master)

## Unit Testing Results

Run all tests:

```bash
python -m pytest -v
```

DB-dependent tests auto-skip if `PGURL` is unset/unreachable. Results below were
captured against the live populated database.

### Slice 1 — Scaffold + DB layer

`tests/test_config.py`, `tests/test_db.py` — **7 passed**

| Test | Result |
|------|--------|
| `test_config.py::test_get_pgurl_returns_value` | PASS |
| `test_config.py::test_get_pgurl_raises_when_unset` | PASS |
| `test_db.py::test_get_conn_commits_and_closes` | PASS |
| `test_db.py::test_get_conn_rolls_back_on_error` | PASS |
| `test_db.py::test_query_returns_dataframe` | PASS |
| `test_db.py::test_list_tables` | PASS |
| `test_db.py::test_live_expected_tables_present_and_populated` | PASS |

`scripts/check_db.py` — 8 tables, all non-zero:

| table | rows |
|-------|------|
| batch_master | 40 |
| batch_results | 40 |
| tag_history | 345,855 |
| alarm_history | 75 |
| capa | 6 |
| equipment_master | 4 |
| poms_route | 9 |
| sap_material | 4 |

### Slice 2 — Feature engineering

`tests/test_features.py` — **6 passed**

| Test | Result | Notes |
|------|--------|-------|
| `test_exactly_40_rows_no_duplicate_batch_ids` | PASS | 40 rows, unique `batch_id` |
| `test_vacuum_trends_with_batch_seq` | PASS | Spearman = **0.99** (> 0.7) |
| `test_filt_time_trends_with_batch_seq` | PASS | Spearman = **0.987** (> 0.7) |
| `test_no_alarms_before_batch_seq_15` | PASS | `alarm_count == 0` for seq ≤ 15 |
| `test_no_column_entirely_nan` | PASS | 0 all-NaN columns |
| `test_write_batch_features_roundtrip` | PASS | `batch_features` returns 40 rows |

`scripts/build_features.py` — `build_batch_features()` -> **(40, 26)**.

### Slice 3 — Degradation model (Databricks notebook)

`notebooks/databricks_degradation_model.py` runs in Databricks; it reads
`batch_features`, fits a PCA/Hotelling-T² equipment-health model + a
RandomForest quality model (train on batches 1–30, holdout 31–40), and writes
`batch_predictions` and `equipment_health` back to Postgres.

Model design (revised):

- **Health baseline = batches 1–15** (documented zero-alarm, in-control Phase I).
  `StandardScaler` + PCA + control limit fit on 1–15 only; batches 16–40 are the
  Phase II monitoring set.
- **Alarm run rule:** a batch is flagged only when **2 consecutive** T² points
  exceed the control limit (Western Electric run rule), so single-batch sensor
  spikes (e.g. batch 9) do not raise false alarms.
- **Quality model = `Ridge` in a `StandardScaler` pipeline**, trained on 1–30,
  holdout 31–40. (A `RandomForestRegressor` was tested first but could not
  extrapolate.)

Model output (executed against the live DB):

- Retained **2 PCs** explaining **93.4%** of baseline (1–15) variance.
- T² control limit (99th pct of baseline 1–15) = **5.45**.
- Holdout (31–40): **MAE = 297.2 ppm**, **R² = −3.10** (see note below).
- Ridge standardized coefficients: `fd_cake_dp_max` +16.3, `fd_vacuum_ultimate`
  −12.0, `fd_filt_time_min` +11.3, `fd_blade_torque_mean` −5.6, `fd_pulldown_min` +5.5.
- **`Equipment health alarm at batch 16. First quality failure at batch 35. Lead time: 19 batches.`**

> **Note on the negative R²:** `residual_solv_ppm` stays in spec (~150–275 ppm)
> through batch 30 and only rises sharply in 31–40, so the failure signal is not
> present in the training target — no model trained on 1–30 can predict the late
> failures. This is *why the multivariate SPC health model, not the regression,
> is the primary early-warning signal.*

`tests/test_model_contract.py` — **2 passed**

| Test | Result |
|------|--------|
| `test_batch_predictions_contract` | PASS — 40 rows, no null `health_score` |
| `test_equipment_health_contract` | PASS — 40 rows, no null `health_score` |

### Slice 4 — Streamlit dashboard (`app.py`)

Five tabs reading from Postgres via `src/db.py` (connection: `st.secrets["PGURL"]`
→ `.env`), all reads cached with `@st.cache_data(ttl=300)`:

1. **Equipment Health** (hero) — four data-computed metric cards, T² control
   chart (baseline shaded, alarm markers) + residual-solvent chart sharing the
   x-axis with three annotated event lines (SPC / conventional / failure).
2. **Failure Drivers** — Ridge standardized-coefficient bar chart + per-sensor
   drift small multiples (no predicted-vs-actual scatter).
3. **Alarm History** — date/equipment/priority filters, table, per-batch bar chart.
4. **CAPA History** — searchable table (`capa` ⋈ `batch_master`), filters,
   expandable per-CAPA detail.
5. **Data Integration** — architecture diagram, source-systems table,
   `poms_route` + `sap_material` stubs.

**Three-tier early-warning story (all computed from the data, not hardcoded):**

| Signal | Batch |
|--------|-------|
| Multivariate SPC health model | **16** |
| Conventional alarm system (FD-301, priority ≤ 2) | **21** |
| Quality failure (residual > 500 ppm) | **35** |
| Early warning vs. status quo | 5 batches earlier than existing alarms; 19 before failure |

`tests/test_metrics.py` — **10 passed** (5 pure-logic + 5 live-DB)

| Test | Result |
|------|--------|
| `test_spc_alarm_needs_two_consecutive` | PASS |
| `test_spc_alarm_none_when_no_run` | PASS |
| `test_conventional_alarm_filters_source_and_priority` | PASS |
| `test_quality_failure_batch` | PASS |
| `test_early_warning` | PASS |
| `test_live_spc_alarm_batch_16` | PASS |
| `test_live_conventional_alarm_batch_21` | PASS |
| `test_live_quality_failure_batch_35` | PASS |
| `test_live_early_warning_19` | PASS |
| `test_live_ridge_coefficients_shape` | PASS |

App render check: Streamlit `AppTest` harness runs all 5 tabs with no uncaught
exception; `streamlit run app.py` serves HTTP 200.

### Combined

**25 passed** (`python -m pytest -v`).
