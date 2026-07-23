"""
NCM Demo - Synthetic Small Molecule API Manufacturing Data Generator
=====================================================================
100% SYNTHETIC DATA. No AbbVie data used, referenced, or derived.

Process: Final API step
    R-101 Reactor -> CR-201 Crystallizer -> FD-301 Nutsche Filter Dryer -> ML-401 Mill

Embedded storyline:
    FD-301 vacuum pump degrades + filter cloth blinds over ~40 batches.
    Pull-down time increases, filtration time increases, drying endpoint
    (LOD) drifts high -> late batches fail residual solvent spec.

Usage:
    # CSV output (no DB needed):
    python generate_ncm_demo_data.py

    # Postgres output (Neon / Supabase):
    export PGURL="postgresql://user:pass@host/db?sslmode=require"
    python generate_ncm_demo_data.py --db
"""

import argparse
import csv
import math
import os
import random
from datetime import datetime, timedelta
from pathlib import Path

random.seed(42)

# ---------------------------------------------------------------- config
N_BATCHES = 40
START_DATE = datetime(2026, 1, 6, 6, 0, 0)
BATCH_INTERVAL_HRS = 108          # ~4.5 days between batch starts
SAMPLE_SEC = 60                   # tag history sample rate
PRODUCT = "ABV-7742 API"
ROUTE = "RTE-7742-03"
OPERATORS = ["OP-1104", "OP-2287", "OP-3391", "OP-4416", "OP-5502"]

OUT = Path("./ncm_demo_data")
OUT.mkdir(exist_ok=True)

# Spec limits
SPEC = {
    "yield_pct":        (88.0, 102.0),
    "purity_pct":       (99.0, 100.0),
    "d50_um":           (45.0, 95.0),
    "residual_solv_ppm": (0.0, 500.0),   # <- the failure mode
    "lod_pct":          (0.0, 0.50),
}


# ---------------------------------------------------------------- helpers
def noise(sigma):
    return random.gauss(0, sigma)


def degradation_factor(batch_idx):
    """
    Sensor-observable wear: 0.0 -> 1.0 across the campaign.
    Drifts EARLY and gradually - this is the leading indicator the model
    is supposed to catch (vacuum, torque, vibration, filtration time).
    """
    x = batch_idx / N_BATCHES
    return x ** 1.8


def dry_penalty_factor(batch_idx):
    """
    Impact on drying capability. Steeper curve, so quality failures land
    LATE in the campaign - well after the sensor signals start drifting.
    That gap is the predictive-maintenance story.
    """
    x = batch_idx / N_BATCHES
    return x ** 2.5


def ramp(t, t0, t1, v0, v1):
    """Linear ramp between two times."""
    if t <= t0:
        return v0
    if t >= t1:
        return v1
    return v0 + (v1 - v0) * (t - t0) / (t1 - t0)


# ---------------------------------------------------------------- phases
def build_batch(batch_idx, start_ts):
    """
    Returns (batch_row, results_row, tag_rows, alarm_rows)
    Timeline (minutes from batch start):
        0-90     R-101   charge + heat to reaction temp
        90-330   R-101   reaction hold
        330-390  R-101   quench / transfer
        390-450  CR-201  charge + seed
        450-750  CR-201  controlled cooling crystallization
        750-780  CR-201  transfer to FD-301
        780-D1   FD-301  filtration (duration degrades)
        D1-D2    FD-301  wash
        D2-D3    FD-301  vacuum drying (duration degrades)
        D3-D4    ML-401  milling
    """
    deg = degradation_factor(batch_idx)
    dry_deg = dry_penalty_factor(batch_idx)
    tags, alarms = [], []

    def T(tagpath, minute, value):
        tags.append((tagpath, start_ts + timedelta(minutes=minute), round(value, 4)))

    def A(minute, source, priority, name, desc):
        alarms.append((
            start_ts + timedelta(minutes=minute), source, priority, name, desc,
            random.choice(OPERATORS),
        ))

    # ---------------- FD-301 degradation-driven durations
    # Vacuum pull-down: 12 min when healthy -> 34 min when worn
    pulldown_min = 12 + 22 * deg + noise(0.8)
    # Filtration: 45 min healthy -> 96 min with cloth blinding
    filt_min = 45 + 51 * deg + noise(2.0)
    # Achievable ultimate vacuum: 15 mbar healthy -> 62 mbar worn
    ult_vac = 15 + 47 * deg + noise(1.5)
    # Drying: 240 min healthy -> 560 min (but capped by schedule -> incomplete dry)
    dry_needed = 240 + 320 * dry_deg + noise(6.0)
    dry_allowed = 300          # fixed campaign schedule window
    dry_actual = min(dry_needed, dry_allowed)
    dry_shortfall = max(0.0, dry_needed - dry_allowed)

    # ---------------- R-101 Reactor
    for m in range(0, 391, SAMPLE_SEC // 60 or 1):
        if m <= 90:
            bt = ramp(m, 0, 90, 22, 78) + noise(0.35)
            jt = bt + 6 + noise(0.4)
        elif m <= 330:
            bt = 78 + noise(0.30) + 0.4 * math.sin(m / 22)
            jt = 82 + noise(0.35)
        else:
            bt = ramp(m, 330, 390, 78, 25) + noise(0.5)
            jt = bt - 5 + noise(0.4)
        T("API01/R-101/TT/Batch", m, bt)
        T("API01/R-101/TT/Jacket", m, jt)
        T("API01/R-101/PT/Head", m, 1.05 + 0.02 * math.sin(m / 15) + noise(0.01))
        T("API01/R-101/AGT/Speed", m, 120 + noise(1.2))
        # slow mechanical wear on reactor agitator too (secondary signal)
        T("API01/R-101/AGT/Torque", m, 41 + 5 * deg + noise(0.9))
        T("API01/R-101/LT/Level", m, 68 + noise(0.5) if m > 60 else ramp(m, 0, 60, 0, 68))
        T("API01/R-101/PH", m, 7.2 + noise(0.05))
        if 60 <= m <= 120:
            T("API01/R-101/FT/Reagent", m, 210 + noise(6))
        else:
            T("API01/R-101/FT/Reagent", m, 0 + abs(noise(0.4)))

    # ---------------- CR-201 Crystallizer
    for m in range(390, 781, 1):
        if m <= 450:
            bt = 60 + noise(0.4)
        elif m <= 750:
            bt = ramp(m, 450, 750, 60, 8) + noise(0.35)
        else:
            bt = 8 + noise(0.3)
        T("API01/CR-201/TT/Batch", m, bt)
        T("API01/CR-201/TT/Jacket", m, bt - 7 + noise(0.5))
        T("API01/CR-201/AGT/Speed", m, 85 + noise(0.9))
        T("API01/CR-201/AGT/Torque", m, 33 + 2.5 * deg + noise(0.8))
        T("API01/CR-201/CoolRate", m, 10.4 + noise(0.35))
        T("API01/CR-201/SeedFlag", m, 1 if 445 <= m <= 455 else 0)

    # ---------------- FD-301 Nutsche Filter Dryer  (HERO EQUIPMENT)
    fd_start = 780
    filt_end = fd_start + filt_min
    wash_end = filt_end + 55
    dry_end = wash_end + dry_actual

    m = fd_start
    while m <= dry_end:
        rel = m - fd_start

        # vacuum: pull down, then hold at achievable ultimate
        if rel <= pulldown_min:
            vac = ramp(rel, 0, pulldown_min, 1013, ult_vac) + noise(3)
        else:
            vac = ult_vac + noise(1.2)
        T("API01/FD-301/PT/Vacuum", m, max(vac, 1))

        # cake differential pressure rises as cloth blinds
        if m <= filt_end:
            dp = ramp(m, fd_start, filt_end, 0.15, 0.55 + 0.85 * deg) + noise(0.02)
        else:
            dp = 0.05 + noise(0.01)
        T("API01/FD-301/DP/Cake", m, max(dp, 0))

        # blade torque climbs with wear + heavier cake
        T("API01/FD-301/AGT/Torque", m, 28 + 14 * deg + noise(1.1))

        # jacket + product temp during drying
        if m <= wash_end:
            jt, pt = 24 + noise(0.4), 22 + noise(0.4)
        else:
            jt = 62 + noise(0.5)
            pt = ramp(m, wash_end, dry_end, 24, 55) + noise(0.5)
        T("API01/FD-301/TT/Jacket", m, jt)
        T("API01/FD-301/TT/Product", m, pt)

        # wash solvent flow
        T("API01/FD-301/FT/WashSolvent",
          m, 145 + noise(4) if filt_end < m <= wash_end else abs(noise(0.3)))

        # LOD trend: exponential dry-down, slower when vacuum is poor
        if m > wash_end:
            frac = (m - wash_end) / max(dry_needed, 1)
            lod = 12.0 * math.exp(-5.48 * frac) + 0.06 + noise(0.015)
        else:
            lod = 12.0 + noise(0.1)
        T("API01/FD-301/LOD", m, max(lod, 0.02))

        m += 1

    T("API01/FD-301/FiltTime", dry_end, filt_min)
    T("API01/FD-301/PullDownTime", dry_end, pulldown_min)

    # FD-301 alarms scale with degradation
    if ult_vac > 30:
        A(fd_start + pulldown_min + 2, "FD-301/PT/Vacuum", 2,
          "VAC_HIGH", f"Vacuum above target ({ult_vac:.0f} mbar vs 25 mbar target)")
    if ult_vac > 45:
        A(fd_start + pulldown_min + 4, "FD-301/PT/Vacuum", 1,
          "VAC_CRITICAL", "Vacuum system unable to achieve process setpoint")
    if filt_min > 75:
        A(filt_end, "FD-301/DP/Cake", 2,
          "FILT_TIME_EXCEEDED", f"Filtration exceeded 75 min ({filt_min:.0f} min)")
    if dry_shortfall > 0:
        A(dry_end, "FD-301/LOD", 1,
          "DRY_ENDPOINT_NOT_MET", "Drying window elapsed before LOD endpoint reached")
    if 28 + 14 * deg > 38:
        A(wash_end, "FD-301/AGT/Torque", 3,
          "TORQUE_HIGH", "Blade drive torque above normal operating band")

    # ---------------- ML-401 Mill
    mill_start = dry_end + 30
    for m in range(int(mill_start), int(mill_start) + 95):
        T("API01/ML-401/MTR/Amps", m, 18.5 + 2.2 * deg + noise(0.4))
        T("API01/ML-401/Speed", m, 4200 + noise(35))
        T("API01/ML-401/Vib", m, 2.1 + 1.6 * deg + noise(0.12))
        T("API01/ML-401/Throughput", m, 42 - 3 * deg + noise(1.1))
    if 2.1 + 1.6 * deg > 3.2:
        A(mill_start + 40, "ML-401/Vib", 3,
          "VIB_ELEVATED", "Mill bearing vibration above baseline")

    # ---------------- Batch results
    # Residual solvent is driven by incomplete drying -> the failure mode
    final_lod = 12.0 * math.exp(-5.48 * (dry_actual / max(dry_needed, 1))) + 0.06
    residual = 180 + 1250 * max(0.0, final_lod - 0.18) + noise(25)
    residual = max(residual, 60)

    yield_pct = 96.5 - 3.0 * deg + noise(0.9)
    purity = 99.62 - 0.22 * deg + noise(0.05)
    d50 = 68 + 9 * deg + noise(3.5)

    results = {
        "yield_pct": round(yield_pct, 2),
        "purity_pct": round(purity, 3),
        "d50_um": round(d50, 1),
        "residual_solv_ppm": round(residual, 1),
        "lod_pct": round(final_lod, 3),
    }

    failures = [k for k, v in results.items()
                if not (SPEC[k][0] <= v <= SPEC[k][1])]
    if not failures:
        disposition = "Released"
    elif "residual_solv_ppm" in failures or "purity_pct" in failures:
        disposition = "Rejected" if residual > 750 else "Investigation"
    else:
        disposition = "Investigation"

    batch_id = f"B26-{7000 + batch_idx:04d}"
    batch_row = {
        "batch_id": batch_id,
        "product": PRODUCT,
        "route": ROUTE,
        "start_ts": start_ts,
        "end_ts": start_ts + timedelta(minutes=mill_start + 95),
        "operator": random.choice(OPERATORS),
        "primary_unit": "FD-301",
        "disposition": disposition,
        "spec_failures": ";".join(failures) if failures else "",
    }
    results_row = {"batch_id": batch_id, **results}

    tag_rows = [(batch_id, tp, ts, v) for tp, ts, v in tags]
    alarm_rows = [(batch_id, ts, src, pri, nm, de, ack)
                  for ts, src, pri, nm, de, ack in alarms]
    return batch_row, results_row, tag_rows, alarm_rows


# ---------------------------------------------------------------- CAPA
def build_capa(batches):
    """CAPA records linked to equipment + batch + operator + route."""
    capa = []
    n = 0
    for b in batches:
        if b["disposition"] in ("Investigation", "Rejected"):
            n += 1
            fails = b["spec_failures"]
            if "residual_solv_ppm" in fails:
                cat, rc = ("Equipment Performance",
                           "FD-301 vacuum system degradation - drying endpoint not achieved "
                           "within scheduled window")
            elif "d50_um" in fails:
                cat, rc = ("Process Performance",
                           "Particle size distribution shift - crystallization cooling profile")
            else:
                cat, rc = ("Under Investigation", "Root cause investigation in progress")
            capa.append({
                "capa_id": f"CAPA-26-{300 + n:04d}",
                "opened_ts": b["end_ts"] + timedelta(days=2),
                "equipment": b["primary_unit"],
                "batch_id": b["batch_id"],
                "route": b["route"],
                "operator": b["operator"],
                "category": cat,
                "description": f"Batch {b['batch_id']} out of specification: {fails}",
                "root_cause": rc,
                "status": "Closed" if n <= 2 else "Open",
            })
    return capa


# ---------------------------------------------------------------- stubs
POMS_ROUTE = [
    {"route": ROUTE, "step_no": 10, "unit": "R-101",
     "phase": "CHARGE",       "phase_desc": "Charge starting material and solvent"},
    {"route": ROUTE, "step_no": 20, "unit": "R-101",
     "phase": "REACT",        "phase_desc": "Heat and hold at reaction temperature"},
    {"route": ROUTE, "step_no": 30, "unit": "R-101",
     "phase": "QUENCH",       "phase_desc": "Quench and cool"},
    {"route": ROUTE, "step_no": 40, "unit": "CR-201",
     "phase": "SEED",         "phase_desc": "Charge and seed"},
    {"route": ROUTE, "step_no": 50, "unit": "CR-201",
     "phase": "CRYSTALLIZE",  "phase_desc": "Controlled cooling crystallization"},
    {"route": ROUTE, "step_no": 60, "unit": "FD-301",
     "phase": "FILTER",       "phase_desc": "Filter mother liquor"},
    {"route": ROUTE, "step_no": 70, "unit": "FD-301",
     "phase": "WASH",         "phase_desc": "Displacement wash"},
    {"route": ROUTE, "step_no": 80, "unit": "FD-301",
     "phase": "DRY",          "phase_desc": "Vacuum dry to LOD endpoint"},
    {"route": ROUTE, "step_no": 90, "unit": "ML-401",
     "phase": "MILL",         "phase_desc": "Mill to target PSD"},
]

SAP_MATERIAL = [
    {"material": "RM-4471", "description": "Starting material 7742-SM",
     "uom": "KG", "lot_prefix": "SM26"},
    {"material": "RM-8820", "description": "Reaction solvent (IPA)",
     "uom": "L",  "lot_prefix": "SV26"},
    {"material": "RM-8821", "description": "Wash solvent (heptane)",
     "uom": "L",  "lot_prefix": "SV26"},
    {"material": "FG-7742", "description": PRODUCT,
     "uom": "KG", "lot_prefix": "AP26"},
]

EQUIP_MASTER = [
    {"equipment": "R-101",  "description": "Jacketed glass-lined reactor, 2000 L",
     "criticality": "High",   "last_pm": "2025-11-14"},
    {"equipment": "CR-201", "description": "Jacketed crystallizer, 2000 L",
     "criticality": "High",   "last_pm": "2025-11-20"},
    {"equipment": "FD-301", "description": "Nutsche filter dryer, 1.5 m2",
     "criticality": "High",   "last_pm": "2025-10-02"},
    {"equipment": "ML-401", "description": "Conical screening mill",
     "criticality": "Medium", "last_pm": "2025-12-08"},
]


# ---------------------------------------------------------------- writers
def write_csv(name, rows, header):
    p = OUT / f"{name}.csv"
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  {p}  ({len(rows):,} rows)")


def write_db(batches, results, tags, alarms, capa):
    import psycopg2
    from psycopg2.extras import execute_values

    url = os.environ.get("PGURL")
    if not url:
        raise SystemExit("Set PGURL env var, e.g. export PGURL='postgresql://...'")

    ddl = """
    DROP TABLE IF EXISTS tag_history, alarm_history, batch_results,
                         capa, batch_master, poms_route, sap_material,
                         equipment_master CASCADE;

    CREATE TABLE batch_master (
        batch_id TEXT PRIMARY KEY, product TEXT, route TEXT,
        start_ts TIMESTAMP, end_ts TIMESTAMP, operator TEXT,
        primary_unit TEXT, disposition TEXT, spec_failures TEXT);

    CREATE TABLE batch_results (
        batch_id TEXT PRIMARY KEY REFERENCES batch_master(batch_id),
        yield_pct REAL, purity_pct REAL, d50_um REAL,
        residual_solv_ppm REAL, lod_pct REAL);

    CREATE TABLE tag_history (
        batch_id TEXT, tagpath TEXT, ts TIMESTAMP, value REAL);
    CREATE INDEX ix_tag ON tag_history (tagpath, ts);
    CREATE INDEX ix_tag_batch ON tag_history (batch_id);

    CREATE TABLE alarm_history (
        batch_id TEXT, ts TIMESTAMP, source TEXT, priority INT,
        alarm_name TEXT, description TEXT, ack_by TEXT);

    CREATE TABLE capa (
        capa_id TEXT PRIMARY KEY, opened_ts TIMESTAMP, equipment TEXT,
        batch_id TEXT, route TEXT, operator TEXT, category TEXT,
        description TEXT, root_cause TEXT, status TEXT);

    CREATE TABLE poms_route (
        route TEXT, step_no INT, unit TEXT, phase TEXT, phase_desc TEXT);
    CREATE TABLE sap_material (
        material TEXT, description TEXT, uom TEXT, lot_prefix TEXT);
    CREATE TABLE equipment_master (
        equipment TEXT, description TEXT, criticality TEXT, last_pm DATE);
    """

    conn = psycopg2.connect(url)
    cur = conn.cursor()
    cur.execute(ddl)

    execute_values(cur, "INSERT INTO batch_master VALUES %s",
                   [tuple(b.values()) for b in batches])
    execute_values(cur, "INSERT INTO batch_results VALUES %s",
                   [tuple(r.values()) for r in results])
    execute_values(cur, "INSERT INTO tag_history VALUES %s", tags, page_size=5000)
    execute_values(cur, "INSERT INTO alarm_history VALUES %s", alarms)
    execute_values(cur, "INSERT INTO capa VALUES %s",
                   [tuple(c.values()) for c in capa])
    execute_values(cur, "INSERT INTO poms_route VALUES %s",
                   [tuple(r.values()) for r in POMS_ROUTE])
    execute_values(cur, "INSERT INTO sap_material VALUES %s",
                   [tuple(r.values()) for r in SAP_MATERIAL])
    execute_values(cur, "INSERT INTO equipment_master VALUES %s",
                   [tuple(r.values()) for r in EQUIP_MASTER])

    conn.commit()
    cur.close()
    conn.close()
    print(f"  Loaded to Postgres: {len(tags):,} tag rows, "
          f"{len(batches)} batches, {len(capa)} CAPA records")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", action="store_true",
                    help="Write to Postgres via PGURL env var")
    args = ap.parse_args()

    batches, results, tags, alarms = [], [], [], []
    ts = START_DATE
    for i in range(1, N_BATCHES + 1):
        b, r, t, a = build_batch(i, ts)
        batches.append(b)
        results.append(r)
        tags.extend(t)
        alarms.extend(a)
        ts += timedelta(hours=BATCH_INTERVAL_HRS)

    capa = build_capa(batches)

    print("\nSYNTHETIC DATA - no AbbVie data used\n")
    print(f"Batches      : {len(batches)}")
    print(f"Tag rows     : {len(tags):,}")
    print(f"Alarms       : {len(alarms)}")
    print(f"CAPA records : {len(capa)}")
    disp = {}
    for b in batches:
        disp[b["disposition"]] = disp.get(b["disposition"], 0) + 1
    print(f"Dispositions : {disp}\n")

    if args.db:
        write_db(batches, results, tags, alarms, capa)
    else:
        write_csv("batch_master", [tuple(b.values()) for b in batches],
                  list(batches[0].keys()))
        write_csv("batch_results", [tuple(r.values()) for r in results],
                  list(results[0].keys()))
        write_csv("tag_history", tags,
                  ["batch_id", "tagpath", "ts", "value"])
        write_csv("alarm_history", alarms,
                  ["batch_id", "ts", "source", "priority",
                   "alarm_name", "description", "ack_by"])
        write_csv("capa", [tuple(c.values()) for c in capa],
                  list(capa[0].keys()))
        write_csv("poms_route", [tuple(r.values()) for r in POMS_ROUTE],
                  list(POMS_ROUTE[0].keys()))
        write_csv("sap_material", [tuple(r.values()) for r in SAP_MATERIAL],
                  list(SAP_MATERIAL[0].keys()))
        write_csv("equipment_master", [tuple(r.values()) for r in EQUIP_MASTER],
                  list(EQUIP_MASTER[0].keys()))


if __name__ == "__main__":
    main()
