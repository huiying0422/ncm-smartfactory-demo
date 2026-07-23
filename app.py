"""NCM Smart Factory Demo — Streamlit dashboard.

Reads from Postgres via src/db.py. Connection resolution order:
  1. st.secrets["PGURL"]  (Streamlit Cloud)
  2. PGURL from .env / environment  (local)

Run:  streamlit run app.py
"""

import os

import streamlit as st

# ---------------------------------------------------------------- connection
# Make the connection string available to src.config.get_pgurl() before the DB
# layer is used. Prefer Streamlit secrets, fall back to .env / env var.
try:
    if "PGURL" in st.secrets:
        os.environ["PGURL"] = st.secrets["PGURL"]
except Exception:
    # No secrets.toml present locally — that's fine, .env will be used.
    pass

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402

from src.db import query  # noqa: E402
from src.metrics import (  # noqa: E402
    FD_SENSOR_FEATURES,
    QUALITY_SPEC_PPM,
    conventional_alarm_batch,
    early_warning_batches,
    quality_failure_batch,
    ridge_coefficients,
    spc_alarm_batch,
)

SYNTHETIC_BANNER = (
    "100% synthetic data generated for demonstration. No AbbVie data used."
)

st.set_page_config(page_title="NCM Smart Factory Demo", layout="wide")


# ---------------------------------------------------------------- cached reads
@st.cache_data(ttl=300)
def load_predictions() -> pd.DataFrame:
    return query("SELECT * FROM batch_predictions ORDER BY batch_seq")


@st.cache_data(ttl=300)
def load_features() -> pd.DataFrame:
    return query("SELECT * FROM batch_features ORDER BY batch_seq")


@st.cache_data(ttl=300)
def load_alarms_with_seq() -> pd.DataFrame:
    return query(
        "SELECT a.batch_id, a.ts, a.source, a.priority, a.alarm_name, "
        "a.description, a.ack_by, bm.start_ts, bp.batch_seq "
        "FROM alarm_history a "
        "JOIN batch_master bm ON bm.batch_id = a.batch_id "
        "LEFT JOIN batch_predictions bp ON bp.batch_id = a.batch_id "
        "ORDER BY a.ts"
    )


@st.cache_data(ttl=300)
def load_capa() -> pd.DataFrame:
    return query(
        "SELECT c.capa_id, c.opened_ts, c.equipment, c.batch_id, "
        "c.category, c.status, c.description, c.root_cause, "
        "bm.route, bm.operator, bm.disposition, bm.start_ts "
        "FROM capa c LEFT JOIN batch_master bm ON bm.batch_id = c.batch_id "
        "ORDER BY c.opened_ts"
    )


@st.cache_data(ttl=300)
def load_table(name: str) -> pd.DataFrame:
    return query(f"SELECT * FROM {name}")


# ---------------------------------------------------------------- sidebar
def render_sidebar() -> None:
    st.sidebar.title("NCM Smart Factory")
    st.sidebar.markdown("**Process flow**")
    st.sidebar.markdown(
        "R-101 Reactor  \n"
        "→ CR-201 Crystallizer  \n"
        "→ FD-301 Nutsche Filter Dryer  \n"
        "→ ML-401 Mill"
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Architecture**")
    st.sidebar.markdown(
        "Ignition (OT) → Postgres historian → Databricks (ML) → this app"
    )
    st.sidebar.markdown("---")
    st.sidebar.caption(SYNTHETIC_BANNER)


# ---------------------------------------------------------------- Tab 1
def tab_equipment_health() -> None:
    st.subheader("FD-301 Equipment Health — predictive early warning")

    pred = load_predictions()
    control_limit = float(pred["control_limit"].iloc[0])

    spc = spc_alarm_batch(pred)
    alarms_seq = load_alarms_with_seq()
    conv = conventional_alarm_batch(alarms_seq)
    fail = quality_failure_batch(pred)
    lead = early_warning_batches(spc, fail)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Multivariate SPC alarm", f"batch {spc}")
    c2.metric("Conventional alarm system", f"batch {conv}")
    c3.metric("Quality failure", f"batch {fail}")
    c4.metric("Early warning", f"{lead} batches")

    # --- health score chart
    fig = go.Figure()
    fig.add_vrect(x0=0.5, x1=15.5, fillcolor="green", opacity=0.08,
                  line_width=0, annotation_text="Baseline 1–15",
                  annotation_position="top left")
    fig.add_trace(go.Scatter(
        x=pred["batch_seq"], y=pred["health_score"],
        mode="lines+markers", name="Health score (T²)",
        line=dict(color="#1f77b4"),
    ))
    alarms = pred[pred["health_alarm"].astype(bool)]
    fig.add_trace(go.Scatter(
        x=alarms["batch_seq"], y=alarms["health_score"],
        mode="markers", name="Alarm (T² > limit)",
        marker=dict(color="red", size=10, symbol="x"),
    ))
    fig.add_hline(y=control_limit, line_dash="dash", line_color="red",
                  annotation_text=f"Control limit ({control_limit:.1f})")
    fig.update_layout(
        height=340, margin=dict(t=30, b=10),
        xaxis_title="Batch sequence", yaxis_title="T² health score",
        legend=dict(orientation="h", y=1.15),
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- residual solvent chart, same x-axis, with the three event lines
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=pred["batch_seq"], y=pred["actual_residual_ppm"],
        mode="lines+markers", name="Residual solvent (ppm)",
        line=dict(color="#555"),
    ))
    fig2.add_hline(y=QUALITY_SPEC_PPM, line_dash="dot", line_color="black",
                   annotation_text=f"{QUALITY_SPEC_PPM:.0f} ppm spec")
    for x, label, color in [
        (spc, f"SPC alarm (b{spc})", "green"),
        (conv, f"Conventional (b{conv})", "orange"),
        (fail, f"Failure (b{fail})", "red"),
    ]:
        if x is not None:
            fig2.add_vline(x=x, line_dash="dash", line_color=color,
                           annotation_text=label, annotation_position="top")
    fig2.update_layout(
        height=340, margin=dict(t=30, b=10),
        xaxis_title="Batch sequence", yaxis_title="Residual solvent (ppm)",
        legend=dict(orientation="h", y=1.15),
    )
    fig2.update_xaxes(range=[0.5, 40.5])
    fig.update_xaxes(range=[0.5, 40.5])
    st.plotly_chart(fig2, use_container_width=True)

    if None not in (spc, conv, fail):
        st.caption(
            f"The multivariate SPC health model flagged FD-301 at **batch {spc}** — "
            f"**{conv - spc} batches earlier** than the conventional alarm system "
            f"(batch {conv}), and **{fail - spc} batches** before the first "
            f"out-of-spec batch (batch {fail}). Same sensors, earlier signal."
        )


# ---------------------------------------------------------------- Tab 2
def tab_failure_drivers() -> None:
    st.subheader("Failure Drivers — which sensors drive the failure")

    feats = load_features()
    coefs = ridge_coefficients(feats)

    colors = ["#d62728" if v > 0 else "#1f77b4" for v in coefs.values]
    fig = go.Figure(go.Bar(
        x=coefs.values, y=coefs.index, orientation="h",
        marker_color=colors,
    ))
    fig.update_layout(
        height=320, margin=dict(t=30, b=10),
        xaxis_title="Standardized Ridge coefficient (ppm per 1 SD)",
        yaxis=dict(autorange="reversed"),
        title="Sensor contribution to predicted residual solvent",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "The **multivariate SPC health model is the trigger** — it detects "
        "abnormal equipment behavior directly from the sensors. This Ridge "
        "regression is a companion that identifies **which** sensors drive the "
        "eventual quality failure. Red bars push residual solvent up, blue bars "
        "pull it down; longer bars matter more."
    )

    st.markdown("**Per-sensor drift across the campaign**")
    cols = st.columns(len(FD_SENSOR_FEATURES))
    for col, feat in zip(cols, FD_SENSOR_FEATURES):
        sub = go.Figure()
        sub.add_trace(go.Scatter(
            x=feats["batch_seq"], y=feats[feat], mode="markers",
            marker=dict(color="#1f77b4", size=5),
        ))
        # simple least-squares trend line to make the drift obvious
        m = feats[["batch_seq", feat]].dropna()
        if len(m) >= 2:
            b1, b0 = np.polyfit(m["batch_seq"], m[feat], 1)
            xs = np.array([m["batch_seq"].min(), m["batch_seq"].max()])
            sub.add_trace(go.Scatter(x=xs, y=b0 + b1 * xs, mode="lines",
                                     line=dict(color="#d62728", dash="dot")))
        sub.update_layout(
            height=240, margin=dict(t=40, b=10, l=10, r=10),
            showlegend=False, title=dict(text=feat, font_size=12),
            xaxis_title=None, yaxis_title=None,
        )
        col.plotly_chart(sub, use_container_width=True)


# ---------------------------------------------------------------- Tab 3
def tab_alarm_history() -> None:
    st.subheader("Alarm History")
    alarms = load_alarms_with_seq()
    alarms = alarms.copy()
    alarms["ts"] = pd.to_datetime(alarms["ts"])

    c1, c2, c3 = st.columns(3)
    min_d, max_d = alarms["ts"].min().date(), alarms["ts"].max().date()
    dr = c1.date_input("Date range", (min_d, max_d),
                       min_value=min_d, max_value=max_d)
    equips = sorted({s.split("/")[0] for s in alarms["source"].dropna()})
    eq_sel = c2.multiselect("Equipment", equips, default=equips)
    prios = sorted(alarms["priority"].dropna().unique().tolist())
    pr_sel = c3.multiselect("Priority", prios, default=prios)

    f = alarms.copy()
    if isinstance(dr, tuple) and len(dr) == 2:
        f = f[(f["ts"].dt.date >= dr[0]) & (f["ts"].dt.date <= dr[1])]
    f = f[f["source"].str.split("/").str[0].isin(eq_sel)]
    f = f[f["priority"].isin(pr_sel)]

    st.dataframe(
        f[["ts", "batch_id", "batch_seq", "source", "priority",
           "alarm_name", "description", "ack_by"]],
        width="stretch", hide_index=True,
    )

    by_batch = (
        f.dropna(subset=["batch_seq"]).groupby("batch_seq").size()
        .reset_index(name="alarm_count")
    )
    if not by_batch.empty:
        fig = px.bar(by_batch, x="batch_seq", y="alarm_count",
                     title="Alarm count per batch")
        fig.update_layout(height=300, margin=dict(t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No alarms match the current filters.")


# ---------------------------------------------------------------- Tab 4
def tab_capa_history() -> None:
    st.subheader("CAPA History")
    capa = load_capa()

    c1, c2, c3 = st.columns(3)
    eqs = sorted(capa["equipment"].dropna().unique().tolist())
    sts = sorted(capa["status"].dropna().unique().tolist())
    cats = sorted(capa["category"].dropna().unique().tolist())
    eq_sel = c1.multiselect("Equipment", eqs, default=eqs)
    st_sel = c2.multiselect("Status", sts, default=sts)
    cat_sel = c3.multiselect("Category", cats, default=cats)

    search = st.text_input("Search (batch, description, root cause)")

    f = capa[
        capa["equipment"].isin(eq_sel)
        & capa["status"].isin(st_sel)
        & capa["category"].isin(cat_sel)
    ]
    if search:
        s = search.lower()
        f = f[
            f["batch_id"].str.lower().str.contains(s, na=False)
            | f["description"].str.lower().str.contains(s, na=False)
            | f["root_cause"].str.lower().str.contains(s, na=False)
        ]

    st.dataframe(
        f[["capa_id", "opened_ts", "equipment", "batch_id", "category",
           "status", "route", "operator"]],
        width="stretch", hide_index=True,
    )

    st.markdown("**Details**")
    for r in f.itertuples(index=False):
        with st.expander(f"{r.capa_id} — {r.equipment} — {r.status}"):
            st.markdown(f"**Linked batch:** {r.batch_id}")
            st.markdown(f"**Route:** {r.route}")
            st.markdown(f"**Operator:** {r.operator}")
            st.markdown(f"**Category:** {r.category}")
            st.markdown(f"**Description:** {r.description}")
            st.markdown(f"**Root cause:** {r.root_cause}")


# ---------------------------------------------------------------- Tab 5
def tab_data_integration() -> None:
    st.subheader("Data Integration Architecture")

    stages = ["Ignition (OT)", "Postgres historian", "Databricks (ML)",
              "Streamlit app"]
    fig = go.Figure()
    for i, s in enumerate(stages):
        fig.add_shape(type="rect", x0=i * 2.4, x1=i * 2.4 + 2.0, y0=0, y1=1,
                      fillcolor="#eef3fb", line=dict(color="#1f77b4"))
        fig.add_annotation(x=i * 2.4 + 1.0, y=0.5, text=s, showarrow=False)
        if i < len(stages) - 1:
            fig.add_annotation(x=i * 2.4 + 2.2, y=0.5, text="→",
                               showarrow=False, font=dict(size=22))
    fig.update_xaxes(visible=False, range=[-0.2, len(stages) * 2.4])
    fig.update_yaxes(visible=False, range=[-0.2, 1.2])
    fig.update_layout(height=160, margin=dict(t=10, b=10),
                      plot_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Source systems**")
    sources = pd.DataFrame([
        ("OSI PI", "Continuous tag historian (temperatures, pressures)",
         "Structure defined"),
        ("Ignition", "OT/SCADA layer — live sensor + alarm feed", "Connected"),
        ("POMS", "MES route / phase definitions", "Connected"),
        ("SAP", "Material master, lot genealogy", "Connected"),
        ("LIMS", "Lab results (assay, residual solvent)", "Structure defined"),
    ], columns=["System", "Contribution", "Status"])
    st.dataframe(sources, width="stretch", hide_index=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**POMS route (stub)**")
        st.dataframe(load_table("poms_route"), width="stretch",
                     hide_index=True)
    with c2:
        st.markdown("**SAP material (stub)**")
        st.dataframe(load_table("sap_material"), width="stretch",
                     hide_index=True)


# ---------------------------------------------------------------- main
def main() -> None:
    render_sidebar()
    st.warning(SYNTHETIC_BANNER)
    st.title("NCM Smart Factory — Predictive Maintenance Demo")

    t1, t2, t3, t4, t5 = st.tabs([
        "Equipment Health", "Failure Drivers", "Alarm History",
        "CAPA History", "Data Integration",
    ])
    with t1:
        tab_equipment_health()
    with t2:
        tab_failure_drivers()
    with t3:
        tab_alarm_history()
    with t4:
        tab_capa_history()
    with t5:
        tab_data_integration()


if __name__ == "__main__":
    main()
