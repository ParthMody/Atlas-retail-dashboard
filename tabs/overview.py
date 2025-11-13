import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

ABS_FILE = Path("data/ABS.csv")

# ------------------- Helper UI utilities for Overview ----------------------
def overview_helpbar():
    """Top helper row — concise guidance for first-time viewers."""
    c1, c2, c3 = st.columns([1,1,1])
    with c1.popover("🗂️ About this data", use_container_width=True):
        st.markdown(
            """
- **Source:** Australian Bureau of Statistics — *Retail Trade, Australia (ABS 8501.0)*.  
- **What it shows:** Monthly **retail turnover ($ millions)** by state and industry.  
- **Update cadence:** released monthly, typically lagging 4–5 weeks.  
- **Coverage:** includes physical + online retail (not pure services).
"""
        )
    with c2.popover("📊 How to read the chart", use_container_width=True):
        st.markdown(
            """
- **Each line** = a region’s monthly turnover.  
- **Peaks** → seasonal demand (Nov–Dec); **dips** → off-season.  
- **YoY %** in KPIs compares the latest month vs. same month last year.  
- Optional **3-month moving average** smooths volatility.
"""
        )
    with c3.popover("❓Common misconceptions", use_container_width=True):
        st.markdown(
            """
- **Turnover ≠ number of sales.** It measures **value**, not transaction count.  
- **ABS data ≠ UCI baskets.** ABS gives macro-level context; UCI rules are micro-level.  
- **Industry shares** depend on ABS classification, not your own product mapping.
"""
        )

def overview_docs():
    """Optional long-form explainer at the bottom of the tab."""
    with st.expander("How this overview is constructed (click to expand)", expanded=False):
        st.markdown(
            """
### Data preparation

1. **Column detection:** header-agnostic parsing finds *region*, *date*, *turnover*, *industry*.  
2. **Normalization:** converts turnover to numeric, applies `UNIT_MULT` (×10ⁿ) if present, and rescales to **$ millions**.  
3. **Filtering:** keeps the most recent 3–5 years for readability.  
4. **Aggregation:** sums duplicates by `region × date (× industry)`.

### Visual design

- **Line chart:** monthly trend by region with color palette  
  (`#264E86` NSW, `#2CA58D` VIC, `#F4A300` AUS, gray others).  
- **KPIs:** dynamic metrics showing total turnover and YoY growth.  
- **Bar chart:** appears only when one region selected; shows top industries (last 12 months).

### Interpretation guidance

- **Rising turnover** = higher retail spending, not necessarily higher unit sales.  
- **Stable pattern with annual spikes** = normal retail seasonality.  
- **Divergent state trends** may indicate local policy, tourism, or population shifts.  

**Tip:** Use this page to anchor *micro-level* basket insights in *macro-level* economic reality.
"""
        )


# ---------- Core Logic for Overview Tab ----------
def _detect_cols(csv_path: Path):
    hdr = pd.read_csv(csv_path, nrows=0)
    cols_norm = {c.strip().lower(): c for c in hdr.columns}
    def get(*cands):
        for c in cands:
            if c in cols_norm: return cols_norm[c]
        return None
    region   = get("region", "state", "state/territory", "geography", "geog")
    date     = get("time_period", "time period", "period", "month", "date")
    value    = get("obs_value", "observation value", "value", "turnover", "amount")
    industry = get("industry", "industry_code", "category", "group")
    return region, date, value, industry

@st.cache_data(show_spinner=False, ttl=600)
def load_abs(csv_path: Path, keep_years: int = 5):
    region_col, date_col, value_col, industry_col = _detect_cols(csv_path)
    need = [c for c in [region_col, date_col, value_col, industry_col, "UNIT_MULT"] if c]
    if not all([region_col, date_col, value_col]):
        return pd.DataFrame(), {"region":region_col, "date":date_col, "value":value_col, "industry":industry_col}
    df = pd.read_csv(csv_path, usecols=need, low_memory=False)
    ren = {region_col:"region", date_col:"date", value_col:"turnover"}
    if industry_col: ren[industry_col] = "industry"
    df = df.rename(columns=ren)
    # Clean strings and unit multipliers
    df["turnover"] = pd.to_numeric(df["turnover"], errors="coerce")
    if "UNIT_MULT" in df.columns:
        with pd.option_context("mode.chained_assignment", None):
            df["UNIT_MULT"] = pd.to_numeric(df["UNIT_MULT"], errors="coerce").fillna(0)
            df["turnover"] = df["turnover"] * (10 ** df["UNIT_MULT"])
        df = df.drop(columns=["UNIT_MULT"])
    # Handle date parsing, force to first of month if only year-month
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    # Drop completely empty or invalid
    df = df.dropna(subset=["date", "region", "turnover"])
    # Aggregate to monthly
    df["month"] = df["date"].dt.to_period("M").dt.start_time
    df["date"] = df["month"]
    df = df.drop(columns=["month"])
    # Keep latest N years
    cut = df["date"].max() - pd.DateOffset(years=keep_years)
    df = df[df["date"] >= cut]
    # Aggregate all duplicates
    keys = ["region", "date"] + (["industry"] if "industry" in df.columns else [])
    df = df.groupby(keys, as_index=False)["turnover"].sum()
    if df["turnover"].max() > 1e6:
        df["turnover"] = df["turnover"] / 1e6
    return df, {"region":region_col, "date":date_col, "value":value_col, "industry":industry_col}

def show(DATA_DIR: Path = Path("data")):
    st.title("Overview: Australian Retail Trends (ABS 8501.0)")
    overview_helpbar()  # <<---------------- helper UI row!

    path = DATA_DIR / "ABS.csv"
    if not path.exists():
        st.error(f"File not found: {path}")
        return
    df, detected = load_abs(path)
    if df.empty:
        st.error(f"No usable rows. Detected columns → {detected}. Check header names in ABS.csv.")
        return

    geos = sorted(df["region"].astype(str).unique().tolist())
    default_geo = [g for g in geos if g.startswith("New South Wales") or g.startswith("Victoria")]
    if not default_geo:
        default_geo = geos[:2]

    st.subheader("Filters")
    sel_geo = st.multiselect("Region", geos, default=default_geo, max_selections=4)
    if not sel_geo:
        st.warning("Select at least one region.")
        return
    df_geo = df[df["region"].isin(sel_geo)].copy()

    st.caption(
        f"Data window: {df_geo['date'].min():%b %Y} → {df_geo['date'].max():%b %Y}  ·  "
        f"Regions: {len(df_geo['region'].unique())}"
    )

    # Check for duplicate dates per region
    if df_geo.groupby(["region", "date"]).size().max() > 1:
        st.warning(
            "Aggregated data had duplicate monthly entries per region; these are summed."
        )

    # KPIs for latest month
    latest = df_geo["date"].max()
    now = df_geo[df_geo["date"] == latest].groupby("region", as_index=False)["turnover"].sum()
    prev = df_geo[df_geo["date"] == (latest - pd.DateOffset(years=1))].groupby("region", as_index=False)["turnover"].sum()
    total_now = float(now["turnover"].sum())
    total_prev = float(prev["turnover"].sum()) if not prev.empty else 0.0
    yoy = ((total_now - total_prev) / total_prev * 100.0) if total_prev > 0 else 0.0

    c1, c2 = st.columns(2) if len(sel_geo) == 1 else st.columns(3)
    c1.metric("Total Turnover (latest)", f"${total_now:,.0f}M", f"{yoy:+.1f}% YoY")
    c2.metric(f"{sel_geo[0]} (latest)", f"${float(now[now.region==sel_geo[0]]['turnover'].sum()):,.0f}M" if sel_geo else "—")
    if len(sel_geo) > 1:
        c3 = c2 if len(sel_geo) == 2 else st.columns(3)[2]
        c3.metric(f"{sel_geo[1]} (latest)", f"${float(now[now.region==sel_geo[1]]['turnover'].sum()):,.0f}M")

    st.download_button(
        "Download filtered (CSV)",
        df_geo.to_csv(index=False).encode("utf-8"),
        file_name="abs_filtered.csv",
        mime="text/csv"
    )

    st.subheader("Time Trends")
    smooth = st.toggle("Show 3-month moving average", value=True, help="Adds a 3M MA overlay per region.")
    plot_df = df_geo.copy()
    if smooth:
        plot_df["ma3"] = (
            plot_df.sort_values(["region", "date"])
            .groupby("region")["turnover"]
            .transform(lambda s: s.rolling(3, min_periods=1).mean())
        )

    fig = px.line(
        plot_df, x="date", y="turnover", color="region",
        color_discrete_sequence=["#264E86", "#2CA58D", "#F4A300", "#6c757d"],
        labels={"turnover":"Turnover ($M)", "date":"Month"},
        title="Monthly Retail Turnover"
    )
    if smooth:
        for r, g in plot_df.groupby("region"):
            fig.add_scatter(x=g["date"], y=g["ma3"], mode="lines", name=f"{r} · 3M MA",
                            line=dict(width=2, dash="dash"),
                            showlegend=True)
    fig.update_layout(margin=dict(l=0, r=0, t=60, b=0))
    st.plotly_chart(fig, use_container_width=True)

    # If industry present and only one region selected, quick bar
    if "industry" in df_geo.columns and len(sel_geo) == 1:
        st.subheader(f"Industry Split — {sel_geo[0]} (last 12 months total)")
        last12 = df_geo[df_geo["date"] >= df_geo["date"].max() - pd.DateOffset(months=12)]
        ind = (last12.groupby("industry", as_index=False)["turnover"].sum()
                    .sort_values("turnover", ascending=False).head(12))
        st.bar_chart(ind.set_index("industry")["turnover"])

    with st.expander("Data preview"):
        st.dataframe(df_geo.sort_values(["region","date"]).tail(24), use_container_width=True)

    # Optional: add detailed docs at the bottom
    overview_docs()
