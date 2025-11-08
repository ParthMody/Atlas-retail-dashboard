import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

ABS_FILE = Path("data/ABS.csv")

# --------- header detection strategy ----------
def _detect_cols(csv_path: Path):
    hdr = pd.read_csv(csv_path, nrows=0)
    cols_norm = {c.strip().lower(): c for c in hdr.columns}

    def get(*cands):
        for c in cands:
            if c in cols_norm:
                return cols_norm[c]
        return None

    region   = get("region", "state", "state/territory", "geography", "geog")
    date     = get("time_period", "time period", "period", "month", "date")
    value    = get("obs_value", "observation value", "value", "turnover", "amount")
    industry = get("industry", "industry_code", "category", "group")
    return region, date, value, industry

@st.cache_data(show_spinner=False)
def load_abs(csv_path: Path, keep_years: int = 5):
    region_col, date_col, value_col, industry_col = _detect_cols(csv_path)
    need = [c for c in [region_col, date_col, value_col, industry_col] if c]
    if not all([region_col, date_col, value_col]):
        return pd.DataFrame(), {"region":region_col, "date":date_col, "value":value_col, "industry":industry_col}

    df = pd.read_csv(
        csv_path,
        usecols=need,
        dtype={region_col: "category", value_col: "float32"},
        low_memory=False,
    )

    # standardise column names
    ren = {region_col:"region", date_col:"date", value_col:"turnover"}
    if industry_col: ren[industry_col] = "industry"
    df = df.rename(columns=ren)

    # if region missing entirely in file, synthesize national total
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "region", "turnover"])

    # recent N years
    cut = df["date"].max() - pd.DateOffset(years=keep_years)
    df = df[df["date"] >= cut]

    # collapse duplicates
    keys = ["region", "date"] + (["industry"] if "industry" in df.columns else [])
    df = df.groupby(keys, as_index=False)["turnover"].sum()
    return df, {"region":region_col, "date":date_col, "value":value_col, "industry":industry_col}

def show(DATA_DIR: Path = Path("data")):
    st.title("Overview: Australian Retail Trends (ABS 8501.0)")

    path = DATA_DIR / "ABS.csv"
    if not path.exists():
        st.error(f"File not found: {path}")
        return

    df, detected = load_abs(path)

    if df.empty:
        st.error(f"No usable rows. Detected columns → {detected}. Check header names in ABS.csv.")
        return

    # available geographies
    geos = sorted(df["region"].astype(str).unique().tolist())

    # try to default to NSW/VIC; if absent fall back to whatever exists
    default_geo = [g for g in geos if g in ("New South Wales", "Victoria")]
    if not default_geo:
        default_geo = geos[:2]

    st.subheader("Filters")
    sel_geo = st.multiselect("Region", geos, default=default_geo, max_selections=4)
    if not sel_geo:
        st.warning("Select at least one region.")
        return

    df_geo = df[df["region"].isin(sel_geo)].copy()

    # KPIs (latest month)
    latest = df_geo["date"].max()
    now = df_geo[df_geo["date"] == latest].groupby("region", as_index=False)["turnover"].sum()
    prev = df_geo[df_geo["date"] == (latest - pd.DateOffset(years=1))].groupby("region", as_index=False)["turnover"].sum()
    total_now = float(now["turnover"].sum())
    total_prev = float(prev["turnover"].sum()) if not prev.empty else 0.0
    yoy = ((total_now - total_prev) / total_prev * 100.0) if total_prev > 0 else 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Turnover (latest)", f"${total_now:,.0f}M", f"{yoy:+.1f}% YoY")
    c2.metric(f"{sel_geo[0]} (latest)", f"${float(now[now.region==sel_geo[0]]['turnover'].sum()):,.0f}M" if sel_geo else "—")
    if len(sel_geo) > 1:
        c3.metric(f"{sel_geo[1]} (latest)", f"${float(now[now.region==sel_geo[1]]['turnover'].sum()):,.0f}M")

    # Line chart
    fig = px.line(
        df_geo,
        x="date", y="turnover", color="region",
        color_discrete_sequence=["#264E86", "#2CA58D", "#F4A300", "#6c757d"],
        labels={"turnover":"Turnover ($M)", "date":"Month"},
        title="Monthly Retail Turnover"
    )
    fig.update_layout(margin=dict(l=0, r=0, t=60, b=0))
    st.plotly_chart(fig, use_container_width=True)

    # If industry is present and only one region selected, show quick bar
    if "industry" in df_geo.columns and len(sel_geo) == 1:
        st.subheader(f"Industry Split — {sel_geo[0]} (last 12 months total)")
        last12 = df_geo[df_geo["date"] >= df_geo["date"].max() - pd.DateOffset(months=12)]
        ind = (last12.groupby("industry", as_index=False)["turnover"].sum()
                      .sort_values("turnover", ascending=False).head(12))
        st.bar_chart(ind.set_index("industry")["turnover"])

    with st.expander("Data preview"):
        st.dataframe(df_geo.sort_values(["region","date"]).tail(24), use_container_width=True)
