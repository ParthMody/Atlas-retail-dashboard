# tabs/time.py  — robust Time Dynamics
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px

DATA_DIR = Path("data")
MONTHLY_FILE = DATA_DIR / "retail_monthly.parquet"
ABS_FILE = DATA_DIR / "ABS.csv"

# --------- same header detection strategy as overview.py ----------
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
def _load_abs(csv_path: Path, keep_years: int = 5):
    region_col, date_col, value_col, industry_col = _detect_cols(csv_path)
    need = [c for c in [region_col, date_col, value_col, industry_col] if c]
    if not all([region_col, date_col, value_col]):
        return pd.DataFrame(), {"region":region_col, "date":date_col, "value":value_col, "industry":industry_col}

    df = pd.read_csv(
        csv_path,
        usecols=need,
        dtype={value_col: "float32"},
        low_memory=False,
    )

    # standardise
    ren = {region_col:"region", date_col:"date", value_col:"turnover"}
    if industry_col: ren[industry_col] = "industry"
    df = df.rename(columns=ren)

    # if region missing entirely in file, synthesize national total
    if "region" not in df.columns:
        df["region"] = "Australia"

    # parse month + clean numeric
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["turnover"] = pd.to_numeric(df["turnover"], errors="coerce")
    df = df.dropna(subset=["date", "turnover"])

    # recent N years
    cut = df["date"].max() - pd.DateOffset(years=keep_years)
    df = df[df["date"] >= cut]

    # collapse duplicates
    keys = ["region", "date"] + (["industry"] if "industry" in df.columns else [])
    df = df.groupby(keys, as_index=False)["turnover"].sum()
    return df, {"region":region_col, "date":date_col, "value":value_col, "industry":industry_col}

@st.cache_data(show_spinner=False)
def _load_monthly(path: Path):
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df.dropna(subset=["date"]).sort_values("date")

def show(DATA_DIR_IN: Path | None = None):
    st.title("Time Dynamics")

    # prefer Team-A rollup
    m = _load_monthly(MONTHLY_FILE)
    if not m.empty:
        st.caption("Using Team-A monthly rollup.")
        c1, c2 = st.columns([3, 2])

        with c1:
            fig1 = px.line(
                m, x="date", y="txn_count",
                labels={"date":"Month", "txn_count":"Transactions"},
                title="Monthly Transaction Frequency"
            )
            st.plotly_chart(fig1, use_container_width=True)

        with c2:
            mm = m.copy()
            mm["weekday"] = mm["date"].dt.weekday
            mm["bucket"] = np.where(mm["weekday"] < 5, "Weekday", "Weekend")
            wk = mm.groupby("bucket", as_index=False)["total_sales"].sum()
            fig2 = px.bar(
                wk, x="bucket", y="total_sales",
                labels={"bucket":"", "total_sales":"Sales"},
                title="Weekday vs Weekend Sales"
            )
            st.plotly_chart(fig2, use_container_width=True)

        with st.expander("Data preview"):
            st.dataframe(m.tail(12), use_container_width=True)
        return

    # fallback to ABS.csv with identical detection to overview.py
    if not ABS_FILE.exists():
        st.error(f"ABS file not found: {ABS_FILE}")
        st.caption("Provide either data/retail_monthly.parquet or data/ABS.csv")
        return

    df_abs, detected = _load_abs(ABS_FILE)
    if df_abs.empty:
        st.error(f"ABS file parsed but produced no usable rows.")
        with st.expander("Detected columns"):
            st.write(detected)
        return

    # prefer NSW/VIC if available, else pick top 2 regions by volume
    regions = df_abs["region"].astype(str).unique().tolist()
    if {"New South Wales", "Victoria"}.issubset(regions):
        sel = ["New South Wales", "Victoria"]
        colors = {"New South Wales":"#264E86", "Victoria":"#2CA58D"}
        title_suffix = "NSW vs VIC"
    elif "Australia" in regions:
        sel = ["Australia"]
        colors = {"Australia":"#264E86"}
        title_suffix = "Australia"
    else:
        sel = (df_abs.groupby("region")["turnover"].sum()
                      .sort_values(ascending=False).head(2).index.tolist())
        colors = {sel[i]: ("#264E86" if i==0 else "#2CA58D") for i in range(len(sel))}
        title_suffix = " — ".join(sel)

    d = df_abs[df_abs["region"].isin(sel)].copy()
    monthly = (d.groupby(["region", "date"], as_index=False)["turnover"]
                 .sum().sort_values(["region","date"]))
    monthly["roll3"] = monthly.groupby("region")["turnover"].transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )

    st.caption("Using ABS Retail Trade fallback.")

    # trend + 3-mo average
    fig = px.line(
        monthly, x="date", y="turnover", color="region",
        color_discrete_map=colors,
        labels={"turnover":"Turnover ($M)", "date":"Month", "region":""},
        title=f"Monthly Turnover — {title_suffix}"
    )
    fig_avg = px.line(monthly, x="date", y="roll3", color="region",
                      color_discrete_map=colors)
    for tr in fig_avg.data:
        tr.name = f"{tr.name} (3-mo avg)"
        tr.line.width = 2.5
        tr.line.dash = "dash"
        fig.add_trace(tr)
    fig.update_layout(legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

    # seasonality (if enough months)
    if monthly["date"].nunique() >= 6:
        tmp = monthly.copy()
        tmp["month"] = tmp["date"].dt.month
        seas = tmp.groupby(["region","month"], as_index=False)["turnover"].mean()
        fig2 = px.bar(
            seas, x="month", y="turnover", color="region", barmode="group",
            color_discrete_map=colors,
            labels={"month":"Month (1–12)", "turnover":"Avg Turnover ($M)", "region":""},
            title="Seasonality — Average Turnover by Month"
        )
        st.plotly_chart(fig2, use_container_width=True)

    with st.expander("ABS data preview"):
        st.dataframe(monthly.tail(24), use_container_width=True)
