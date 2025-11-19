import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
import plotly.graph_objects as go

ABS_FILE = Path("data/ABS.csv")

# ------------------- Helper UI utilities for Overview ----------------------


def overview_helpbar():
    """Top helper row — concise guidance for first-time viewers."""
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1.popover("🗂️ About this data", use_container_width=True):
        st.markdown(
            """
- **Source** – Australian Bureau of Statistics, *Retail Trade, Australia (ABS 8501.0)*.  
- **Measure** – Monthly **retail turnover ($ millions)** for **Australia total**, plus a split by industry.  
- **Update frequency** – Published monthly, usually with a 4–5 week delay.
- **Coverage** – Includes both in-store and online retail (excludes pure services). 
"""
        )
    with c2.popover("📊 How to read these chart", use_container_width=True):
        st.markdown(
            """
- The top KPIs summarise **latest month turnover** and the **average month over the last year**.  
- **Seasonality Profile** shows a “typical year”: average turnover by month with a band for how much years vary.
- **Growth vs Same Month Last Year** shows whether turnover is up or down once you strip out normal seasonality.
- **Turnover by Industry** (last 12 months) highlights which sectors drive the Australian total.
"""
        )
    with c3.popover("❓ Things to watch out for", use_container_width=True):
        st.markdown(
            """
- **Turnover ≠ transaction count** – it’s the dollar value of sales, not the number of baskets.  
- **ABS totals ≠ your store sales** – this is macro context; your own data will be smaller but should rhyme in pattern.  
- All numbers are shown in **$ millions**, seasonally adjusted; year-to-year comparisons are **YoY %**, not raw differences.
"""
        )


def overview_docs():
    """Optional long-form explainer at the bottom of the tab."""
    with st.expander("Methodology & Data Notes", expanded=False):
        st.markdown(
            """
### Data preparation

1. **Column detection:** header-agnostic parsing finds *region*, *date*, *turnover*, *industry*, plus `Measure` and `Adjustment Type`.  
2. **Series selection:** keeps only **“Current Prices”** and **“Seasonally Adjusted”** to avoid mixing levels, trends and percentage-change series.  
3. **Normalization:** converts turnover to numeric, applies `UNIT_MULT` (×10ⁿ) if present, and rescales to **$ millions**.  
4. **Filtering:** keeps the most recent 3–5 years for readability.  
5. **Aggregation:** sums duplicates by `region × date (× industry)`.

### Visual design

- **KPIs:** latest month turnover and the average month over the last 12 months.  
- **Seasonality profile:** average month across years with a min–max band → shows a typical year pattern.  
- **YoY line:** year-over-year change (%) in total Australian retail turnover.  
- **Industry bar chart:** last 12 months of turnover by industry (excluding the overall “Total” series).

### Interpretation guidance

- **Rising YoY with stable seasonality** = genuine growth, not just Christmas spikes.  
- **Peaks in specific months** guide stock and staffing planning windows.  
- **Industry mix** highlights which sectors are gaining or losing share within total retail.

Use this page to anchor *micro-level* basket insights in the *macro-level* Australian retail environment.
"""
        )



# ---------- Core Logic for Overview Tab ----------
def _detect_cols(csv_path: Path):
    hdr = pd.read_csv(csv_path, nrows=0)
    cols_norm = {c.strip().lower(): c for c in hdr.columns}

    def get(*cands):
        for c in cands:
            if c in cols_norm:
                return cols_norm[c]
        return None
    region = get("region", "state", "state/territory", "geography", "geog")
    date = get("time_period", "time period", "period", "month", "date")
    value = get("obs_value", "observation value",
                "value", "turnover", "amount")
    industry = get("industry", "industry_code", "category", "group")
    return region, date, value, industry


@st.cache_data(show_spinner=False, ttl=600)
def load_abs(csv_path: Path, keep_years: int = 5):
    # detect core columns as before
    region_col, date_col, value_col, industry_col = _detect_cols(csv_path)

    # second pass over header to find Measure / Adjustment Type
    hdr = pd.read_csv(csv_path, nrows=0)
    cols_norm = {c.strip().lower(): c for c in hdr.columns}

    def pick(*names):
        for name in names:
            if name in cols_norm:
                return cols_norm[name]
        return None

    measure_col = pick("measure", "measures")
    adj_col = pick("adjustment type", "adjustment_type", "adjustment")

    need = [
        c
        for c in [
            region_col,
            date_col,
            value_col,
            industry_col,
            "UNIT_MULT",
            measure_col,
            adj_col,
        ]
        if c
    ]

    if not all([region_col, date_col, value_col]):
        return pd.DataFrame(), {
            "region": region_col,
            "date": date_col,
            "value": value_col,
            "industry": industry_col,
        }

    df = pd.read_csv(csv_path, usecols=need, low_memory=False)

    # rename to internal names
    ren = {region_col: "region", date_col: "date", value_col: "turnover"}
    if industry_col:
        ren[industry_col] = "industry"
    if measure_col:
        ren[measure_col] = "measure"
    if adj_col:
        ren[adj_col] = "adj_type"
    df = df.rename(columns=ren)

    # ---- keep ONLY the level series you care about ----
    if "measure" in df.columns:
        df = df[df["measure"] == "Current Prices"]
    if "adj_type" in df.columns:
        # choose one; this matches what most people use
        df = df[df["adj_type"] == "Seasonally Adjusted"]

    # numeric + unit scaling
    df["turnover"] = pd.to_numeric(df["turnover"], errors="coerce")
    if "UNIT_MULT" in df.columns:
        with pd.option_context("mode.chained_assignment", None):
            df["UNIT_MULT"] = pd.to_numeric(
                df["UNIT_MULT"], errors="coerce"
            ).fillna(0)
            df["turnover"] = df["turnover"] * (10 ** df["UNIT_MULT"])
        df = df.drop(columns=["UNIT_MULT"])

    # dates → month start
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "region", "turnover"])

    df["month"] = df["date"].dt.to_period("M").dt.start_time
    df["date"] = df["month"]
    df = df.drop(columns=["month"])

    # keep last N years
    cut = df["date"].max() - pd.DateOffset(years=keep_years)
    df = df[df["date"] >= cut]

    # aggregate duplicates just in case
    keys = ["region", "date"] + \
        (["industry"] if "industry" in df.columns else [])
    df = df.groupby(keys, as_index=False)["turnover"].sum()

    # scale to $M
    if df["turnover"].max() > 1e6:
        df["turnover"] = df["turnover"] / 1e6

    df["region"] = df["region"].astype(str).str.title()
    if "industry" in df.columns:
        df["industry"] = df["industry"].astype(str).str.title()

    return df, {
        "region": region_col,
        "date": date_col,
        "value": value_col,
        "industry": industry_col,
    }


def _national_total_series(df_geo: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a monthly total turnover series for Australia in $M.

    If an 'Industry' = 'Total' series exists, we use that;
    otherwise we sum across industries.
    """
    df2 = df_geo.copy()

    if "industry" in df2.columns:
        mask_total = df2["industry"].astype(str).str.lower() == "total"
        if mask_total.any():
            df2 = df2[mask_total]

    monthly = (
        df2.groupby("date", as_index=False)["turnover"]
        .sum()
        .sort_values("date")
    )
    return monthly



def show(DATA_DIR: Path = Path("data")):
    st.title("Overview: Australian Retail Trends")
    overview_helpbar()  # <<---------------- helper UI row!

    path = DATA_DIR / "ABS.csv"
    if not path.exists():
        st.error(f"File not found: {path}")
        return
    df, detected = load_abs(path)
    if df.empty:
        st.error(
            f"No usable rows. Detected columns → {detected}. Check header names in ABS.csv.")
        return

    geos = sorted(df["region"].astype(str).unique().tolist())
    default_geo = [g for g in geos if g.startswith(
        "New South Wales") or g.startswith("Victoria")]
    if not default_geo:
        default_geo = geos[:2]

    sel_geo = geos
    df_geo = df.copy()

    start = df_geo["date"].min()
    end = df_geo["date"].max()

    # Clean national total series (Australia, $M)
    monthly_total = _national_total_series(df_geo)

    st.markdown(
        f"""
        <div style="
            margin-top:4px;
            margin-bottom:16px;
            padding:6px 12px;
            border-radius:999px;
            background:#EEF2FF;
            display:inline-flex;
            align-items:center;
            gap:12px;
            font-size:0.85rem;
            color:#111827;">
          <span>📅 <b>{start:%b %Y}</b> – <b>{end:%b %Y}</b></span>
          <span>·</span>
          <span>🌍 <b>Scope:</b> Australia</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------- KPIs ----------
    latest = monthly_total["date"].max()
    latest_val = float(
        monthly_total.loc[monthly_total["date"] == latest, "turnover"].sum()
    )

    prev_date = latest - pd.DateOffset(years=1)
    prev_val = float(
        monthly_total.loc[monthly_total["date"] == prev_date, "turnover"].sum()
    )
    yoy = ((latest_val - prev_val) / prev_val * 100.0) if prev_val > 0 else 0.0

    # average of EXACT last 12 months
    last_12 = monthly_total.tail(12)
    avg_12m = float(last_12["turnover"].mean()) if not last_12.empty else 0.0

    c1, c2 = st.columns(2)
    c1.metric(
        "Australia — Turnover (latest month)",
        f"${latest_val:,.0f}M",
        f"{yoy:+.1f}% vs same month last year",
    )
    c2.metric(
        "Average monthly turnover (last 12 months)",
        f"${avg_12m:,.0f}M",
    )

    st.download_button(
        "Download filtered (CSV)",
        df_geo.to_csv(index=False).encode("utf-8"),
        file_name="abs_filtered.csv",
        mime="text/csv"
    )

    # ---------- Seasonality profile instead of duplicate line chart ----------
    st.subheader("Seasonality Profile")

    season = monthly_total.copy()
    season["MonthNum"] = season["date"].dt.month

    season_stats = (
        season.groupby("MonthNum")["turnover"]
        .agg(["mean", "min", "max"])
        .reset_index()
    )

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    season_stats["Month"] = season_stats["MonthNum"].map(
        {i + 1: m for i, m in enumerate(month_labels)}
    )
    season_stats = season_stats.sort_values("MonthNum")

    fig_season = go.Figure()

    # shaded min–max band
    fig_season.add_trace(
        go.Scatter(
            x=season_stats["Month"],
            y=season_stats["max"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig_season.add_trace(
        go.Scatter(
            x=season_stats["Month"],
            y=season_stats["min"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            name="Range across years",
            hovertemplate=(
                "<b>Month:</b> %{x}<br>"
                "<b>Min:</b> %{y:,.1f}M<br>"
                "<b>Max:</b> %{customdata:,.1f}M"
                "<extra></extra>"
            ),
            customdata=season_stats["max"],
        )
    )

    # average line
    fig_season.add_trace(
        go.Scatter(
            x=season_stats["Month"],
            y=season_stats["mean"],
            mode="lines+markers",
            name="Average month (all years)",
            line=dict(color="#264E86", width=3),
            hovertemplate=(
                "<b>Month:</b> %{x}<br>"
                "<b>Average turnover:</b> %{y:,.1f}M"
                "<extra></extra>"
            ),
        )
    )

    fig_season.update_layout(
        title="Typical Year Pattern — Total Australian Retail Turnover",
        xaxis_title="Month",
        yaxis_title="Turnover ($M)",
        margin=dict(l=0, r=0, t=60, b=0),
    )

    st.plotly_chart(fig_season, use_container_width=True)

    # ---------- YoY growth line (extra overview insight) ----------
    st.subheader("Growth vs Same Month Last Year")

    monthly_total["yoy_pct"] = (
        monthly_total["turnover"].pct_change(periods=12) * 100
    )
    yoy_df = monthly_total.dropna(subset=["yoy_pct"])

    fig_yoy = px.line(
        yoy_df,
        x="date",
        y="yoy_pct",
        labels={"date": "Month", "yoy_pct": "YoY change (%)"},
        title="Year-over-Year Change in Australian Retail Turnover",
    )
    fig_yoy.add_hline(y=0, line_dash="dash", line_width=1)
    fig_yoy.update_traces(
        hovertemplate=(
            "<b>Month:</b> %{x|%b %Y}<br>"
            "<b>YoY change:</b> %{y:.1f}%"
            "<extra></extra>"
        )
    )
    fig_yoy.update_layout(margin=dict(l=0, r=0, t=60, b=0))

    st.plotly_chart(fig_yoy, use_container_width=True)

    # If industry present and only one region selected, quick bar
    if "industry" in df_geo.columns and len(sel_geo) == 1:
        st.subheader(
            f"Turnover by industry (last 12 months)")
        last_cut = monthly_total["date"].max() - pd.DateOffset(months=11)
        last12 = df_geo[df_geo["date"] >= last_cut].copy()

        ind = last12[last12["industry"].str.lower() != "total"]
        ind = ind.groupby("industry", as_index=False)["turnover"].sum()
        ind["industry"] = ind["industry"].astype(str).str.title()

        ind_display = ind.rename(
            columns={
                "industry": "Industry",
                "turnover": "Turnover ($M)",
            }
        )
        fig_ind = px.bar(
            ind_display,
            x="Industry",
            y="Turnover ($M)",
            title="",
        )
        fig_ind.update_traces(
            hovertemplate=(
                "<b>Industry:</b> %{x}<br>"
                "<b>Turnover ($M):</b> %{y:,.1f}"
                "<extra></extra>"
            )
        )
        fig_ind.update_layout(
            xaxis_tickangle=-35,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_ind, use_container_width=True)

    with st.expander("Data Preview"):
        preview = df_geo.sort_values(["region", "date"]).head(24).copy()

        # nicer month label
        preview["date"] = preview["date"].dt.strftime("%b %Y")

        # capitalised / descriptive column names
        preview = preview.rename(
            columns={
                "region": "Region",
                "date": "Month",
                "industry": "Industry",
                "turnover": "Turnover ($M)",
            }
        )

        st.dataframe(preview, use_container_width=True)

    # Optional: add detailed docs at the bottom
    overview_docs()
