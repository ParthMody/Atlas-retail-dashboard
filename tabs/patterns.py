# tabs/patterns.py
# Basket Patterns (UK rules only) — segment-aware, ID-free, exploration-focused

import re
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

RULES_CSV = Path("data/all_segment_rules_with_industry.csv")

PALETTE = {
    "blue": "#264E86",
    "green": "#2CA58D",
    "orange": "#F4A300",
    "grey": "#8F9BA6",
}

# Default labels for integer-coded segments (edit here if Team A used different coding)
SEGMENT_MAP_DEFAULT = {
    1: "Browsers · low freq / low spend",
    2: "Deal-seekers · promo-driven",
    3: "Occasional gifters · seasonal",
    4: "Core repeaters · medium freq",
    5: "Loyal multi-basket · high freq",
    6: "High-value · high AOV",
    7: "Bulk buyers · large qty",
    8: "At-risk returners · churn risk",
}

# --- Helper UI utilities ------------------------------------------------------
def _popover(label: str, body_md: str, icon: str = "ℹ️"):
    """
    Renders a compact helper. Uses st.popover when available; falls back to expander.
    """
    if hasattr(st, "popover"):
        with st.popover(f"{icon} {label}", use_container_width=True):
            st.markdown(body_md)
    else:
        with st.expander(f"{icon} {label}"):
            st.markdown(body_md)


def _glossary():
    _popover(
        "Glossary",
        """
**Support** — share of all baskets that contain *both* the left-hand side (antecedent) and right-hand side (consequent).  
**Confidence** — probability of seeing the consequent **given** the antecedent.  
**Lift** — how many times more often the rule happens than random chance ( >1 is good ).  
**Antecedent / Consequent** — “IF … THEN …” parts of a rule (e.g., *IF* Gift Bag *THEN* Greeting Card).  
**Redundant rule** — a rule that adds no new information compared to a simpler rule with similar or higher metrics.
        """,
        icon="📚",
    )


def _filters_help():
    _popover(
        "About the rule filters",
        """
- **Min Lift**: raises the bar for “go-together” strength (how much more often than chance).  
- **Min Confidence**: ensures “IF A THEN B” fires often when A appears.  
- **Min Support**: keeps only rules that occur in enough baskets to matter.

**Why limit Lift to 20?**  
Very high lift values on extremely rare rules can be noisy; capping the slider avoids chasing one-off combinations.
        """,
        icon="🎚️",
    )


def _how_rules_made():
    _popover(
        "How we built these rules (short version)",
        """
**Dataset**: UCI Online Retail II (UK), cleaned to **UK-only**, removed **returns** (Invoice starts with `C`) and lines with non-positive quantity/price.  
**Basketisation**: each invoice → a shopping *basket* of products.  
**Apriori**: `min_support ≥ 0.002`, `min_confidence ≥ 0.40`, `min_lift ≥ 1.20`.  
**Pruning**: dropped duplicates & near-duplicates; kept strongest rules by **lift**.  
**Export**: `antecedent, consequent, support, confidence, lift` (+ optional `industry`).  

**Categories** (if shown): product descriptions were mapped to a handful of readable groups (e.g., *Apparel*, *Household*, *Gifts*) via keyword rules; this is for **storytelling only** and doesn’t affect the mining.
        """,
        icon="🧪",
    )


def _network_help():
    _popover(
        "How to read the network",
        """
- **Nodes** are products. **Edges** connect products that appear together in strong rules.  
- **Size = node strength**: sum of connected rule lifts → bigger nodes = more “network-central” products.  
- **Highlight product**: shows its neighbourhood to reveal bundles and companion items.

**“Top nodes by strength”**  
This keeps only the most connected items (by total lift of edges touching them).  
It declutters the view while preserving the *structure* of popular bundles.
        """,
        icon="🕸️",
    )


# ---------- helpers ----------

def _strip_id(token: str) -> str:
    s = str(token).strip()
    if "|" in s:
        left, right = s.split("|", 1)
        if re.fullmatch(r"\s*\d+\s*", left):
            return right.strip()
    s = re.sub(r"^\s*\d+\s*[-:]\s*", "", s)
    return s.strip()


def _split_items(s: str):
    return [_strip_id(x) for x in str(s).split(",") if x.strip()]


def _label_segment(x, segmap):
    """Map ints like 1,2,3 → descriptive labels; pass through strings."""
    try:
        xi = int(str(x).strip())
        return segmap.get(xi, f"Segment {xi}")
    except ValueError:
        return str(x)


# ---------- data load/clean ----------

@st.cache_data(show_spinner=False)
def load_rules(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()

    seg_col = next((c for c in ["segment", "customer_segment", "rfm_segment", "buyer_segment"] if c in df.columns), None)

    keep_cols = [c for c in ["antecedent", "consequent", "support", "confidence", "lift", "industry"] if c in df.columns]
    if seg_col:
        keep_cols.append(seg_col)

    df = df[keep_cols].dropna(subset=["antecedent", "consequent"])

    for c in ("antecedent", "consequent"):
        if c in df.columns:
            df[c] = (
                df[c]
                .astype(str)
                .str.replace(r"[\{\}\[\]\(\)\"']", "", regex=True)
                .str.replace("  ", " ", regex=False)
                .str.strip()
            )

    for c in ("support", "confidence", "lift"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["support", "confidence", "lift"])

    df["antecedent_clean"] = df["antecedent"].map(lambda s: ", ".join(_split_items(s)))
    df["consequent_clean"] = df["consequent"].map(lambda s: ", ".join(_split_items(s)))

    if seg_col:
        df.rename(columns={seg_col: "segment_raw"}, inplace=True)
        # build a segment map using defaults for ints, pass-through for strings
        # resolve label column now to keep UI clean everywhere
        df["segment"] = df["segment_raw"].map(lambda x: _label_segment(x, SEGMENT_MAP_DEFAULT))

    return df


# ---------- main UI ----------

def show(DATA_DIR: Path = Path("data")):
    st.markdown(
        """
    <div style="background:#F8FAFC;border:1px solid #E5E7EB;padding:10px 14px;border-radius:10px;margin-top:6px">
      <b>What’s here?</b> Explore mined association rules, filter by strength and frequency,
      then use the network to spot bundles and companion products.
    </div>
    """,
        unsafe_allow_html=True,
    )

    st.title("Basket Patterns: Co-Purchase Relationships")

    df = load_rules(RULES_CSV)
    if df.empty:
        st.error(f"No rules found at {RULES_CSV}. Ensure Team A exported the file.")
        return

    # Quick explainer + glossary
    cols = st.columns([1, 1, 1])
    with cols[0]:
        _how_rules_made()
    with cols[1]:
        _filters_help()
    with cols[2]:
        _glossary()

    with st.sidebar:
        st.subheader("Rule Filters")
        _filters_help()  # helper inside the sidebar too (optional)
        lift_min = st.slider("Min Lift", 1.0, 50.0, 1.2, 0.1)   # widened cap to 50 as requested
        conf_min = st.slider("Min Confidence", 0.0, 1.0, 0.40, 0.05)
        sup_min  = st.slider("Min Support", 0.0, 0.05, 0.002, 0.001)

        industry_opt = None
        if "industry" in df.columns:
            inds = ["(all)"] + sorted(df["industry"].astype(str).unique().tolist())
            industry_opt = st.selectbox("Industry (optional)", inds, index=0)

        segment_opt = None
        if "segment" in df.columns:
            segs = ["(all)"] + sorted(df["segment"].astype(str).unique().tolist())
            segment_opt = st.selectbox("Customer segment (optional)", segs, index=0)

    q = (df["lift"] >= lift_min) & (df["confidence"] >= conf_min) & (df["support"] >= sup_min)
    if industry_opt and industry_opt != "(all)" and "industry" in df.columns:
        q &= (df["industry"].astype(str) == industry_opt)
    if segment_opt and segment_opt != "(all)" and "segment" in df.columns:
        q &= (df["segment"].astype(str) == segment_opt)

    df_f = df.loc[q].copy()
    if df_f.empty:
        st.warning("No rules match current thresholds.")
        return

    # KPIs
    st.subheader("Key Statistics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rules shown", f"{len(df_f):,}")
    c2.metric("Avg Lift", f"{df_f['lift'].mean():.2f}")
    c3.metric("Avg Confidence", f"{df_f['confidence'].mean():.2f}")
    c4.metric("Avg Support", f"{df_f['support'].mean():.3f}")

    # Segment legend + distribution
    if "segment" in df_f.columns:
        with st.expander("Customer segments (legend and distribution)"):
            # Legend (all segments present in the filtered set)
            seg_present = df_f["segment"].astype(str).unique().tolist()
            st.markdown("**Legend**")
            st.markdown("\n".join([f"- {s}" for s in sorted(seg_present)]))
            # Distribution
            seg_ct = df_f.groupby("segment")["lift"].agg(["size", "mean"]).reset_index()
            seg_ct = seg_ct.rename(columns={"size": "rules", "mean": "avg_lift"}).sort_values("rules", ascending=False)
            fig_seg = px.bar(
                seg_ct, x="segment", y="rules",
                title="Rules per segment", labels={"rules": "Rules"},
            )
            st.plotly_chart(fig_seg, use_container_width=True)

    # Scatter: Support vs Confidence (clean hover only)
    st.subheader("Rule Distribution (Support vs Confidence)")
    st.caption("Color & size = Lift. Use filters to focus on actionable regions.")
    customdata = np.stack([df_f["antecedent_clean"], df_f["consequent_clean"]], axis=-1)
    fig_sc = px.scatter(
        df_f, x="support", y="confidence", color="lift", size="lift",
        color_continuous_scale="Viridis",
        title="Association Rule Scatter — color & size by Lift",
    )
    fig_sc.update_traces(
        hovertemplate=(
            "<b>Support:</b> %{x:.4f}<br>"
            "<b>Confidence:</b> %{y:.3f}<br>"
            "<b>Lift:</b> %{marker.color:.2f}<br><br>"
            "<b>Antecedent:</b> %{customdata[0]}<br>"
            "<b>Consequent:</b> %{customdata[1]}<extra></extra>"
        ),
        customdata=customdata,
        marker=dict(opacity=0.85, line=dict(width=0)),
    )
    fig_sc.update_layout(margin=dict(l=0, r=0, t=60, b=0))
    st.plotly_chart(fig_sc, use_container_width=True)

    # Network (force layout; ID-free)
    st.subheader("Network View of Top Rules")
    _network_help()
    st.caption("Tip: Start with 80–120 top nodes. Use “Highlight product” to reveal companion items.")

    df_net = df_f.sort_values("lift", ascending=False).head(3000)

    edges = []
    for _, r in df_net.iterrows():
        A = _split_items(r["antecedent"])
        C = _split_items(r["consequent"])
        if not A or not C:
            continue
        w = float(r["lift"])
        for a in A:
            for c in C:
                if a != c:
                    edges.append((a, c, w))
    if not edges:
        st.info("No edges to render with current filters.")
        return

    edges_df = (
        pd.DataFrame(edges, columns=["src", "dst", "lift"])
        .groupby(["src", "dst"], as_index=False)["lift"]
        .mean()
    )
    node_strength = (
        pd.concat(
            [
                edges_df[["src", "lift"]].rename(columns={"src": "item"}),
                edges_df[["dst", "lift"]].rename(columns={"dst": "item"}),
            ]
        )
        .groupby("item", as_index=False)["lift"]
        .sum()
        .rename(columns={"lift": "strength"})
    )

    top_n = st.slider("Top nodes by strength", 40, 250, 120, 10)
    keep_nodes = set(node_strength.sort_values("strength", ascending=False).head(top_n)["item"])
    edges_df = edges_df[edges_df["src"].isin(keep_nodes) & edges_df["dst"].isin(keep_nodes)]
    node_strength = node_strength[node_strength["item"].isin(keep_nodes)]

    G = nx.Graph()
    for _, r in edges_df.iterrows():
        G.add_edge(r["src"], r["dst"], weight=float(r["lift"]))

    pos = nx.spring_layout(G, k=0.6 / np.sqrt(max(len(G), 1)), iterations=200, seed=7)

    valid_nodes = [n for n in node_strength["item"] if n in G.nodes]
    ns = node_strength.set_index("item").loc[valid_nodes, "strength"]

    # edge trace
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x += [x0, x1, None]; edge_y += [y0, y1, None]
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines", hoverinfo="none",
        line=dict(color="rgba(140,140,140,0.45)", width=1),
    )

    # node trace
    size = 10 + 20 * (ns - ns.min()) / (ns.max() - ns.min() + 1e-9)
    node_x = [pos[n][0] for n in ns.index]
    node_y = [pos[n][1] for n in ns.index]
    label_toggle = st.checkbox("Show labels", value=False)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text" if label_toggle else "markers",
        text=list(ns.index) if label_toggle else None,
        textposition="top center",
        hovertext=[f"{n}<br>Strength: {ns[n]:.2f}" for n in ns.index],
        hoverinfo="text",
        marker=dict(size=size, color=PALETTE["green"], line=dict(width=1, color=PALETTE["blue"]), opacity=0.9),
    )

    # highlight
    hi_choice = st.selectbox("Highlight product", ["(none)"] + list(ns.index))
    traces = [edge_trace, node_trace]
    if hi_choice != "(none)" and hi_choice in G:
        nbrs = set(G.neighbors(hi_choice)) | {hi_choice}
        colors = [PALETTE["orange"] if n == hi_choice else ("#6EC5A3" if n in nbrs else "#B5C3D1") for n in ns.index]
        opac   = [1.0 if n in nbrs else 0.25 for n in ns.index]
        node_trace.marker.color = colors
        node_trace.marker.opacity = opac

        hi_x, hi_y = [], []
        for u, v in G.edges():
            if u in nbrs and v in nbrs:
                x0, y0 = pos[u]; x1, y1 = pos[v]
                hi_x += [x0, x1, None]; hi_y += [y0, y1, None]
        hi_edge = go.Scatter(x=hi_x, y=hi_y, mode="lines",
                             line=dict(color=PALETTE["orange"], width=2.5),
                             hoverinfo="none")
        traces = [edge_trace, hi_edge, node_trace]

    fig_net = go.Figure(data=traces)
    fig_net.update_layout(
        title="Product Co-Purchase Network (edge ≈ lift; size ≈ node strength)",
        showlegend=False,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=0, r=0, t=60, b=0), height=650,
    )
    st.plotly_chart(fig_net, use_container_width=True)

    with st.expander("Rule Table Preview"):
        cols = ["antecedent_clean", "consequent_clean", "support", "confidence", "lift"]
        if "industry" in df_f.columns: cols.append("industry")
        if "segment" in df_f.columns: cols.append("segment")
        st.dataframe(df_f[cols].head(200), use_container_width=True)
