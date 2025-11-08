import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import numpy as np
from pathlib import Path

RULES_CSV = Path("data/all_segment_rules_with_industry.csv")

PALETTE = {
    "blue": "#264E86",
    "green": "#2CA58D",
    "orange": "#F4A300",
    "grey": "#8F9BA6",
}

@st.cache_data(show_spinner=False)
def load_rules(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()

    keep = [c for c in ["antecedent","consequent","support","confidence","lift","industry"] if c in df.columns]
    df = df[keep].dropna()

    for c in ("antecedent","consequent"):
        if c in df.columns:
            df[c] = (
                df[c].astype(str)
                     .str.replace(r"[\{\}\[\]\(\)\"']", "", regex=True)
                     .str.replace("  ", " ", regex=False)
                     .str.strip()
            )
    for c in ("support","confidence","lift"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["support","confidence","lift"])

def _split_items(s: str):
    return [x.strip() for x in str(s).split(",") if x.strip()]

def show(DATA_DIR: Path = Path("data")):
    st.title("Basket Patterns: Co-Purchase Relationships")

    df = load_rules(RULES_CSV)
    if df.empty:
        st.error(f"No rules found at {RULES_CSV}. Ensure Team A exported the file.")
        return

    with st.sidebar:
        st.subheader("Rule Filters")
        lift_min = st.slider("Min Lift", 1.0, 20.0, 1.2, 0.1)
        conf_min = st.slider("Min Confidence", 0.0, 1.0, 0.40, 0.05)
        sup_min  = st.slider("Min Support", 0.0, 0.05, 0.002, 0.001)
        industry_opt = None
        if "industry" in df.columns:
            inds = ["(all)"] + sorted(df["industry"].astype(str).unique().tolist())
            industry_opt = st.selectbox("Industry (if available)", inds, index=0)

    q = (df["lift"] >= lift_min) & (df["confidence"] >= conf_min) & (df["support"] >= sup_min)
    if industry_opt and industry_opt != "(all)" and "industry" in df.columns:
        q &= (df["industry"].astype(str) == industry_opt)
    df_f = df.loc[q].copy()
    if df_f.empty:
        st.warning("No rules match current thresholds.")
        return

    st.subheader("Key Statistics")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Rules shown", f"{len(df_f):,}")
    c2.metric("Avg Lift", f"{df_f['lift'].mean():.2f}")
    c3.metric("Avg Confidence", f"{df_f['confidence'].mean():.2f}")
    c4.metric("Avg Support", f"{df_f['support'].mean():.3f}")

    st.subheader("Rule Distribution (Support vs Confidence)")
    fig_sc = px.scatter(
        df_f, x="support", y="confidence", color="lift", size="lift",
        color_continuous_scale="Viridis",
        hover_data=["antecedent","consequent"] + (["industry"] if "industry" in df_f.columns else []),
        title="Association Rule Scatter — color & size by Lift"
    )
    fig_sc.update_traces(marker=dict(opacity=0.85, line=dict(width=0)))
    fig_sc.update_layout(margin=dict(l=0,r=0,t=60,b=0))
    st.plotly_chart(fig_sc, use_container_width=True)

    # ------------------------ Network ------------------------
    st.subheader("Network View of Top Rules")

    edges = []
    for _, r in df_f.iterrows():
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

    edges_df = (pd.DataFrame(edges, columns=["src","dst","lift"])
                  .groupby(["src","dst"], as_index=False)["lift"].mean())

    node_strength = (
        pd.concat([
            edges_df[["src","lift"]].rename(columns={"src":"item"}),
            edges_df[["dst","lift"]].rename(columns={"dst":"item"})
        ])
        .groupby("item", as_index=False)["lift"].sum()
        .rename(columns={"lift":"strength"})
    )

    top_n = st.slider("Top nodes by strength", 40, 250, 120, 10)
    keep_nodes = set(node_strength.sort_values("strength", ascending=False)
                                  .head(top_n)["item"])
    edges_df = edges_df[edges_df["src"].isin(keep_nodes) & edges_df["dst"].isin(keep_nodes)]
    node_strength = node_strength[node_strength["item"].isin(keep_nodes)]

    G = nx.Graph()
    for _, r in edges_df.iterrows():
        G.add_edge(r["src"], r["dst"], weight=float(r["lift"]))

    # spring layout
    pos = nx.spring_layout(G, k=0.6/np.sqrt(max(len(G),1)), iterations=200, seed=7)

  
    valid_nodes = [n for n in node_strength["item"] if n in G.nodes]
    node_strength = node_strength.set_index("item").loc[valid_nodes, "strength"]

    # edge trace
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]; x1, y1 = pos[v]
        edge_x += [x0, x1, None]; edge_y += [y0, y1, None]
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y, mode="lines", hoverinfo="none",
        line=dict(color="rgba(140,140,140,0.45)", width=1)
    )

    # node trace
    ns = node_strength
    size = 10 + 20*(ns - ns.min())/(ns.max() - ns.min() + 1e-9)
    node_x = [pos[n][0] for n in ns.index]
    node_y = [pos[n][1] for n in ns.index]
    label_toggle = st.checkbox("Show labels", value=False)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text" if label_toggle else "markers",
        text=ns.index if label_toggle else None,
        textposition="top center",
        hovertext=[f"{n}<br>Strength: {ns[n]:.2f}" for n in ns.index],
        hoverinfo="text",
        marker=dict(size=size, color=PALETTE["green"],
                    line=dict(width=1, color=PALETTE["blue"]), opacity=0.9),
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
        margin=dict(l=0, r=0, t=60, b=0), height=650
    )
    st.plotly_chart(fig_net, use_container_width=True)

    with st.expander("Rule Table Preview"):
        st.dataframe(df_f.head(100), use_container_width=True)