"""
Delhivery — Network Intelligence Dashboard
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time
import random

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Delhivery Network Intelligence",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background:#0D1117; }
  [data-testid="stSidebar"]          { background:#161B22; }
  [data-testid="metric-container"]   { background:#161B22; border-radius:8px;
                                       padding:12px; border:1px solid #30363D; }
  .block-container { padding-top:1.2rem; }
  h1,h2,h3        { color:#E6EDF3 !important; }
  p, label        { color:#8B949E !important; }
  .stMetric label { color:#8B949E !important; font-size:12px; }
  .stMetric [data-testid="stMetricValue"] { color:#E6EDF3 !important; font-size:22px; }
  .badge-critical { background:#E8354A22; color:#E8354A; border:1px solid #E8354A55;
                    padding:2px 10px; border-radius:12px; font-size:11px; font-weight:600; }
  .badge-high     { background:#F5A62322; color:#F5A623; border:1px solid #F5A62355;
                    padding:2px 10px; border-radius:12px; font-size:11px; font-weight:600; }
  .badge-medium   { background:#4A90D922; color:#4A90D9; border:1px solid #4A90D955;
                    padding:2px 10px; border-radius:12px; font-size:11px; font-weight:600; }
  div[data-testid="stHorizontalBlock"] { gap: 0.75rem; }
  .section-card   { background:#161B22; border:1px solid #30363D; border-radius:10px;
                    padding:16px 20px; margin-bottom:12px; }
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────
@st.cache_data
def load_data():
    corridor_df = pd.read_csv("corridor_audit_final.csv")

    hub_sla_df = pd.DataFrame({
        "hub_id": [
            "IND000000ACB","IND562132AAA","IND421302AAG",
            "IND501359AAE","IND712311AAA","IND411033AAA",
            "IND110037AAM","IND160002AAC","IND751002AAB","IND382430AAB"
        ],
        "sla_contribution_pct": [25.6,14.1,10.8,6.2,5.5,5.2,4.2,3.7,3.0,2.7],
        "median_delay_ratio":   [1.56,1.50,1.76,1.62,1.58,1.45,1.61,1.48,1.53,1.42],
        "intervention": [
            "Staggered dispatch + parallel corridor",
            "Staggered dispatch",
            "Capacity expansion",
            "Route type shift",
            "Scheduling optimisation",
            "Route type shift",
            "Scheduling optimisation",
            "Monitor",
            "Monitor",
            "Monitor"
        ],
        "severity":["Critical","Critical","High","High","High",
                    "Medium","Medium","Medium","Medium","Medium"]
    })

    model_df = pd.DataFrame({
        "Model":      ["Graph+XGBoost (corrected)","Graph+XGBoost","GraphSAGE+XGBoost",
                       "Graph+RF","XGBoost Baseline","Random Forest","Linear Regression","OSRM Raw"],
        "Type":       ["Graph","Graph","Graph","Graph","Baseline","Baseline","Baseline","Baseline"],
        "MAE":        [10.55, 11.01, 11.08, 11.99, 13.07, 13.52, 14.46, 18.21],
        "Within15":   [33.9,  32.1,  31.4,  26.3,  25.0,  24.1,  23.7,  8.8],
    })

    ftl_df = pd.DataFrame({
        "Bucket":        ["Hyper-local (<50km)","Short (50-150km)","Long (>150km)"],
        "Carting_time":  [17, 45, None],
        "FTL_time":      [33, 48, None],
        "FTL_extra_cost":[152, 418, None],
        "SLA_savings":   [81,  242, None],
        "Recommendation":["Carting","Carting","FTL (reliability)"],
    })

    return corridor_df, hub_sla_df, model_df, ftl_df

corridor_df, hub_sla_df, model_df, ftl_df = load_data()


# ── Simulated live delay scores ────────────────────────────────
def get_live_scores(hub_df, noise=0.06):
    """Add small random noise to delay ratios to simulate live feed."""
    df = hub_df.copy()
    df["live_delay"] = df["median_delay_ratio"] + np.random.uniform(
        -noise, noise, len(df))
    df["risk_score"] = ((df["live_delay"] - 1.0) /
                        (df["live_delay"].max() - 1.0) * 100).clip(0, 100)
    df["risk_score"] = df["risk_score"].round(1)
    return df


# ── Build graph ───────────────────────────────────────────────
@st.cache_resource
def build_graph(corridor_df):
    G = nx.DiGraph()
    for _, r in corridor_df.iterrows():
        G.add_edge(r["source_center"], r["destination_center"],
                   weight=r["median_delay_ratio"],
                   sla=r["sla_contribution_pct"],
                   is_chronic=int(r["is_chronic"]),
                   intervention=r["intervention"])
    return G

G = build_graph(corridor_df)


# ── Network Plotly figure ──────────────────────────────────────
def make_network_figure(live_df, corridor_df, top_n=150, highlight_hub=None):
    top_c   = corridor_df.nlargest(top_n, "trip_count")
    G_sub   = nx.DiGraph()
    for _, r in top_c.iterrows():
        G_sub.add_edge(r["source_center"], r["destination_center"],
                       intervention=r["intervention"],
                       is_chronic=int(r["is_chronic"]),
                       sla=r["sla_contribution_pct"],
                       delay=r["median_delay_ratio"])

    pos = nx.spring_layout(G_sub, k=2.8, seed=42, iterations=60)

    # Pull top hubs to centre
    for h in live_df["hub_id"].head(3):
        if h in pos:
            pos[h] = pos[h] * 0.25

    int_colors = {
        "scheduling_optimisation": "#E8354A",
        "facility_upgrade":        "#F5A623",
        "route_type_shift":        "#7B68EE",
        "parallel_route":          "#FF6B6B",
        "monitor":                 "#2D3748"
    }

    edge_traces = []
    for src, dst, d in G_sub.edges(data=True):
        x0,y0 = pos.get(src,(0,0))
        x1,y1 = pos.get(dst,(0,0))
        col   = int_colors.get(d.get("intervention","monitor"), "#2D3748")
        alpha = 0.85 if d.get("is_chronic") else 0.3
        edge_traces.append(go.Scatter(
            x=[x0,x1,None], y=[y0,y1,None],
            mode="lines",
            line=dict(width=1.2 if d.get("is_chronic") else 0.5, color=col),
            opacity=alpha,
            hoverinfo="none",
            showlegend=False
        ))

    sev_color = {"Critical":"#E8354A","High":"#F5A623",
                 "Medium":"#4A90D9","Normal":"#607D8B"}
    live_map  = live_df.set_index("hub_id")[["live_delay","risk_score","severity",
                                              "sla_contribution_pct","intervention"]].to_dict("index")

    nx_list, ny_list, nc, ns, nt = [], [], [], [], []
    for node in G_sub.nodes():
        x,y = pos.get(node,(0,0))
        info = live_map.get(node,{})
        sev  = info.get("severity","Normal")
        sla  = info.get("sla_contribution_pct", 0)
        ld   = info.get("live_delay",1.0)
        rs   = info.get("risk_score",0)
        itn  = info.get("intervention","monitor")
        nx_list.append(x); ny_list.append(y)
        nc.append(sev_color.get(sev,"#607D8B"))
        ns.append(max(10, sla * 14) if sla > 0 else 8)
        nt.append(
            f"<b>{node}</b><br>"
            f"SLA contribution : {sla:.1f}%<br>"
            f"Live delay ratio : {ld:.2f}×<br>"
            f"Risk score       : {rs:.0f}/100<br>"
            f"Severity         : {sev}<br>"
            f"Action           : {itn.replace('_',' ')}"
        )

    node_trace = go.Scatter(
        x=nx_list, y=ny_list,
        mode="markers",
        marker=dict(size=ns, color=nc, line=dict(width=1, color="#0D1117")),
        text=nt, hoverinfo="text",
        showlegend=False
    )

    # Labels for top-5
    ann = []
    for _, row in live_df.head(5).iterrows():
        hub = row["hub_id"]
        if hub in pos:
            x,y = pos[hub]
            ann.append(dict(x=x, y=y+0.055,
                            text=f"<b>{hub.replace('IND','')}</b><br>{row['sla_contribution_pct']:.1f}%",
                            showarrow=False,
                            font=dict(size=9, color="white"),
                            bgcolor="#1C2331", bordercolor="#555",
                            borderwidth=1, borderpad=3, opacity=0.9))

    legend_shapes = []
    legend_texts  = []
    items = [("Critical hub","#E8354A"),("High hub","#F5A623"),
             ("Medium hub","#4A90D9"),("",""),
             ("Scheduling fix","#E8354A"),("Facility upgrade","#F5A623"),
             ("Route shift","#7B68EE"),("Monitor","#2D3748")]
    for i,(label,color) in enumerate(items):
        if not label: continue
        yp = 0.98 - i*0.055
        legend_shapes.append(dict(type="circle" if i<4 else "rect",
                                  x0=-1.55,x1=-1.45,y0=yp-0.018,y1=yp+0.018,
                                  fillcolor=color, line=dict(width=0),
                                  xref="x", yref="y"))
        legend_texts.append(dict(x=-1.40,y=yp,text=label,
                                 showarrow=False,xref="x",yref="y",
                                 font=dict(size=9,color="#B0BEC5"),xanchor="left"))

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        paper_bgcolor="#0D1117", plot_bgcolor="#0D1117",
        margin=dict(l=0,r=0,t=0,b=0),
        xaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
        yaxis=dict(showgrid=False,zeroline=False,showticklabels=False),
        annotations=ann + legend_texts,
        shapes=legend_shapes,
        height=520,
        hoverlabel=dict(bgcolor="#161B22",bordercolor="#30363D",
                        font=dict(color="white",size=11))
    )
    return fig


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🚚 Delhivery Network Intelligence")
    st.markdown("---")

    page = st.radio("Navigate", [
        "🌐 Live Network",
        "🔥 Bottleneck Hubs",
        "📊 ETA Model",
        "🚛 FTL vs Carting",
        "📋 Corridor Audit"
    ])

    st.markdown("---")
    st.markdown("**Network Summary**")
    st.metric("Total Hubs",       f"{G.number_of_nodes():,}")
    st.metric("Active Corridors", f"{G.number_of_edges():,}")
    st.metric("Chronic Corridors",f"{int(corridor_df['is_chronic'].sum()):,}")

    st.markdown("---")
    auto_refresh = st.toggle("🔄 Auto-refresh live scores", value=True)
    refresh_sec  = st.slider("Refresh interval (sec)", 5, 60, 15)
    top_n_edges  = st.slider("Corridors shown on map", 50, 300, 150, step=50)

    st.markdown("---")
    st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")


# ═══════════════════════════════════════════════════════════════
# PAGE 1 — LIVE NETWORK
# ═══════════════════════════════════════════════════════════════
if page == "🌐 Live Network":

    st.markdown("## 🌐 Live Network — Delay Risk Monitor")

    if auto_refresh:
        st_autorefresh = st.empty()

    live_df = get_live_scores(hub_sla_df)

    # ── KPI row ──────────────────────────────────────────────
    k1,k2,k3,k4,k5 = st.columns(5)
    breach_now = round(7.71 + random.uniform(-0.3, 0.3), 2)
    k1.metric("Network SLA Breach %", f"{breach_now}%",
              delta=f"{breach_now-7.71:+.2f}%", delta_color="inverse")
    k2.metric("Critical Hubs",   "2",  "ACB + IND562132AAA")
    k3.metric("Chronic Corridors",
              str(int(corridor_df["is_chronic"].sum())), "of 2,781")
    k4.metric("Revenue at Risk (mo)", "₹41.1L",
              "Top 3 hubs")
    k5.metric("ETA Model Accuracy", "33.9%",
              "+25.1pp vs OSRM", delta_color="normal")

    st.markdown("---")

    # ── Network map ──────────────────────────────────────────
    col_map, col_risk = st.columns([3, 1])
    with col_map:
        st.markdown("#### Network Graph — node size ∝ SLA breach contribution")
        fig = make_network_figure(live_df, corridor_df, top_n=top_n_edges)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})

    with col_risk:
        st.markdown("#### Live Risk Scores")
        for _, row in live_df.iterrows():
            sev   = row["severity"]
            badge = f'<span class="badge-{sev.lower()}">{sev}</span>'
            score = row["risk_score"]
            bar_c = "#E8354A" if sev=="Critical" else \
                    "#F5A623" if sev=="High" else "#4A90D9"
            pct   = int(score)
            st.markdown(
                f"""<div style='margin-bottom:10px;padding:8px 10px;
                background:#161B22;border-radius:8px;border:1px solid #30363D'>
                <div style='display:flex;justify-content:space-between;
                align-items:center;margin-bottom:4px'>
                <span style='color:#E6EDF3;font-size:11px;font-weight:600'>
                {row['hub_id'].replace('IND','')}</span>{badge}</div>
                <div style='background:#0D1117;border-radius:4px;height:6px'>
                <div style='background:{bar_c};width:{pct}%;height:6px;
                border-radius:4px'></div></div>
                <div style='display:flex;justify-content:space-between;margin-top:3px'>
                <span style='color:#8B949E;font-size:10px'>
                {row['live_delay']:.2f}× delay</span>
                <span style='color:{bar_c};font-size:10px;font-weight:600'>
                {score:.0f}/100</span></div></div>""",
                unsafe_allow_html=True
            )

    if auto_refresh:
        time.sleep(refresh_sec)
        st.rerun()


# ═══════════════════════════════════════════════════════════════
# PAGE 2 — BOTTLENECK HUBS
# ═══════════════════════════════════════════════════════════════
elif page == "🔥 Bottleneck Hubs":
    st.markdown("## 🔥 Bottleneck Hub Analysis")

    # SLA concentration
    c1, c2 = st.columns([1.6, 1])
    with c1:
        st.markdown("#### SLA Breach Contribution — Top 10 Hubs")
        sev_color_map = {"Critical":"#E8354A","High":"#F5A623","Medium":"#4A90D9"}
        fig = px.bar(
            hub_sla_df.sort_values("sla_contribution_pct"),
            x="sla_contribution_pct", y="hub_id",
            orientation="h",
            color="severity",
            color_discrete_map=sev_color_map,
            text="sla_contribution_pct",
            labels={"sla_contribution_pct":"% of total SLA breaches","hub_id":"Hub"}
        )
        fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig.update_layout(
            paper_bgcolor="#0D1117", plot_bgcolor="#0D1117",
            font=dict(color="#8B949E"), height=380,
            legend=dict(bgcolor="#161B22",bordercolor="#30363D"),
            xaxis=dict(gridcolor="#21262D"),
            yaxis=dict(gridcolor="rgba(0,0,0,0)")
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("#### Concentration Summary")
        st.markdown("""
<div class='section-card'>
<p style='color:#E6EDF3;font-size:15px;font-weight:600;margin:0'>3 hubs = 50.5% of all breaches</p>
<p style='font-size:12px;margin:4px 0 12px'>IND000000ACB, IND562132AAA, IND421302AAG</p>
<p style='color:#E6EDF3;font-size:15px;font-weight:600;margin:0'>10 hubs = 80% of all breaches</p>
<p style='font-size:12px;margin:4px 0 12px'>Extreme Pareto skew</p>
<p style='color:#E8354A;font-size:13px;font-weight:600;margin:0'>⚠ Single Point of Failure</p>
<p style='font-size:12px;margin:4px 0'>Removing ACB → 15 isolated components</p>
</div>""", unsafe_allow_html=True)

        st.markdown("#### Revenue at Risk")
        st.metric("Monthly (top 3 hubs)", "₹41,11,909")
        st.metric("Recoverable (30% fix)", "₹12,33,573 / month")
        st.metric("Annual potential",      "₹1,48,02,873")

    st.markdown("---")
    st.markdown("#### Hub Detail Cards")
    for i in range(0, 5, 1):
        row = hub_sla_df.iloc[i]
        sev = row["severity"]
        bc  = "#E8354A" if sev=="Critical" else \
              "#F5A623" if sev=="High" else "#4A90D9"
        # corridors through this hub
        n_out = len(corridor_df[corridor_df["source_center"]==row["hub_id"]])
        n_in  = len(corridor_df[corridor_df["destination_center"]==row["hub_id"]])
        st.markdown(f"""
<div style='background:#161B22;border:1px solid {bc}44;border-left:3px solid {bc};
border-radius:8px;padding:14px 18px;margin-bottom:10px'>
<div style='display:flex;justify-content:space-between;align-items:center'>
  <div>
    <span style='color:#E6EDF3;font-size:15px;font-weight:700'>
    #{i+1} {row['hub_id']}</span>
    <span style='margin-left:10px;background:{bc}22;color:{bc};
    border:1px solid {bc}55;padding:2px 10px;border-radius:12px;font-size:11px'>
    {sev}</span>
  </div>
  <span style='color:{bc};font-size:22px;font-weight:700'>
  {row['sla_contribution_pct']}%</span>
</div>
<div style='display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:12px'>
  <div><span style='color:#8B949E;font-size:11px'>Delay ratio</span>
       <p style='color:#E6EDF3;font-size:16px;font-weight:600;margin:0'>
       {row['median_delay_ratio']:.2f}×</p></div>
  <div><span style='color:#8B949E;font-size:11px'>Inbound corridors</span>
       <p style='color:#E6EDF3;font-size:16px;font-weight:600;margin:0'>{n_in}</p></div>
  <div><span style='color:#8B949E;font-size:11px'>Outbound corridors</span>
       <p style='color:#E6EDF3;font-size:16px;font-weight:600;margin:0'>{n_out}</p></div>
  <div><span style='color:#8B949E;font-size:11px'>Recommended action</span>
       <p style='color:{bc};font-size:12px;font-weight:600;margin:0'>
       {row['intervention']}</p></div>
</div>
</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 3 — ETA MODEL
# ═══════════════════════════════════════════════════════════════
elif page == "📊 ETA Model":
    st.markdown("## 📊 ETA Prediction — Model Benchmark")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### MAE Comparison (lower = better)")
        colors = ["#4CAF50" if t=="Graph" else "#607D8B"
                  for t in model_df["Type"]]
        fig = go.Figure(go.Bar(
            x=model_df["Model"], y=model_df["MAE"],
            marker_color=colors,
            text=model_df["MAE"].apply(lambda x: f"{x:.2f}"),
            textposition="outside"
        ))
        fig.update_layout(
            paper_bgcolor="#0D1117", plot_bgcolor="#0D1117",
            font=dict(color="#8B949E"), height=340,
            yaxis=dict(gridcolor="#21262D", title="MAE (minutes)"),
            xaxis=dict(tickangle=-20)
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.markdown("#### Within-15% Accuracy (higher = better)")
        fig2 = go.Figure(go.Bar(
            x=model_df["Model"], y=model_df["Within15"],
            marker_color=colors,
            text=model_df["Within15"].apply(lambda x: f"{x:.1f}%"),
            textposition="outside"
        ))
        fig2.add_hline(y=8.8, line_dash="dot", line_color="#E8354A",
                       annotation_text="OSRM raw 8.8%",
                       annotation_font_color="#E8354A")
        fig2.add_hline(y=50, line_dash="dash", line_color="#F5A623",
                       annotation_text="50% production target",
                       annotation_font_color="#F5A623")
        fig2.update_layout(
            paper_bgcolor="#0D1117", plot_bgcolor="#0D1117",
            font=dict(color="#8B949E"), height=340,
            yaxis=dict(gridcolor="#21262D", title="% trips within 15% of actual"),
            xaxis=dict(tickangle=-20)
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown("#### Key Findings")
    f1,f2,f3,f4 = st.columns(4)
    f1.metric("Best Model MAE",      "10.55 min",   "-7.66 vs OSRM")
    f2.metric("Graph Advantage",     "3.8× better", "on Within-15%")
    f3.metric("Leakage removed",     "is_cutoff",   "83.6% delayed rate")
    f4.metric("Production status",   "Decision-support ready",
              "Not customer-SLA ready yet")

    st.markdown("---")
    st.markdown("#### MAPE by Segment Duration")
    mape_df = pd.DataFrame({
        "Bucket": ["0-10 min","10-30 min","30-60 min","60-120 min","120+ min"],
        "MAPE":   [110.2,     36.4,        20.5,        36.0,        47.4],
        "Status": ["Artifact","Unreliable","Acceptable","Unreliable","Unreliable"]
    })
    col_m = ["#E8354A" if m>40 else "#F5A623" if m>20 else "#4CAF50"
             for m in mape_df["MAPE"]]
    fig3 = go.Figure(go.Bar(
        x=mape_df["Bucket"], y=mape_df["MAPE"],
        marker_color=col_m,
        text=mape_df["MAPE"].apply(lambda x: f"{x:.1f}%"),
        textposition="outside"
    ))
    fig3.add_hline(y=20, line_dash="dash", line_color="#4CAF50",
                   annotation_text="20% operational target",
                   annotation_font_color="#4CAF50")
    fig3.update_layout(
        paper_bgcolor="#0D1117", plot_bgcolor="#0D1117",
        font=dict(color="#8B949E"), height=300,
        yaxis=dict(gridcolor="#21262D", title="MAPE (%)"),
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.caption("30-60 min segments are most reliable (20.5% MAPE). "
               "Apply model only on segments > 10 min for SLA decisions.")


# ═══════════════════════════════════════════════════════════════
# PAGE 4 — FTL VS CARTING
# ═══════════════════════════════════════════════════════════════
elif page == "🚛 FTL vs Carting":
    st.markdown("## 🚛 FTL vs Carting Decision Framework")

    c1,c2,c3 = st.columns(3)
    c1.metric("Delay Risk Classifier AUC", "0.840",
              "Genuine prediction task")
    c2.metric("Graph position signal",     "7%",
              "of route-type decisions")
    c3.metric("Corridor stats signal",     "55%",
              "primary driver")

    st.markdown("---")
    st.markdown("#### Time & Cost Comparison — Reliable Buckets (n ≥ 100)")
    for _, row in ftl_df.iterrows():
        if row["Carting_time"] is None:
            st.markdown(f"""
<div style='background:#161B22;border:1px solid #30363D;border-radius:8px;
padding:12px 16px;margin-bottom:8px'>
<span style='color:#E6EDF3;font-weight:600'>{row['Bucket']}</span>
<span style='margin-left:12px;color:#8B949E;font-size:12px'>
Insufficient Carting data (n&lt;100) — FTL recommended by default for reliability</span>
</div>""", unsafe_allow_html=True)
            continue
        winner = "Carting" if row["Recommendation"]=="Carting" else "FTL"
        wc     = "#4CAF50" if winner=="Carting" else "#4A90D9"
        st.markdown(f"""
<div style='background:#161B22;border:1px solid #30363D;border-left:3px solid {wc};
border-radius:8px;padding:14px 18px;margin-bottom:10px'>
<div style='display:flex;justify-content:space-between;align-items:center'>
  <span style='color:#E6EDF3;font-size:15px;font-weight:700'>{row['Bucket']}</span>
  <span style='background:{wc}22;color:{wc};border:1px solid {wc}55;
  padding:4px 14px;border-radius:12px;font-size:12px;font-weight:700'>
  ✓ USE {winner.upper()}</span>
</div>
<div style='display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-top:12px'>
  <div><span style='color:#8B949E;font-size:11px'>Carting time</span>
       <p style='color:#E6EDF3;font-size:18px;font-weight:600;margin:0'>
       {row['Carting_time']} min</p></div>
  <div><span style='color:#8B949E;font-size:11px'>FTL time</span>
       <p style='color:#E6EDF3;font-size:18px;font-weight:600;margin:0'>
       {row['FTL_time']} min</p></div>
  <div><span style='color:#8B949E;font-size:11px'>FTL extra cost</span>
       <p style='color:#E8354A;font-size:18px;font-weight:600;margin:0'>
       ₹{row['FTL_extra_cost']}/shipment</p></div>
  <div><span style='color:#8B949E;font-size:11px'>SLA savings (FTL)</span>
       <p style='color:#4CAF50;font-size:18px;font-weight:600;margin:0'>
       ₹{row['SLA_savings']}/100 trips</p></div>
</div>
</div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### Operational Rules")
    rules = [
        ("Hyper-local (<50km)",    "Carting preferred",
         "Faster (17 vs 33 min) AND cheaper (₹152 saving/shipment)",   "#4CAF50"),
        ("Short (50-150km)",       "Carting preferred",
         "Cost savings outweigh small SLA benefit of FTL",             "#4CAF50"),
        ("Long (>150km)",          "FTL by default",
         "Reliability justifies cost — Carting variance too high",     "#4A90D9"),
        ("High-betweenness hubs",  "Always FTL",
         "IND000000ACB & IND562132AAA amplify Carting delays via dock congestion","#E8354A"),
    ]
    for bucket, rec, reason, col in rules:
        st.markdown(f"""
<div style='display:flex;gap:14px;align-items:flex-start;padding:10px 0;
border-bottom:1px solid #21262D'>
<div style='min-width:180px;color:#E6EDF3;font-size:13px;font-weight:600'>
{bucket}</div>
<div style='min-width:130px'>
<span style='background:{col}22;color:{col};border:1px solid {col}55;
padding:2px 10px;border-radius:10px;font-size:11px;font-weight:600'>{rec}</span>
</div>
<div style='color:#8B949E;font-size:12px'>{reason}</div>
</div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 5 — CORRIDOR AUDIT
# ═══════════════════════════════════════════════════════════════
elif page == "📋 Corridor Audit":
    st.markdown("## 📋 Corridor Audit Table")

    # Filters
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        int_filter = st.multiselect(
            "Intervention type",
            options=corridor_df["intervention"].unique().tolist(),
            default=corridor_df["intervention"].unique().tolist()
        )
    with fc2:
        chronic_filter = st.selectbox(
            "Chronic corridors", ["All","Chronic only","Non-chronic only"])
    with fc3:
        delay_min = st.slider(
            "Min delay ratio", 1.0,
            float(corridor_df["median_delay_ratio"].max()), 1.0, 0.05)

    filtered = corridor_df[
        corridor_df["intervention"].isin(int_filter) &
        (corridor_df["median_delay_ratio"] >= delay_min)
    ]
    if chronic_filter == "Chronic only":
        filtered = filtered[filtered["is_chronic"]==1]
    elif chronic_filter == "Non-chronic only":
        filtered = filtered[filtered["is_chronic"]==0]

    st.markdown(f"**{len(filtered):,} corridors** match filters")

    # Summary metrics
    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Shown corridors",   f"{len(filtered):,}")
    m2.metric("Chronic",           f"{int(filtered['is_chronic'].sum()):,}")
    m3.metric("Avg delay ratio",   f"{filtered['median_delay_ratio'].mean():.2f}×")
    m4.metric("Total SLA contrib", f"{filtered['sla_contribution_pct'].sum():.1f}%")

    # Intervention donut
    ic1, ic2 = st.columns([1, 2])
    with ic1:
        st.markdown("#### Intervention Breakdown")
        icount = filtered["intervention"].value_counts().reset_index()
        icount.columns = ["intervention","count"]
        fig_d = px.pie(
            icount, values="count", names="intervention",
            hole=0.6,
            color_discrete_sequence=["#E8354A","#F5A623","#7B68EE","#4CAF50","#607D8B"]
        )
        fig_d.update_layout(
            paper_bgcolor="#0D1117", font=dict(color="#8B949E"),
            height=260, showlegend=True,
            legend=dict(bgcolor="#161B22", bordercolor="#30363D",
                        font=dict(size=10))
        )
        st.plotly_chart(fig_d, use_container_width=True)

    with ic2:
        st.markdown("#### Top Corridors by SLA Contribution")
        display_cols = ["source_center","destination_center",
                        "median_delay_ratio","sla_contribution_pct",
                        "trip_count","is_chronic","intervention"]
        st.dataframe(
            filtered[display_cols]
            .sort_values("sla_contribution_pct", ascending=False)
            .head(20)
            .rename(columns={
                "source_center":"Source","destination_center":"Destination",
                "median_delay_ratio":"Delay Ratio","sla_contribution_pct":"SLA %",
                "trip_count":"Trips","is_chronic":"Chronic","intervention":"Action"
            }),
            use_container_width=True, height=240,
            hide_index=True
        )

    st.download_button(
        "⬇ Download filtered corridors as CSV",
        data=filtered.to_csv(index=False),
        file_name="filtered_corridors.csv",
        mime="text/csv"
    )
