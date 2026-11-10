import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from handover_engine import SatelliteHandoverEngine


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Satellite Handover AI",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM SPACE-THEMED UI
# ============================================================

st.markdown("""
<style>

    .stApp {
        background:
            radial-gradient(circle at 20% 20%, rgba(30, 60, 120, 0.25), transparent 35%),
            radial-gradient(circle at 80% 10%, rgba(80, 30, 130, 0.18), transparent 30%),
            linear-gradient(135deg, #050816 0%, #0b1026 50%, #080b18 100%);
        color: #EAF2FF;
    }

    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: #F4F8FF;
        margin-bottom: 0;
    }

    .subtitle {
        color: #9FB3D9;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    .metric-card {
        background: rgba(16, 25, 52, 0.82);
        border: 1px solid rgba(100, 150, 255, 0.25);
        border-radius: 16px;
        padding: 18px;
        box-shadow: 0 0 20px rgba(40, 100, 255, 0.08);
    }

    .decision-card {
        background: linear-gradient(
            135deg,
            rgba(22, 42, 85, 0.9),
            rgba(15, 24, 55, 0.9)
        );
        border: 1px solid rgba(80, 160, 255, 0.35);
        border-radius: 18px;
        padding: 22px;
        margin-bottom: 15px;
    }

    .section-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #DCE8FF;
        margin-top: 15px;
        margin-bottom: 10px;
    }

    div[data-testid="stMetric"] {
        background: rgba(16, 25, 52, 0.75);
        border: 1px solid rgba(100, 150, 255, 0.2);
        padding: 15px;
        border-radius: 14px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# INITIALIZE SESSION STATE
# ============================================================

if "engine" not in st.session_state:
    st.session_state.engine = SatelliteHandoverEngine()

if "running" not in st.session_state:
    st.session_state.running = False

if "latest_result" not in st.session_state:
    st.session_state.latest_result = None


engine = st.session_state.engine


# ============================================================
# SIDEBAR CONTROLS
# ============================================================

with st.sidebar:

    st.title("🛰️ Simulation Control")

    st.caption("AI-Based LEO Satellite Network Simulation")

    st.divider()

    advance = st.button(
        "▶ Advance Simulation",
        use_container_width=True
    )

    auto_steps = st.slider(
        "Steps to simulate",
        min_value=1,
        max_value=20,
        value=1
    )

    if st.button(
        "⚡ Run Multiple Steps",
        use_container_width=True
    ):

        for _ in range(auto_steps):
            st.session_state.latest_result = engine.step()

    if st.button(
        "🔄 Reset Simulation",
        use_container_width=True
    ):

        engine.reset()

        st.session_state.latest_result = None

        st.rerun()


# ============================================================
# ADVANCE ONE STEP
# ============================================================

if advance:

    st.session_state.latest_result = engine.step()


# ============================================================
# INITIAL STATE
# ============================================================

if st.session_state.latest_result is None:

    st.session_state.latest_result = engine.step()


result = st.session_state.latest_result

satellites = result["satellites"]

history = result["history"]

events = result["events"]


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🛰️ AI Satellite Handover Control Center</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'LEO Network Simulation • Machine Learning Assisted Handover Decisions'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# NETWORK OVERVIEW
# ============================================================

st.markdown(
    '<div class="section-title">📡 Network Overview</div>',
    unsafe_allow_html=True
)

visible_count = int(satellites["visible"].sum())

actual_handovers = max(0, engine.handover_count - 1)

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "Total Satellites",
    len(satellites)
)

c2.metric(
    "Visible Satellites",
    visible_count
)

c3.metric(
    "Current Connection",
    result["current_satellite"]
)

c4.metric(
    "AI Recommendation",
    result["recommended_satellite"]
)

c5.metric(
    "Satellite Handovers",
    actual_handovers
)


# ============================================================
# AI DECISION PANEL
# ============================================================

st.markdown(
    '<div class="section-title">🤖 AI Decision Engine</div>',
    unsafe_allow_html=True
)

decision_col, status_col = st.columns([2, 1])

with decision_col:

    st.markdown(
        f"""
        <div class="decision-card">
            <h3>Recommended Satellite: 🛰️ {result["recommended_satellite"]}</h3>
            <p><b>Current Connection:</b> {result["current_satellite"]}</p>
            <p><b>Decision:</b> {result["reason"]}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


with status_col:

    if result["handover_triggered"]:

        st.success("⚡ HANDOVER TRIGGERED")

    else:

        st.info("🛡️ CONNECTION RETAINED")


# ============================================================
# SATELLITE NETWORK MAP
# ============================================================

st.markdown(
    '<div class="section-title">🌍 Satellite Network View</div>',
    unsafe_allow_html=True
)

map_data = satellites.copy()

map_data["x"] = range(len(map_data))

map_data["y"] = map_data["distance_km"]


fig_network = go.Figure()

fig_network.add_trace(
    go.Scatter(
        x=map_data["x"],
        y=map_data["y"],
        mode="markers+text",
        text=map_data["satellite_id"],
        textposition="top center",
        marker=dict(
            size=18,
            color=map_data["ai_score"],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(
                title="AI Score"
            )
        ),
        hovertemplate=
            "<b>%{text}</b><br>"
            "Distance: %{y:.1f} km<br>"
            "<extra></extra>"
    )
)

fig_network.update_layout(
    height=420,
    template="plotly_dark",
    title="Simulated Satellite Positions by Distance",
    xaxis_title="Satellite Index",
    yaxis_title="Distance (km)",
    margin=dict(l=20, r=20, t=60, b=20)
)

st.plotly_chart(
    fig_network,
    use_container_width=True
)


# ============================================================
# SATELLITE STATUS
# ============================================================

st.markdown(
    '<div class="section-title">📡 Live Satellite Status</div>',
    unsafe_allow_html=True
)

display_columns = [
    "satellite_id",
    "signal_dbm",
    "latency_ms",
    "bandwidth_mbps",
    "congestion_pct",
    "distance_km",
    "visible",
    "eligible",
    "ai_score"
]

display_df = satellites[
    display_columns
].copy()

display_df["ai_score"] = (
    display_df["ai_score"] * 100
).round(2)

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# VISUALIZATION ROW 1
# ============================================================

st.markdown(
    '<div class="section-title">📊 Satellite Performance Analysis</div>',
    unsafe_allow_html=True
)

chart1, chart2 = st.columns(2)


with chart1:

    fig_signal = px.bar(
        satellites,
        x="satellite_id",
        y="signal_dbm",
        color="eligible",
        title="Signal Strength by Satellite",
        labels={
            "signal_dbm": "Signal Strength (dBm)",
            "satellite_id": "Satellite"
        },
        template="plotly_dark"
    )

    fig_signal.update_layout(
        height=380
    )

    st.plotly_chart(
        fig_signal,
        use_container_width=True
    )


with chart2:

    fig_latency = px.bar(
        satellites,
        x="satellite_id",
        y="latency_ms",
        color="eligible",
        title="Latency Comparison",
        labels={
            "latency_ms": "Latency (ms)",
            "satellite_id": "Satellite"
        },
        template="plotly_dark"
    )

    fig_latency.update_layout(
        height=380
    )

    st.plotly_chart(
        fig_latency,
        use_container_width=True
    )


# ============================================================
# VISUALIZATION ROW 2
# ============================================================

chart3, chart4 = st.columns(2)


with chart3:

    fig_ai = px.bar(
        satellites.sort_values(
            "ai_score",
            ascending=False
        ),
        x="satellite_id",
        y="ai_score",
        title="AI Candidate Suitability Ranking",
        labels={
            "ai_score": "Predicted Suitability",
            "satellite_id": "Satellite"
        },
        template="plotly_dark"
    )

    fig_ai.update_layout(
        height=380
    )

    st.plotly_chart(
        fig_ai,
        use_container_width=True
    )


with chart4:

    fig_congestion = px.scatter(
        satellites,
        x="congestion_pct",
        y="bandwidth_mbps",
        size="ai_score",
        color="satellite_id",
        title="Congestion vs Available Bandwidth",
        labels={
            "congestion_pct": "Congestion (%)",
            "bandwidth_mbps": "Bandwidth (Mbps)"
        },
        template="plotly_dark"
    )

    fig_congestion.update_layout(
        height=380
    )

    st.plotly_chart(
        fig_congestion,
        use_container_width=True
    )


# ============================================================
# HISTORICAL TRENDS
# ============================================================

st.markdown(
    '<div class="section-title">📈 Connection Performance Over Time</div>',
    unsafe_allow_html=True
)

if len(history) > 1:

    fig_history = px.line(
        history,
        x="step",
        y="serving_latency_ms",
        markers=True,
        title="Serving Satellite Latency Trend",
        labels={
            "serving_latency_ms": "Latency (ms)",
            "step": "Simulation Step"
        },
        template="plotly_dark"
    )

    fig_history.update_layout(
        height=400
    )

    st.plotly_chart(
        fig_history,
        use_container_width=True
    )

else:

    st.info(
        "Advance the simulation to generate historical performance trends."
    )


# ============================================================
# HANDOVER HISTORY
# ============================================================

st.markdown(
    '<div class="section-title">📜 Handover Event History</div>',
    unsafe_allow_html=True
)

if not events.empty:

    st.dataframe(
        events,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "No handover events have been recorded yet."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Simulation-based academic prototype • "
    "Synthetic satellite network conditions • "
    "Not real-time satellite telemetry"
)