import streamlit as st
from textwrap import dedent
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Coal Transport AI",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "coal_transport_delay_model.pkl"
DATA_PATH = BASE_DIR / "data" / "raw" / "coal_transport_data_10000.csv"
RISK_DATA_PATH = (
    BASE_DIR / "data" / "processed" /
    "coal_transport_business_risk.csv"
)


# ============================================================
# CUSTOM CSS — PROFESSIONAL COMMAND CENTER UI
# ============================================================
st.markdown(dedent("""
<style>
:root {
    --bg: #070b12;
    --panel: rgba(17, 24, 39, 0.78);
    --panel-2: rgba(30, 41, 59, 0.72);
    --border: rgba(148, 163, 184, 0.16);
    --text: #f8fafc;
    --muted: #94a3b8;
    --accent: #f59e0b;
    --accent-2: #fbbf24;
    --danger: #ef4444;
    --success: #22c55e;
}

.stApp {
    background:
        radial-gradient(circle at 75% 0%, rgba(245,158,11,.10), transparent 28%),
        radial-gradient(circle at 15% 20%, rgba(14,165,233,.07), transparent 25%),
        linear-gradient(135deg, #070b12 0%, #0b1220 48%, #070b12 100%);
    color: var(--text);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b0f18 0%, #101722 100%);
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] > div:first-child { padding-top: 1rem; }

.block-container { max-width: 1500px; padding-top: 1.2rem; padding-bottom: 3rem; }

h1 { font-size: 2.55rem !important; letter-spacing: -0.04em; font-weight: 800 !important; }
h2 { font-weight: 750 !important; letter-spacing: -0.02em; }
h3 { font-weight: 700 !important; }

/* Navigation */
[data-testid="stSidebarNav"] { padding-top: 0; }
[data-testid="stSidebar"] .stRadio label {
    border-radius: 10px;
    padding: 7px 10px;
    transition: all .2s ease;
}

/* Hero */
.hero {
    position: relative;
    overflow: hidden;
    min-height: 210px;
    padding: 30px 34px;
    border-radius: 24px;
    border: 1px solid rgba(245,158,11,.22);
    background:
      radial-gradient(circle at 82% 35%, rgba(245,158,11,.22), transparent 24%),
      linear-gradient(135deg, rgba(17,24,39,.96), rgba(15,23,42,.78));
    box-shadow: 0 25px 80px rgba(0,0,0,.32);
    margin-bottom: 22px;
}
.hero:after {
    content: ''; position:absolute; width:260px; height:260px; right:-90px; top:-110px;
    border: 1px solid rgba(251,191,36,.18); border-radius:50%;
    box-shadow: 0 0 0 35px rgba(251,191,36,.03), 0 0 0 70px rgba(251,191,36,.02);
}
.hero-kicker { color:#fbbf24; font-size:.78rem; font-weight:800; letter-spacing:.16em; text-transform:uppercase; }
.hero-title { font-size:2.35rem; font-weight:850; margin-top:6px; }
.hero-sub { color:#a8b3c7; max-width:700px; line-height:1.55; margin-top:8px; }
.hero-badge { display:inline-flex; align-items:center; gap:8px; padding:7px 11px; border-radius:999px; background:rgba(34,197,94,.10); border:1px solid rgba(34,197,94,.28); color:#86efac; font-size:.78rem; font-weight:700; margin-top:18px; }

/* Cards */
.card, .metric-card, .model-card {
    background: linear-gradient(145deg, rgba(17,24,39,.92), rgba(15,23,42,.72));
    border: 1px solid var(--border);
    border-radius: 18px;
    box-shadow: 0 14px 40px rgba(0,0,0,.18);
}
.metric-card { padding: 18px 20px; min-height: 105px; }
.metric-title { color:#8fa0b7; font-size:.72rem; font-weight:800; letter-spacing:.11em; text-transform:uppercase; }
.metric-value { font-size:1.72rem; font-weight:800; margin-top:7px; }
.metric-delta { color:#64748b; font-size:.76rem; margin-top:4px; }

.model-card { padding: 22px; height: 100%; }
.model-card-title { color:#94a3b8; text-transform:uppercase; letter-spacing:.12em; font-size:.72rem; font-weight:800; }
.model-card-value { font-size:2rem; font-weight:850; margin-top:8px; }
.model-card-sub { color:#8b9ab0; font-size:.8rem; margin-top:4px; }

.status-dot { display:inline-block; width:9px; height:9px; border-radius:50%; background:#22c55e; box-shadow:0 0 14px rgba(34,197,94,.8); margin-right:7px; }

.feature-pill {
    display:inline-block; padding:7px 10px; margin:4px 4px 0 0; border-radius:9px;
    background:rgba(51,65,85,.35); border:1px solid rgba(148,163,184,.13);
    color:#cbd5e1; font-size:.76rem;
}

.arch-node {
    padding:14px 10px; text-align:center; border-radius:14px;
    background:rgba(30,41,59,.6); border:1px solid rgba(148,163,184,.14);
}
.arch-arrow { text-align:center; color:#f59e0b; font-size:1.4rem; padding-top:9px; }

.insight { padding:16px 18px; border-radius:15px; background:rgba(15,23,42,.72); border:1px solid rgba(148,163,184,.13); }
.insight strong { color:#f8fafc; }

[data-testid="stMetric"] { background: transparent; }
[data-testid="stButton"] button { border-radius:11px; font-weight:750; }

/* Plotly containers */
[data-testid="stPlotlyChart"] { border-radius:16px; overflow:hidden; }

/* Tables */
[data-testid="stDataFrame"] { border-radius:14px; overflow:hidden; border:1px solid var(--border); }

/* Hide Streamlit chrome */
#MainMenu { visibility:hidden; }
footer { visibility:hidden; }

/* ===== V2 PREMIUM MODEL CENTER ===== */
.model-hero-v2 {
    position: relative;
    overflow: hidden;
    min-height: 330px;
    border-radius: 28px;
    border: 1px solid rgba(245,158,11,.25);
    background:
        linear-gradient(90deg, rgba(5,9,15,.98) 0%, rgba(5,9,15,.88) 42%, rgba(5,9,15,.30) 100%),
        url("https://images.unsplash.com/photo-1509391366360-2e959784a276?auto=format&fit=crop&w=1800&q=85") center/cover;
    box-shadow: 0 30px 90px rgba(0,0,0,.42);
    margin-bottom: 24px;
}
.model-hero-content {
    position: relative;
    z-index: 2;
    padding: 42px;
    max-width: 760px;
}
.model-hero-eyebrow {
    color: #fbbf24;
    font-size: .72rem;
    font-weight: 900;
    letter-spacing: .22em;
    text-transform: uppercase;
}
.model-hero-title-v2 {
    font-size: 3rem;
    line-height: 1.03;
    font-weight: 900;
    letter-spacing: -.055em;
    margin-top: 10px;
}
.model-hero-copy {
    color: #b7c2d3;
    line-height: 1.65;
    margin-top: 14px;
    max-width: 650px;
}
.live-chip {
    display:inline-flex;
    align-items:center;
    gap:8px;
    margin-top:20px;
    padding:8px 13px;
    border-radius:999px;
    color:#bbf7d0;
    background:rgba(34,197,94,.10);
    border:1px solid rgba(34,197,94,.28);
    font-size:.76rem;
    font-weight:800;
}
.live-chip i {
    width:8px;height:8px;border-radius:50%;background:#22c55e;
    box-shadow:0 0 16px rgba(34,197,94,.9);
}
.v2-card {
    border-radius: 20px;
    border: 1px solid rgba(148,163,184,.13);
    background: linear-gradient(145deg, rgba(17,24,39,.94), rgba(9,14,24,.78));
    box-shadow: 0 18px 55px rgba(0,0,0,.20);
    padding: 22px;
}
.v2-label {
    color:#8190a6;
    font-size:.68rem;
    font-weight:900;
    letter-spacing:.14em;
    text-transform:uppercase;
}
.v2-big {
    font-size:2rem;
    font-weight:900;
    margin-top:7px;
}
.v2-muted { color:#8795aa; font-size:.78rem; margin-top:4px; }
.v2-good { color:#86efac; }
.v2-warn { color:#fcd34d; }
.v2-danger { color:#fca5a5; }
.signal {
    display:flex; justify-content:space-between; align-items:center;
    padding:11px 0; border-bottom:1px solid rgba(148,163,184,.09);
}
.signal:last-child { border-bottom:0; }
.signal-name { color:#cbd5e1; font-size:.82rem; }
.signal-value { color:#f8fafc; font-weight:800; font-size:.82rem; }
.pipeline {
    display:flex; align-items:stretch; gap:10px; margin-top:12px;
}
.pipeline-node {
    flex:1; padding:18px 12px; text-align:center; border-radius:16px;
    background:rgba(30,41,59,.55);
    border:1px solid rgba(148,163,184,.12);
}
.pipeline-node b { display:block; font-size:.9rem; }
.pipeline-node span { display:block; color:#8391a6; font-size:.7rem; margin-top:5px; }
.pipeline-arrow { display:flex; align-items:center; color:#f59e0b; font-size:1.3rem; }
.feature-grid {
    display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:9px; margin-top:12px;
}
.feature-box {
    padding:11px 12px; border-radius:12px;
    background:rgba(30,41,59,.42); border:1px solid rgba(148,163,184,.10);
}
.feature-box b { font-size:.76rem; color:#e2e8f0; }
.feature-box span { display:block; color:#718096; font-size:.67rem; margin-top:3px; }
@media (max-width: 900px) {
    .model-hero-title-v2 { font-size:2.2rem; }
    .model-hero-content { padding:28px; }
    .feature-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
}

</style>
"""), unsafe_allow_html=True)

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_risk_data():

    if RISK_DATA_PATH.exists():
        return pd.read_csv(RISK_DATA_PATH)

    return None


model = load_model()
df = load_data()
risk_df = load_risk_data()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def fuel_efficiency(distance, fuel):

    if fuel <= 0:
        return 0

    return distance / fuel


def get_risk_level(probability):

    if probability >= 0.80:
        return "CRITICAL"

    if probability >= 0.60:
        return "HIGH"

    if probability >= 0.40:
        return "MEDIUM"

    return "LOW"


def get_risk_emoji(level):

    return {
        "CRITICAL": "🔴",
        "HIGH": "🟠",
        "MEDIUM": "🟡",
        "LOW": "🟢"
    }.get(level, "⚪")


def recommendation(
    probability,
    weather,
    traffic,
    speed,
    previous_delays
):

    if probability >= 0.80:
        return "Immediate operational intervention"

    if weather == "Heavy_Rain":
        return "Heavy-rain monitoring and route assessment"

    if traffic == "High":
        return "Monitor traffic and consider alternate route"

    if speed < 30:
        return "Review vehicle speed and route conditions"

    if previous_delays >= 5:
        return "Review previous delay history before dispatch"

    if probability >= 0.60:
        return "Enhanced trip monitoring"

    if probability >= 0.40:
        return "Routine monitoring"

    return "Normal operation"


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🚛 Coal Transport AI")

st.sidebar.caption(
    "Intelligent Delay & Risk Monitoring System"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Control Center",
        "🔮 Trip Prediction",
        "🚨 Risk Monitor",
        "📊 Analytics",
        "🧠 Explainability",
        "ℹ️ Model"
    ]
)

st.sidebar.markdown("---")

st.sidebar.success("Model: ONLINE")

st.sidebar.metric(
    "Model ROC-AUC",
    "72.95%"
)


# ============================================================
# CONTROL CENTER
# ============================================================

if page == "🏠 Control Center":

    st.title("🚛 Coal Transport AI")
    st.caption(
        "AI-powered operational control center for coal transportation"
    )

    # --------------------------------------------------------
    # KPIs
    # --------------------------------------------------------

    total_trips = len(df)

    delay_count = int(df["Delay"].sum())

    delay_rate = df["Delay"].mean() * 100

    avg_distance = df["Distance_km"].mean()

    avg_speed = df["Average_Speed_kmh"].mean()

    cols = st.columns(5)

    metrics = [
        ("TOTAL TRIPS", f"{total_trips:,}"),
        ("DELAYED TRIPS", f"{delay_count:,}"),
        ("DELAY RATE", f"{delay_rate:.2f}%"),
        ("AVG DISTANCE", f"{avg_distance:.1f} km"),
        ("AVG SPEED", f"{avg_speed:.1f} km/h")
    ]

    for col, (title, value) in zip(cols, metrics):

        with col:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">{title}</div>
                    <div class="metric-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.markdown("---")

    # --------------------------------------------------------
    # CHARTS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🌦️ Delay Rate by Weather")

        weather_data = (
            df.groupby("Weather")["Delay"]
            .mean()
            .mul(100)
            .reset_index()
        )

        fig = px.bar(
            weather_data,
            x="Weather",
            y="Delay",
            text_auto=".1f",
            labels={"Delay": "Delay Rate (%)"}
        )

        fig.update_layout(
            template="plotly_dark",
            height=350
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        st.subheader("🚦 Delay Rate by Traffic")

        traffic_data = (
            df.groupby("Traffic_Level")["Delay"]
            .mean()
            .mul(100)
            .reset_index()
        )

        fig = px.bar(
            traffic_data,
            x="Traffic_Level",
            y="Delay",
            text_auto=".1f",
            labels={"Delay": "Delay Rate (%)"}
        )

        fig.update_layout(
            template="plotly_dark",
            height=350
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # --------------------------------------------------------
    # SPEED / DISTANCE
    # --------------------------------------------------------

    st.subheader("🛣️ Distance vs Average Speed")

    sample = df.sample(
        min(1500, len(df)),
        random_state=42
    )

    fig = px.scatter(
        sample,
        x="Distance_km",
        y="Average_Speed_kmh",
        color="Delay",
        hover_data=[
            "Weather",
            "Traffic_Level",
            "Previous_Delays"
        ],
        labels={
            "Delay": "Delay"
        }
    )

    fig.update_layout(
        template="plotly_dark",
        height=450
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # INSIGHTS
    # --------------------------------------------------------

    st.subheader("🧠 Operational Insights")

    weather_risk = (
        df.groupby("Weather")["Delay"]
        .mean()
        .sort_values(ascending=False)
    )

    traffic_risk = (
        df.groupby("Traffic_Level")["Delay"]
        .mean()
        .sort_values(ascending=False)
    )

    c1, c2 = st.columns(2)

    with c1:

        st.info(
            f"Highest weather delay rate: "
            f"**{weather_risk.index[0]} "
            f"({weather_risk.iloc[0] * 100:.1f}%)**"
        )

    with c2:

        st.warning(
            f"Highest traffic delay rate: "
            f"**{traffic_risk.index[0]} "
            f"({traffic_risk.iloc[0] * 100:.1f}%)**"
        )


# ============================================================
# TRIP PREDICTION
# ============================================================

elif page == "🔮 Trip Prediction":

    st.title("🔮 AI Trip Risk Prediction")

    st.write(
        "Enter the planned trip parameters."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        distance = st.number_input(
            "Distance (km)",
            1.0,
            500.0,
            140.0
        )

        coal_weight = st.number_input(
            "Coal Weight (Ton)",
            1.0,
            100.0,
            30.0
        )

        fuel = st.number_input(
            "Fuel Used (Liters)",
            1.0,
            200.0,
            30.0
        )

        speed = st.number_input(
            "Average Speed (km/h)",
            1.0,
            120.0,
            25.0
        )

    with col2:

        loading = st.number_input(
            "Loading Time (Min)",
            1.0,
            180.0,
            45.0
        )

        unloading = st.number_input(
            "Unloading Time (Min)",
            1.0,
            180.0,
            30.0
        )

        weather = st.selectbox(
            "Weather",
            [
                "Clear",
                "Cloudy",
                "Rain",
                "Heavy_Rain"
            ]
        )

        traffic = st.selectbox(
            "Traffic Level",
            [
                "Low",
                "Medium",
                "High"
            ]
        )

    with col3:

        experience = st.number_input(
            "Driver Experience (Years)",
            0.0,
            40.0,
            5.0
        )

        previous_delays = st.number_input(
            "Previous Delays",
            0,
            20,
            5
        )

        safety = st.slider(
            "Safety Score",
            0.0,
            100.0,
            70.0
        )

        efficiency = fuel_efficiency(
            distance,
            fuel
        )

        st.metric(
            "Fuel Efficiency",
            f"{efficiency:.2f} km/L"
        )

    st.markdown("---")

    if st.button(
        "🚀 ANALYZE TRIP RISK",
        type="primary",
        use_container_width=True
    ):

        input_data = pd.DataFrame([{

            "Distance_km": distance,

            "Coal_Weight_Ton": coal_weight,

            "Fuel_Used_Liters": fuel,

            "Average_Speed_kmh": speed,

            "Loading_Time_Min": loading,

            "Unloading_Time_Min": unloading,

            "Weather": weather,

            "Traffic_Level": traffic,

            "Driver_Experience_Years": experience,

            "Previous_Delays": previous_delays,

            "Safety_Score": safety,

            "Fuel_Efficiency_km_per_liter":
                efficiency

        }])

        probability = model.predict_proba(
            input_data
        )[0][1]

        prediction = probability >= 0.50

        risk = get_risk_level(probability)

        action = recommendation(
            probability,
            weather,
            traffic,
            speed,
            previous_delays
        )

        st.markdown("---")

        st.subheader("🎯 Prediction Result")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Delay Probability",
                f"{probability * 100:.2f}%"
            )

        with c2:
            st.metric(
                "Prediction",
                "DELAY" if prediction else "NO DELAY"
            )

        with c3:
            st.metric(
                "Risk Level",
                f"{get_risk_emoji(risk)} {risk}"
            )

        st.progress(
            float(probability)
        )

        st.markdown("---")

        st.subheader("⚠️ Risk Factors")

        factors = []

        if weather == "Heavy_Rain":
            factors.append("Heavy rain")

        if traffic == "High":
            factors.append("High traffic")

        if speed < 30:
            factors.append("Low average speed")

        if previous_delays >= 5:
            factors.append("High previous delays")

        if distance >= 100:
            factors.append("Long transport distance")

        if loading >= 45:
            factors.append("Long loading time")

        if factors:

            for factor in factors:
                st.write(f"• {factor}")

        else:

            st.success(
                "No major operational risk factor detected."
            )

        st.markdown("---")

        st.subheader("💡 Recommended Action")

        if risk == "CRITICAL":
            st.error(action)

        elif risk == "HIGH":
            st.warning(action)

        elif risk == "MEDIUM":
            st.info(action)

        else:
            st.success(action)


# ============================================================
# RISK MONITOR
# ============================================================

elif page == "🚨 Risk Monitor":

    st.title("🚨 Transport Risk Monitor")

    if risk_df is None:

        st.warning(
            "Business risk dataset not found."
        )

    else:

        risk_counts = (
            risk_df["Risk_Level"]
            .value_counts()
            .reindex(
                ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
                fill_value=0
            )
        )

        cols = st.columns(4)

        for col, level in zip(
            cols,
            ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        ):

            with col:

                st.metric(
                    f"{get_risk_emoji(level)} {level}",
                    int(risk_counts[level])
                )

        st.markdown("---")

        st.subheader("Risk Distribution")

        fig = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            hole=0.45
        )

        fig.update_layout(
            template="plotly_dark"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader("🔴 Highest Risk Trips")

        risk_levels = st.multiselect(
            "Risk Level",
            ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
            default=["HIGH", "CRITICAL"]
        )

        filtered = risk_df[
            risk_df["Risk_Level"].isin(risk_levels)
        ].copy()

        sort_column = (
            "Delay_Probability"
            if "Delay_Probability" in filtered.columns
            else filtered.columns[0]
        )

        filtered = filtered.sort_values(
            sort_column,
            ascending=False
        )

        st.dataframe(
            filtered.head(100),
            use_container_width=True,
            height=500
        )

        csv = filtered.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "📥 Download Risk Report",
            csv,
            "coal_transport_risk_report.csv",
            "text/csv"
        )


# ============================================================
# ANALYTICS
# ============================================================

elif page == "📊 Analytics":

    st.title("📊 Transport Analytics")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Weather Performance")

        weather = (
            df.groupby("Weather")
            .agg(
                Trips=("Delay", "count"),
                Delay_Rate=("Delay", "mean"),
                Avg_Speed=("Average_Speed_kmh", "mean")
            )
            .reset_index()
        )

        weather["Delay_Rate"] *= 100

        st.dataframe(
            weather,
            use_container_width=True
        )

    with col2:

        st.subheader("Traffic Performance")

        traffic = (
            df.groupby("Traffic_Level")
            .agg(
                Trips=("Delay", "count"),
                Delay_Rate=("Delay", "mean"),
                Avg_Speed=("Average_Speed_kmh", "mean")
            )
            .reset_index()
        )

        traffic["Delay_Rate"] *= 100

        st.dataframe(
            traffic,
            use_container_width=True
        )

    st.markdown("---")

    st.subheader("Previous Delays vs Current Delay")

    previous = (
        df.groupby("Previous_Delays")["Delay"]
        .mean()
        .mul(100)
        .reset_index()
    )

    fig = px.line(
        previous,
        x="Previous_Delays",
        y="Delay",
        markers=True,
        labels={
            "Delay": "Delay Rate (%)",
            "Previous_Delays": "Previous Delays"
        }
    )

    fig.update_layout(
        template="plotly_dark"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ============================================================
# EXPLAINABILITY
# ============================================================

elif page == "🧠 Explainability":

    st.title("🧠 Model Explainability")

    st.write(
        "Feature importance from the trained Gradient Boosting model."
    )

    importance_path = (
        BASE_DIR /
        "data" /
        "processed" /
        "feature_importance.csv"
    )

    if importance_path.exists():

        importance = pd.read_csv(
            importance_path
        )

        st.dataframe(
            importance,
            use_container_width=True
        )

        feature_col = importance.columns[0]
        importance_col = importance.columns[1]

        importance = importance.sort_values(
            importance_col,
            ascending=True
        ).tail(15)

        fig = px.bar(
            importance,
            x=importance_col,
            y=feature_col,
            orientation="h",
            labels={
                importance_col: "Importance"
            }
        )

        fig.update_layout(
            template="plotly_dark",
            height=600
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.warning(
            "Feature importance file not found."
        )


# ============================================================
# ============================================================
# MODEL INFORMATION — V2 PREMIUM AI MODEL CENTER
# ============================================================

elif page == "ℹ️ Model":

    # --------------------------------------------------------
    # HERO
    # --------------------------------------------------------
    st.markdown(dedent("""
    <div class="model-hero-v2">
        <div class="model-hero-content">
            <div class="model-hero-eyebrow">COAL TRANSPORT AI · MODEL CENTER</div>
            <div class="model-hero-title-v2">Predict the trip.<br>Protect the operation.</div>
            <div class="model-hero-copy">
                A production-ready Gradient Boosting intelligence layer that converts
                trip conditions into delay probability, operational risk and recommended action.
            </div>
            <div class="live-chip"><i></i> LIVE MODEL · READY FOR INFERENCE</div>
        </div>
    </div>
    """), unsafe_allow_html=True)

    # --------------------------------------------------------
    # EXECUTIVE KPIs
    # --------------------------------------------------------
    k1, k2, k3, k4 = st.columns(4)

    with k1:
        st.markdown(dedent("""
        <div class="v2-card">
            <div class="v2-label">Model Status</div>
            <div class="v2-big v2-good">● ONLINE</div>
            <div class="v2-muted">Pipeline loaded successfully</div>
        </div>
        """), unsafe_allow_html=True)

    with k2:
        st.markdown(dedent("""
        <div class="v2-card">
            <div class="v2-label">Algorithm</div>
            <div class="v2-big">Gradient Boosting</div>
            <div class="v2-muted">Binary classification engine</div>
        </div>
        """), unsafe_allow_html=True)

    with k3:
        st.markdown(dedent("""
        <div class="v2-card">
            <div class="v2-label">ROC-AUC</div>
            <div class="v2-big v2-warn">72.95%</div>
            <div class="v2-muted">Held-out test performance</div>
        </div>
        """), unsafe_allow_html=True)

    with k4:
        st.markdown(dedent("""
        <div class="v2-card">
            <div class="v2-label">Accuracy</div>
            <div class="v2-big">70.75%</div>
            <div class="v2-muted">2,000 trip test set</div>
        </div>
        """), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # MODEL SCORE + HEALTH
    # --------------------------------------------------------
    left, right = st.columns([1.15, 1])

    with left:
        st.markdown("### 🎯 Model Performance")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=72.95,
            number={"suffix": "%", "font": {"size": 34}},
            title={"text": "ROC-AUC", "font": {"size": 15}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": "#f59e0b"},
                "bgcolor": "rgba(15,23,42,.8)",
                "borderwidth": 1,
                "bordercolor": "rgba(148,163,184,.18)",
                "steps": [
                    {"range": [0, 50], "color": "rgba(239,68,68,.15)"},
                    {"range": [50, 70], "color": "rgba(245,158,11,.12)"},
                    {"range": [70, 100], "color": "rgba(34,197,94,.10)"}
                ],
                "threshold": {
                    "line": {"color": "#22c55e", "width": 3},
                    "thickness": .8,
                    "value": 70
                }
            }
        ))
        fig.update_layout(
            height=290,
            margin=dict(l=25, r=25, t=35, b=15),
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#f8fafc"}
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with right:
        st.markdown("### 🩺 Model Health")

        st.markdown(dedent("""
        <div class="v2-card">
            <div class="signal">
                <span class="signal-name">Inference engine</span>
                <span class="signal-value v2-good">Operational</span>
            </div>
            <div class="signal">
                <span class="signal-name">Training observations</span>
                <span class="signal-value">8,000</span>
            </div>
            <div class="signal">
                <span class="signal-name">Test observations</span>
                <span class="signal-value">2,000</span>
            </div>
            <div class="signal">
                <span class="signal-name">Total observations</span>
                <span class="signal-value">10,000</span>
            </div>
            <div class="signal">
                <span class="signal-name">Decision target</span>
                <span class="signal-value">Trip Delay</span>
            </div>
            <div class="signal">
                <span class="signal-name">Model artifact</span>
                <span class="signal-value">.pkl loaded</span>
            </div>
        </div>
        """), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # PIPELINE
    # --------------------------------------------------------
    st.markdown("### ⚙️ AI Decision Pipeline")

    st.markdown(dedent("""
    <div class="pipeline">
        <div class="pipeline-node">
            <b>🚛 Trip Data</b>
            <span>12 operational signals</span>
        </div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-node">
            <b>🧹 Preprocessing</b>
            <span>Categorical + numerical features</span>
        </div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-node">
            <b>🧠 Gradient Boosting</b>
            <span>Delay probability</span>
        </div>
        <div class="pipeline-arrow">→</div>
        <div class="pipeline-node">
            <b>🚨 Risk Engine</b>
            <span>Risk + action</span>
        </div>
    </div>
    """), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # PERFORMANCE SNAPSHOT
    # --------------------------------------------------------
    p1, p2 = st.columns([1.05, 1])

    with p1:
        st.markdown("### 📊 Performance Snapshot")

        perf = pd.DataFrame({
            "Metric": ["Accuracy", "ROC-AUC", "Average Precision"],
            "Score": [70.75, 72.95, 58.93]
        })

        fig = px.bar(
            perf,
            x="Score",
            y="Metric",
            orientation="h",
            text="Score"
        )
        fig.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside"
        )
        fig.update_layout(
            template="plotly_dark",
            height=280,
            margin=dict(l=10, r=40, t=15, b=10),
            xaxis=dict(range=[0, 100], title="Score (%)"),
            yaxis_title="",
            showlegend=False,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with p2:
        st.markdown("### 🔍 Operational Signal Strength")

        signals = [
            ("Average Speed", "Primary", "High influence"),
            ("Distance", "Strong", "High influence"),
            ("Loading Time", "Strong", "Operational"),
            ("Previous Delays", "Strong", "Historical"),
            ("Weather", "Strong", "Environmental"),
            ("Fuel Efficiency", "Moderate", "Vehicle"),
        ]

        signal_items = []
        for name, strength, meaning in signals:
            signal_items.append(
                f'<div class="signal"><span class="signal-name">{name}</span>'
                f'<span><span class="v2-warn">{strength}</span> '
                f'<span style="color:#64748b;font-size:.68rem;">· {meaning}</span></span></div>'
            )
        html = '<div class="v2-card">' + ''.join(signal_items) + '</div>'
        st.markdown(html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # INPUT FEATURES
    # --------------------------------------------------------
    st.markdown("### 🧩 Model Input Layer")

    feature_details = [
        ("Distance", "km", "Route length"),
        ("Coal Weight", "ton", "Cargo load"),
        ("Fuel Used", "L", "Fuel consumption"),
        ("Average Speed", "km/h", "Vehicle speed"),
        ("Loading Time", "min", "Loading duration"),
        ("Unloading Time", "min", "Unloading duration"),
        ("Weather", "class", "Environmental condition"),
        ("Traffic Level", "class", "Road congestion"),
        ("Driver Experience", "years", "Driver history"),
        ("Previous Delays", "count", "Delay history"),
        ("Safety Score", "0–100", "Safety indicator"),
        ("Fuel Efficiency", "km/L", "Vehicle efficiency"),
    ]

    feature_items = []
    for name, unit, desc in feature_details:
        feature_items.append(
            f'<div class="feature-box"><b>{name}</b>'
            f'<span>{unit} · {desc}</span></div>'
        )
    html = '<div class="feature-grid">' + ''.join(feature_items) + '</div>'
    st.markdown(html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # RISK BANDS
    # --------------------------------------------------------
    st.markdown("### 🚦 Risk Decision Bands")

    bands = [
        ("LOW", "< 40%", "Normal operation", "v2-good"),
        ("MEDIUM", "40–59%", "Routine monitoring", "v2-warn"),
        ("HIGH", "60–79%", "Enhanced monitoring", "v2-warn"),
        ("CRITICAL", "≥ 80%", "Immediate intervention", "v2-danger"),
    ]

    cols = st.columns(4)
    for col, (level, band, action, cls) in zip(cols, bands):
        with col:
            st.markdown(dedent(f"""
            <div class="v2-card">
                <div class="v2-label">{level}</div>
                <div class="v2-big {cls}">{band}</div>
                <div class="v2-muted">{action}</div>
            </div>
            """), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --------------------------------------------------------
    # FEATURE IMPORTANCE PREVIEW
    # --------------------------------------------------------
    importance_path = BASE_DIR / "data" / "processed" / "feature_importance.csv"

    if importance_path.exists():
        st.markdown("### 🏆 Top Model Drivers")

        importance = pd.read_csv(importance_path)
        feature_col = importance.columns[0]
        importance_col = importance.columns[1]

        top = (
            importance
            .sort_values(importance_col, ascending=False)
            .head(8)
            .sort_values(importance_col, ascending=True)
        )

        fig = px.bar(
            top,
            x=importance_col,
            y=feature_col,
            orientation="h",
            text=importance_col
        )
        fig.update_traces(
            texttemplate="%{text:.3f}",
            textposition="outside"
        )
        fig.update_layout(
            template="plotly_dark",
            height=390,
            margin=dict(l=10, r=45, t=15, b=10),
            xaxis_title="Importance",
            yaxis_title="",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.info(
        "Production note: this model is a decision-support system. "
        "Risk scores indicate operational likelihood and should be combined "
        "with dispatch conditions, route knowledge and human judgment."
    )

