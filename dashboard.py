from pathlib import Path
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "train.csv"
API_URL = "http://127.0.0.1:8000"

# Only these 8 features are used for individual churn prediction.
FEATURE_COLUMNS = [
    "subscription_type",
    "payment_plan",
    "num_subscription_pauses",
    "customer_service_inquiries",
    "weekly_hours",
    "song_skip_rate",
    "weekly_songs_played",
    "notifications_clicked",
]

st.set_page_config(
    page_title="Smart Music Churn & Retention",
    page_icon="🎧",
    layout="wide",
)

st.markdown("""
<style>
.stApp { background: #0e1117; color: #f4f6fb; }
[data-testid="stSidebar"] { background: #252832; }
div[data-testid="stMetric"] {
    background: #191d27; border: 1px solid #303746;
    padding: 16px; border-radius: 12px;
}
div.stButton > button[kind="primary"],
div.stFormSubmitButton > button {
    background: #ff454f; border: 0; color: white;
}
</style>
""", unsafe_allow_html=True)

st.title("🎧 Smart Music Streaming Churn & Retention System")
st.caption("Subscriber analytics, churn-risk prediction, and transparent retention suggestions.")

if not DATA_PATH.exists():
    st.error("Dataset not found. Put train.csv inside the project's data/ folder.")
    st.code("customer-churn-predict/data/train.csv")
    st.stop()

df = pd.read_csv(DATA_PATH)
if "churned" not in df.columns:
    st.error("The training dataset must contain a 'churned' target column.")
    st.stop()

missing_features = [col for col in FEATURE_COLUMNS if col not in df.columns]
if missing_features:
    st.error(f"Your train.csv is missing required prediction columns: {missing_features}")
    st.stop()

try:
    meta_response = requests.get(f"{API_URL}/metadata", timeout=3)
    metadata = meta_response.json() if meta_response.ok else None
except requests.RequestException:
    metadata = None

if metadata is None:
    st.warning("The prediction API is not available. Start it in another terminal with: uvicorn app:app --reload")
    st.info("Analytics remain available. Train the reduced-feature model first with `python train.py`.")
else:
    st.caption(
        f"Active model: {metadata['model_name']} · "
        f"Holdout ROC-AUC: {metadata['validation_roc_auc']:.3f} · "
        f"Prediction inputs: {len(metadata['feature_columns'])}"
    )

total = len(df)
churn_count = int(pd.to_numeric(df["churned"], errors="coerce").fillna(0).sum())
churn_rate = churn_count / total * 100 if total else 0
c1, c2, c3 = st.columns(3)
c1.metric("Training subscribers", f"{total:,}")
c2.metric("Churned subscribers", f"{churn_count:,}")
c3.metric("Observed churn rate", f"{churn_rate:.1f}%")

left, right = st.columns(2)
with left:
    st.subheader("Churn distribution")
    counts = (
        df["churned"].map({0: "Stayed", 1: "Churned"})
        .fillna(df["churned"].astype(str))
        .value_counts()
        .rename_axis("Status")
        .reset_index(name="Subscribers")
    )
    fig = px.pie(
        counts, names="Status", values="Subscribers", hole=0.58,
        color_discrete_sequence=["#2ca25f", "#ef4444"],
    )
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("Churn by subscription type")
    group = df.groupby("subscription_type", dropna=False)["churned"].mean().mul(100).reset_index(name="Churn rate (%)")
    fig = px.bar(
        group, x="subscription_type", y="Churn rate (%)",
        color="Churn rate (%)",
        color_continuous_scale=["#22c55e", "#f59e0b", "#ef4444"],
    )
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", coloraxis_showscale=False,
        xaxis_title="Subscription type",
    )
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Listening behaviour")
numeric_cols = [
    c for c in df.select_dtypes(include="number").columns
    if c not in ("churned", "customer_id")
]
candidate = [
    c for c in [
        "weekly_hours", "song_skip_rate", "weekly_songs_played",
        "weekly_unique_songs", "average_session_length",
    ] if c in numeric_cols
]
if candidate:
    selected_metric = st.selectbox("Choose a listening metric", candidate)
    outcome = df["churned"].map({0: "Stayed", 1: "Churned"})
    fig = px.histogram(
        df, x=selected_metric, color=outcome, barmode="overlay",
        opacity=0.72, nbins=35,
        color_discrete_map={"Stayed": "#38bdf8", "Churned": "#fb7185"},
    )
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)", legend_title_text="Outcome",
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No recognised listening metrics were found in this dataset.")

def numeric_bounds(column, fallback_min, fallback_max, integer=False):
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if series.empty:
        low, high, default = fallback_min, fallback_max, (fallback_min + fallback_max) / 2
    else:
        low, high = float(series.min()), float(series.max())
        default = float(series.median())
    if low == high:
        high = low + (1 if integer else 0.1)
    default = min(max(default, low), high)
    return low, high, default

def category_options(column, preferred=None):
    values = df[column].dropna().astype(str).unique().tolist()
    values = sorted(values, key=str.casefold)
    if preferred:
        # Keep dataset-supported options only, but show them in a predictable order.
        ordered = [value for value in preferred if value in values]
        values = ordered + [value for value in values if value not in ordered]
    return values or ["Unknown"]

st.divider()
st.header("Individual Subscriber Risk Assessment")
st.write("Enter a profile using just 8 inputs. The model has been trained specifically on these features.")

if metadata is None:
    st.info("Start the API and train the reduced-feature model to enable risk prediction.")
else:
    model_features = metadata.get("feature_columns", [])
    if set(model_features) != set(FEATURE_COLUMNS):
        st.error(
            "The API model is not trained with the expected 8 features. "
            "Stop the API, run `python train.py`, then restart the API."
        )
        st.stop()

    with st.form("subscriber_profile"):
        st.markdown("#### Subscriber profile")
        left_form, right_form = st.columns(2)
        values = {}

        with left_form:
            st.markdown("**Subscription Type**")
            subscription_options = category_options(
                "subscription_type", ["basic", "standard", "premium"]
            )
            values["subscription_type"] = st.radio(
                "Subscription Type", subscription_options,
                horizontal=True, label_visibility="collapsed",
                key="subscription_type",
            )

            st.markdown("**Subscription Pauses**")
            lo, hi, default = numeric_bounds("num_subscription_pauses", 0, 5, integer=True)
            values["num_subscription_pauses"] = st.slider(
                "Subscription Pauses", min_value=int(lo), max_value=int(hi),
                value=int(round(default)), step=1, label_visibility="collapsed",
                key="num_subscription_pauses",
            )

            st.markdown("**Weekly Listening Hours**")
            lo, hi, default = numeric_bounds("weekly_hours", 0, 40)
            values["weekly_hours"] = st.slider(
                "Weekly Listening Hours", min_value=float(lo), max_value=float(hi),
                value=float(default), step=0.5, label_visibility="collapsed",
                key="weekly_hours",
            )

            st.markdown("**Weekly Songs Played**")
            lo, hi, default = numeric_bounds("weekly_songs_played", 0, 500, integer=True)
            values["weekly_songs_played"] = st.slider(
                "Weekly Songs Played", min_value=int(lo), max_value=int(hi),
                value=int(round(default)), step=max(1, int(round((hi - lo) / 100))),
                label_visibility="collapsed", key="weekly_songs_played",
            )

        with right_form:
            st.markdown("**Payment Plan**")
            payment_options = category_options("payment_plan", ["monthly", "annual"])
            values["payment_plan"] = st.radio(
                "Payment Plan", payment_options,
                horizontal=True, label_visibility="collapsed", key="payment_plan",
            )

            st.markdown("**Customer Service Inquiries**")
            inquiry_options = category_options(
                "customer_service_inquiries", ["few", "many"]
            )
            values["customer_service_inquiries"] = st.radio(
                "Customer Service Inquiries", inquiry_options,
                horizontal=True, label_visibility="collapsed",
                key="customer_service_inquiries",
            )

            st.markdown("**Song Skip Rate**")
            lo, hi, default = numeric_bounds("song_skip_rate", 0.0, 1.0)
            values["song_skip_rate"] = st.slider(
                "Song Skip Rate", min_value=float(lo), max_value=float(hi),
                value=float(default), step=0.01, format="%.2f",
                label_visibility="collapsed", key="song_skip_rate",
            )

            st.markdown("**Notifications Clicked**")
            lo, hi, default = numeric_bounds("notifications_clicked", 0, 100, integer=True)
            values["notifications_clicked"] = st.slider(
                "Notifications Clicked", min_value=int(lo), max_value=int(hi),
                value=int(round(default)), step=max(1, int(round((hi - lo) / 100))),
                label_visibility="collapsed", key="notifications_clicked",
            )

        submitted = st.form_submit_button(
            "Evaluate Churn Risk", type="primary", use_container_width=True
        )

    if submitted:
        try:
            response = requests.post(
                f"{API_URL}/predict", json={"features": values}, timeout=10
            )
            if not response.ok:
                st.error(f"Prediction failed: {response.text}")
            else:
                result = response.json()
                risk_pct = result["risk_score_percentage"]
                a, b = st.columns([1, 1])
                with a:
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=risk_pct,
                        title={"text": "Churn Probability (%)"},
                        number={"suffix": "%", "font": {"color": "#f4f6fb"}},
                        gauge={
                            "axis": {"range": [0, 100]},
                            "bar": {"color": "#3b82f6"},
                            "steps": [
                                {"range": [0, 35], "color": "#16a34a"},
                                {"range": [35, 70], "color": "#f59e0b"},
                                {"range": [70, 100], "color": "#ef4444"},
                            ],
                        },
                    ))
                    fig.update_layout(
                        template="plotly_dark", height=330,
                        paper_bgcolor="rgba(0,0,0,0)",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with b:
                    st.subheader("Classification status")
                    if result["risk_level"] == "HIGH RISK":
                        st.error(f"⚠️ Status: **{result['risk_level']}**")
                    elif result["risk_level"] == "MEDIUM RISK":
                        st.warning(f"⚡ Status: **{result['risk_level']}**")
                    else:
                        st.success(f"✅ Status: **{result['risk_level']}**")
                    st.subheader("Recommended retention strategy")
                    st.info(result["recommendation"])
                    st.caption(
                        "Risk bands are operational thresholds, not calibrated business guarantees. "
                        "Recommendations are rule-based suggestions."
                    )
        except requests.RequestException:
            st.error("Could not connect to FastAPI. Start it with `uvicorn app:app --reload`.")

with st.expander("Preview training data"):
    st.dataframe(df.head(20), use_container_width=True)
