"""
Streamlit dashboard for UPI FraudGuard AI.
Run with: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import joblib
import json
import os
import plotly.express as px
import plotly.graph_objects as go
from xgboost import XGBClassifier

st.set_page_config(page_title="UPI FraudGuard AI", page_icon="🛡️", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "upi_transactions_features.csv")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
OUT_DIR = os.path.join(BASE_DIR, "..", "outputs")

PLOTLY_TEMPLATE = "plotly_dark"
GOOD = "#22D3A6"
BAD = "#FF5C7A"
MUTED = "#9AA5B8"
BLUE = "#7C9CFF"
AMBER = "#F5B942"
PURPLE = "#C77DFF"
PINK = "#FF7AC6"
CYAN = "#4FD8EA"

MODEL_COLORS = {
    "Logistic Regression": BLUE,
    "Decision Tree": PURPLE,
    "Random Forest": AMBER,
    "XGBoost": GOOD,
}
METRIC_COLORS = {
    "accuracy": GOOD, "precision": BLUE, "recall": AMBER,
    "f1": PURPLE, "roc_auc": PINK, "pr_auc": CYAN,
}

# ------------------------------------------------------------------
# Custom CSS
# ------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"]  { font-family: 'Inter', sans-serif; }

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }

.hero {
    background: linear-gradient(120deg, #122A4A 0%, #1B1B4D 45%, #0B0F19 100%);
    border: 1px solid #2A3350;
    border-radius: 18px;
    padding: 28px 32px;
    margin-bottom: 28px;
    display: flex;
    align-items: center;
    gap: 20px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "";
    position: absolute; top: -60%; right: -10%;
    width: 320px; height: 320px; border-radius: 50%;
    background: radial-gradient(circle, rgba(34,211,166,0.28) 0%, rgba(124,156,255,0.12) 45%, transparent 70%);
    pointer-events: none;
}
.hero-icon {
    font-size: 42px;
    background: linear-gradient(135deg, rgba(34,211,166,0.22), rgba(124,156,255,0.18));
    border: 1px solid rgba(34, 211, 166, 0.45);
    border-radius: 16px;
    width: 72px; height: 72px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    z-index: 1;
}
.hero-title {
    font-size: 30px; font-weight: 800; margin: 0; letter-spacing: -0.5px;
    background: linear-gradient(90deg, #F4F6FA 0%, #B9F5E4 100%);
    -webkit-background-clip: text; background-clip: text; color: transparent;
    z-index: 1; position: relative;
}
.hero-sub { font-size: 15px; color: #AEB7CC; margin-top: 4px; z-index: 1; position: relative; }
.hero-strip {
    height: 4px; width: 100%; border-radius: 4px; margin-top: 16px;
    background: linear-gradient(90deg, #22D3A6, #7C9CFF, #C77DFF, #FF7AC6, #F5B942);
    z-index: 1; position: relative;
}

.section-label {
    font-size: 12px; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase;
    color: #22D3A6; margin-bottom: 4px;
}
.section-title { font-size: 22px; font-weight: 700; color: #F4F6FA; margin-bottom: 18px; }

.kpi-card {
    background: linear-gradient(160deg, #131A2B 0%, #10141F 100%);
    border: 1px solid #212A3D; border-left: 3px solid var(--accent, #22D3A6);
    border-radius: 14px;
    padding: 18px 20px; text-align: left;
    transition: transform 0.15s ease, border-color 0.15s ease;
}
.kpi-card:hover { transform: translateY(-2px); }
.kpi-label { font-size: 12px; color: #9AA5B8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.6px; }
.kpi-value { font-size: 28px; font-weight: 800; color: #F4F6FA; margin-top: 4px; }
.kpi-bar-bg { background: #212A3D; border-radius: 6px; height: 6px; margin-top: 10px; overflow: hidden; }
.kpi-bar-fill { height: 100%; border-radius: 6px; }

.model-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 12px; border-radius: 999px; font-size: 12.5px; font-weight: 700;
    background: rgba(255,255,255,0.06); border: 1px solid;
}
.model-badge .dot { width: 8px; height: 8px; border-radius: 50%; }

.cmp-table { width: 100%; border-collapse: separate; border-spacing: 0; font-size: 14px; }
.cmp-table th {
    text-align: left; padding: 10px 14px; color: #9AA5B8; font-weight: 700;
    text-transform: uppercase; font-size: 11px; letter-spacing: 0.6px;
    border-bottom: 1px solid #212A3D;
}
.cmp-table td {
    padding: 12px 14px; color: #E7ECF5; border-bottom: 1px solid #1B2333;
}
.cmp-table tr:last-child td { border-bottom: none; }
.cmp-table tr.best-row td { background: rgba(34, 211, 166, 0.06); }
.cmp-best {
    font-weight: 800; padding: 2px 8px; border-radius: 8px;
    background: rgba(34, 211, 166, 0.16); color: #22D3A6;
}

.verdict-card {
    border-radius: 16px; padding: 26px 28px; margin-top: 18px;
    display: flex; align-items: center; gap: 18px; border: 1px solid;
}
.verdict-icon { font-size: 34px; }
.verdict-title { font-size: 20px; font-weight: 800; margin: 0; }
.verdict-sub { font-size: 14px; margin-top: 2px; opacity: 0.85; }
.verdict-bar-bg { background: rgba(255,255,255,0.08); border-radius: 8px; height: 10px; margin-top: 12px; width: 100%; overflow: hidden; }
.verdict-bar-fill { height: 100%; border-radius: 8px; }

div[data-testid="stForm"] {
    background: #10141F; border: 1px solid #1F2937; border-radius: 16px; padding: 24px 26px 8px 26px;
}

.stButton > button, div[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(135deg, #22D3A6, #7C9CFF) !important;
    color: #06120F !important; font-weight: 700 !important; border: none !important; border-radius: 10px !important;
    padding: 10px 26px !important; transition: transform 0.15s ease;
}
.stButton > button:hover, div[data-testid="stFormSubmitButton"] button:hover {
    transform: translateY(-1px); filter: brightness(1.05);
}
div[data-testid="stFormSubmitButton"] button p { color: #06120F !important; font-weight: 700 !important; }

.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
    background-color: #131A2B; border-radius: 10px 10px 0 0; padding: 10px 18px; font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(34,211,166,0.16), rgba(124,156,255,0.10));
    color: #22D3A6 !important;
    border-bottom: 2px solid #22D3A6;
}

.badge {
    display: inline-block; padding: 3px 12px; border-radius: 999px; font-size: 12px; font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

def kpi_card(label, value, pct_of_100, color):
    st.markdown(f"""
    <div class="kpi-card" style="--accent:{color};">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-bar-bg"><div class="kpi-bar-fill" style="width:{min(pct_of_100,100)}%; background:{color};"></div></div>
    </div>
    """, unsafe_allow_html=True)


def model_badge(name, color):
    return (f'<span class="model-badge" style="border-color:{color}55; color:{color};">'
            f'<span class="dot" style="background:{color};"></span>{name}</span>')


@st.cache_resource
def load_artifacts():
    model = XGBClassifier()
    model.load_model(os.path.join(MODEL_DIR, "xgb_fraud_model.json"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    encoders = joblib.load(os.path.join(MODEL_DIR, "label_encoders.pkl"))
    with open(os.path.join(MODEL_DIR, "best_threshold.json")) as f:
        cfg = json.load(f)
    return model, scaler, encoders, cfg

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

model, scaler, encoders, cfg = load_artifacts()
df = load_data()
with open(os.path.join(OUT_DIR, "model_metrics.json")) as f:
    metrics = json.load(f)

# ------------------------------------------------------------------
# Hero header
# ------------------------------------------------------------------
st.markdown("""
<div class="hero">
    <div class="hero-icon">🛡️</div>
    <div style="flex:1;">
        <p class="hero-title">UPI FraudGuard AI</p>
        <p class="hero-sub">ML-based anomaly detection for Unified Payments Interface (UPI) transactions — 4 models trained &amp; compared</p>
        <div class="hero-strip"></div>
    </div>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔍  Score a Transaction", "📊  Model Performance", "🗺️  Regional & Category Risk"])

# ==================== TAB 1: SCORE A TRANSACTION ====================
with tab1:
    st.markdown('<div class="section-label">Live Scoring</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Score a hypothetical transaction</div>', unsafe_allow_html=True)

    with st.form("score_form"):
        col1, col2, col3 = st.columns(3)
        amount = col1.number_input("Amount (₹)", min_value=1.0, value=1500.0)

        hour_labels = [f"{(h % 12) if (h % 12 != 0) else 12} {'AM' if h < 12 else 'PM'}" for h in range(24)]
        hour_label = col2.select_slider("Hour of day", options=hour_labels, value=hour_labels[14])
        hour = hour_labels.index(hour_label)

        is_weekend = col3.selectbox("Weekend?", [0, 1])

        col4, col5, col6 = st.columns(3)
        sender_state = col4.selectbox("Sender state", encoders["sender_state"].classes_)
        sender_bank = col5.selectbox("Sender bank", encoders["sender_bank"].classes_)
        receiver_bank = col6.selectbox("Receiver bank", encoders["receiver_bank"].classes_)

        col7, col8, col9 = st.columns(3)
        merchant_category = col7.selectbox("Merchant category", encoders["merchant_category"].classes_)
        device_type = col8.selectbox("Device type", encoders["device_type"].classes_)
        network_type = col9.selectbox("Network type", encoders["network_type"].classes_)

        velocity = st.slider("Transactions by this sender in last hour", 1, 15, 1)
        dev_ratio = st.slider("Amount ÷ sender's usual amount", 0.1, 10.0, 1.0)

        submitted = st.form_submit_button("Check transaction")

    if submitted:
        row = {
            "amount": amount, "hour_of_day": hour, "is_weekend": is_weekend,
            "txn_velocity_1h": velocity, "amount_deviation_ratio": dev_ratio,
            "is_odd_hour": 1 if (hour <= 5 or hour == 23) else 0,
            "device_switch_flag": 0, "cross_bank_flag": 1 if sender_bank != receiver_bank else 0,
            "sender_state_enc": encoders["sender_state"].transform([sender_state])[0],
            "sender_bank_enc": encoders["sender_bank"].transform([sender_bank])[0],
            "receiver_bank_enc": encoders["receiver_bank"].transform([receiver_bank])[0],
            "merchant_category_enc": encoders["merchant_category"].transform([merchant_category])[0],
            "device_type_enc": encoders["device_type"].transform([device_type])[0],
            "network_type_enc": encoders["network_type"].transform([network_type])[0],
        }
        X_new = pd.DataFrame([row])[cfg["feature_cols"]]
        numeric_cols = ["amount", "hour_of_day", "is_weekend", "txn_velocity_1h",
                         "amount_deviation_ratio", "is_odd_hour", "device_switch_flag", "cross_bank_flag"]
        X_new[numeric_cols] = scaler.transform(X_new[numeric_cols])
        proba = model.predict_proba(X_new)[0, 1]
        is_fraud = proba >= cfg["threshold"]
        pct = proba * 100

        if is_fraud:
            st.markdown(f"""
            <div class="verdict-card" style="background: rgba(255,92,122,0.08); border-color: rgba(255,92,122,0.35);">
                <div class="verdict-icon">⚠️</div>
                <div style="flex:1;">
                    <p class="verdict-title" style="color:{BAD};">Flagged as HIGH RISK</p>
                    <p class="verdict-sub" style="color:{BAD};">Fraud probability: {pct:.1f}% — above the {cfg['threshold']*100:.1f}% decision threshold</p>
                    <div class="verdict-bar-bg"><div class="verdict-bar-fill" style="width:{pct}%; background:{BAD};"></div></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="verdict-card" style="background: rgba(34,211,166,0.08); border-color: rgba(34,211,166,0.35);">
                <div class="verdict-icon">✅</div>
                <div style="flex:1;">
                    <p class="verdict-title" style="color:{GOOD};">Looks genuine</p>
                    <p class="verdict-sub" style="color:{GOOD};">Fraud probability: {pct:.1f}% — below the {cfg['threshold']*100:.1f}% decision threshold</p>
                    <div class="verdict-bar-bg"><div class="verdict-bar-fill" style="width:{pct}%; background:{GOOD};"></div></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ==================== TAB 2: MODEL PERFORMANCE ====================
with tab2:
    st.markdown('<div class="section-label">Diagnostics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Deployed model — XGBoost (tuned threshold)</div>', unsafe_allow_html=True)

    m = metrics["XGBoost (tuned threshold)"]
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: kpi_card("Accuracy", f"{m['accuracy']*100:.2f}%", m['accuracy']*100, METRIC_COLORS["accuracy"])
    with c2: kpi_card("Precision", f"{m['precision']*100:.1f}%", m['precision']*100, METRIC_COLORS["precision"])
    with c3: kpi_card("Recall", f"{m['recall']*100:.1f}%", m['recall']*100, METRIC_COLORS["recall"])
    with c4: kpi_card("F1 Score", f"{m['f1']*100:.1f}%", m['f1']*100, METRIC_COLORS["f1"])
    with c5: kpi_card("ROC-AUC", f"{m['roc_auc']*100:.1f}%", m['roc_auc']*100, METRIC_COLORS["roc_auc"])
    with c6: kpi_card("PR-AUC", f"{m['pr_auc']*100:.1f}%", m['pr_auc']*100, METRIC_COLORS["pr_auc"])

    st.write("")

    # ---------- Full 4-model comparison table ----------
    st.markdown('<div class="section-label">Leaderboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">All 4 models — full metrics</div>', unsafe_allow_html=True)

    model_rows = [
        ("Logistic Regression", metrics["Logistic Regression (baseline)"]),
        ("Decision Tree", metrics["Decision Tree"]),
        ("Random Forest", metrics["Random Forest"]),
        ("XGBoost", metrics["XGBoost (tuned threshold)"]),
    ]
    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC", "PR-AUC"]

    # best value per metric, for highlighting
    best_vals = {k: max(vals[k] for _, vals in model_rows) for k in metric_keys}

    legend_html = " ".join(model_badge(name, MODEL_COLORS[name]) for name, _ in model_rows)
    st.markdown(f'<div style="margin-bottom:14px;">{legend_html}</div>', unsafe_allow_html=True)

    header_html = "".join(f"<th>{lbl}</th>" for lbl in metric_labels)
    rows_html = ""
    for name, vals in model_rows:
        color = MODEL_COLORS[name]
        is_deployed = name == "XGBoost"
        cells = ""
        for k in metric_keys:
            v = vals[k]
            txt = f"{v*100:.1f}%"
            if v == best_vals[k]:
                cells += f'<td><span class="cmp-best">{txt}</span></td>'
            else:
                cells += f"<td>{txt}</td>"
        star = " ⭐" if is_deployed else ""
        row_class = ' class="best-row"' if is_deployed else ""
        rows_html += (
            f'<tr{row_class}><td>{model_badge(name + star, color)}</td>{cells}</tr>'
        )

    st.markdown(f"""
    <div style="overflow-x:auto; background:#10141F; border:1px solid #1F2937; border-radius:16px; padding:8px 6px;">
    <table class="cmp-table">
        <thead><tr><th>Model</th>{header_html}</tr></thead>
        <tbody>{rows_html}</tbody>
    </table>
    </div>
    <p style="color:#9AA5B8; font-size:12.5px; margin-top:10px;">⭐ = model deployed for live scoring in the "Score a Transaction" tab. Highlighted cells mark the best score for that metric across all 4 models.</p>
    """, unsafe_allow_html=True)

    st.write("")
    colA, colB = st.columns(2)
    with colA:
        with st.container(border=True):
            st.markdown("**Model comparison — all 4 models**")
            st.image(os.path.join(OUT_DIR, "model_comparison.png"), use_container_width=True)
    with colB:
        with st.container(border=True):
            st.markdown("**Confusion matrix — test set (XGBoost)**")
            st.image(os.path.join(OUT_DIR, "confusion_matrix.png"), use_container_width=True)

    with st.container(border=True):
        st.markdown("**Top features driving fraud predictions**")
        st.image(os.path.join(OUT_DIR, "feature_importance.png"), use_container_width=True)

# ==================== TAB 3: REGIONAL & CATEGORY RISK ====================
with tab3:
    st.markdown('<div class="section-label">Risk Aggregation</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Where is fraud risk concentrated right now?</div>', unsafe_allow_html=True)

    region_risk = pd.read_csv(os.path.join(OUT_DIR, "regional_risk.csv"))
    category_risk = pd.read_csv(os.path.join(OUT_DIR, "category_risk.csv"))

    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            fig = px.bar(region_risk, x="sender_state", y="flagged_rate_pct",
                         title="Flagged fraud rate by state (%)", color="flagged_rate_pct",
                         color_continuous_scale=["#151B2C", "#22D3A6"], template=PLOTLY_TEMPLATE)
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               coloraxis_showscale=False, margin=dict(t=50, b=10))
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        with st.container(border=True):
            fig2 = px.bar(category_risk, x="merchant_category", y="flagged_rate_pct",
                          title="Flagged fraud rate by merchant category (%)", color="flagged_rate_pct",
                          color_continuous_scale=["#151B2C", "#FF5C7A"], template=PLOTLY_TEMPLATE)
            fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                coloraxis_showscale=False, margin=dict(t=50, b=10))
            st.plotly_chart(fig2, use_container_width=True)

    colC, colD = st.columns(2)
    with colC:
        with st.container(border=True):
            st.markdown("**State-wise detail**")
            st.dataframe(region_risk, use_container_width=True, hide_index=True)
    with colD:
        with st.container(border=True):
            st.markdown("**Category-wise detail**")
            st.dataframe(category_risk, use_container_width=True, hide_index=True)
