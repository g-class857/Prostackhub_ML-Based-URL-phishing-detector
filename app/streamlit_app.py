"""
streamlit_app.py
-----------------
Streamlit interface for the Phishing URL Detector — an alternative to
flask_app.py using the same trained model and feature pipeline.

Run with:
    pip install -r requirements.txt
    streamlit run app/streamlit_app.py
"""

import json
import os
import sys

import joblib
import pandas as pd
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feature_extractor import FEATURE_NAMES, extract_features  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "models", "metrics.json")

st.set_page_config(page_title="Phishing URL Detector", page_icon="🛡️", layout="centered")


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    if not os.path.exists(METRICS_PATH):
        return {}
    with open(METRICS_PATH) as f:
        return json.load(f)


model = load_model()
metrics = load_metrics()

st.title("🛡️ Phishing URL Detector")
st.caption(
    "Paste any URL to get a real-time phishing risk score based on its "
    "structure — length, host pattern, IP usage, keywords, subdomains and more."
)

if model is None:
    st.error(
        "No trained model found at `models/best_model.pkl`. "
        "Run `python src/train_model.py` first, then reload this page."
    )
    st.stop()

with st.sidebar:
    st.header("Model performance")
    results = metrics.get("results", {})
    selected = results.get("selected_model")
    best = results.get(selected, {})
    if best:
        st.metric("Selected model", selected)
        c1, c2 = st.columns(2)
        c1.metric("Accuracy", f"{best['accuracy']*100:.1f}%")
        c2.metric("Precision", f"{best['precision']*100:.1f}%")
        c1.metric("Recall", f"{best['recall']*100:.1f}%")
        c2.metric("F1-score", f"{best['f1_score']*100:.1f}%")
        st.caption(f"ROC-AUC: {best['roc_auc']:.3f}")
        cm = best.get("confusion_matrix")
        if cm:
            st.write("Confusion matrix (test set)")
            st.dataframe(
                pd.DataFrame(cm, index=["Actual: Legit", "Actual: Phishing"],
                              columns=["Pred: Legit", "Pred: Phishing"])
            )
    st.divider()
    st.caption(
        "⚠️ This is a demo trained on a synthetic dataset that mirrors "
        "common phishing URL patterns. Swap in the real Kaggle dataset "
        "(see data/generate_dataset.py) before using in production."
    )

url = st.text_input(
    "Enter a URL to analyze",
    placeholder="http://paypal-secure-login.verify-account.tk/signin",
)
lookup_age = st.checkbox(
    "Look up live domain age via WHOIS (slower, requires network)", value=False
)

col_a, col_b = st.columns([1, 4])
scan_clicked = col_a.button("Scan URL", type="primary")

if scan_clicked and url.strip():
    with st.spinner("Extracting features and scoring..."):
        feats = extract_features(url.strip(), lookup_domain_age=lookup_age)
        vector_df = pd.DataFrame(
            [[feats.values[n] for n in FEATURE_NAMES]], columns=FEATURE_NAMES
        )
        proba = model.predict_proba(vector_df)[0]
        phishing_prob = float(proba[1])
        is_phishing = phishing_prob >= 0.5

    st.subheader("Result")
    if is_phishing:
        st.error(f"⚠️ Likely **PHISHING** — {phishing_prob*100:.1f}% phishing probability")
    else:
        st.success(f"✅ Likely **LEGITIMATE** — {phishing_prob*100:.1f}% phishing probability")

    st.progress(phishing_prob, text=f"Phishing risk: {phishing_prob*100:.1f}%")

    st.subheader("Why the model made this call")
    importance = metrics.get("feature_importance", {})
    rows = []
    for name in sorted(importance, key=importance.get, reverse=True)[:10]:
        rows.append({
            "feature": name,
            "value": feats.values.get(name),
            "importance": round(importance[name], 4),
            "why it matters": feats.explanation.get(name, ""),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with st.expander("See all 22 extracted features"):
        st.json(feats.values)

    st.caption(
        "This is a statistical estimate based on URL structure only. "
        "Never enter credentials on a site you're unsure about — navigate "
        "there directly instead of clicking a link."
    )
elif scan_clicked:
    st.warning("Please enter a URL first.")
