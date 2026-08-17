"""
flask_app.py
------------
Flask web application for real-time phishing URL detection.

Run with:
    pip install -r requirements.txt
    python app/flask_app.py
Then open http://127.0.0.1:5000
"""

import json
import os
import sys

from flask import Flask, jsonify, render_template, request

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feature_extractor import FEATURE_NAMES, extract_features  # noqa: E402

import joblib
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "models", "metrics.json")

app = Flask(__name__)

_model = None
_metrics = None


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "No trained model found at models/best_model.pkl. "
                "Run `python src/train_model.py` first."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def get_metrics():
    global _metrics
    if _metrics is None and os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            _metrics = json.load(f)
    return _metrics or {}


def predict_url(url: str):
    model = get_model()
    feats = extract_features(url, lookup_domain_age=True)
    vector_df = pd.DataFrame([[feats.values[name] for name in FEATURE_NAMES]],
                              columns=FEATURE_NAMES)

    proba = model.predict_proba(vector_df)[0]
    phishing_prob = float(proba[1])
    label = "Phishing" if phishing_prob >= 0.5 else "Legitimate"

    # Per-URL explainability: combine the model's global feature importance
    # with which "risky" signals actually fired for THIS url, so the
    # explanation is specific to the URL being analyzed, not generic.
    global_importance = get_metrics().get("feature_importance", {})
    risky_flags = {
        "has_at_symbol": "Contains an '@' symbol",
        "has_ip_address": "Uses a raw IP address instead of a domain name",
        "is_shortened": "Uses a known URL-shortening service",
        "has_suspicious_words": "Contains phishing-associated keywords",
        "has_double_slash_redirect": "Contains a suspicious '//' redirect pattern",
    }
    contributing = []
    for name, importance in sorted(global_importance.items(), key=lambda kv: kv[1], reverse=True):
        if name not in feats.values:
            continue
        value = feats.values[name]
        fired = False
        if name in risky_flags and value == 1:
            fired = True
        elif name == "has_https" and value == 0:
            fired = True
        elif name == "domain_age_days" and 0 <= value < 180:
            fired = True
        if fired or len(contributing) < 5:
            contributing.append({
                "feature": name,
                "value": value,
                "importance": round(importance, 4),
                "reason": risky_flags.get(name, feats.explanation.get(name, "")),
            })
        if len(contributing) >= 6:
            break

    return {
        "url": url,
        "label": label,
        "phishing_probability": round(phishing_prob, 4),
        "legitimate_probability": round(1 - phishing_prob, 4),
        "features": feats.values,
        "top_factors": contributing,
    }


@app.route("/")
def index():
    metrics = get_metrics()
    return render_template("index.html", metrics=metrics)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(force=True)
    url = (data or {}).get("url", "").strip()
    if not url:
        return jsonify({"error": "Please provide a URL."}), 400
    try:
        result = predict_url(url)
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500
    return jsonify(result)


@app.route("/api/metrics")
def api_metrics():
    return jsonify(get_metrics())


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
