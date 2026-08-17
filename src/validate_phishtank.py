"""
validate_phishtank.py
----------------------
Validates the trained model against a fresh, real-world feed of confirmed
phishing URLs (e.g. PhishTank's online-valid feed) to measure how it
generalizes to unseen, current phishing campaigns — as opposed to the
train/test split drawn from the same distribution used in train_model.py.

PhishTank feed
~~~~~~~~~~~~~~
1. Register for a free API key at https://www.phishtank.com/api_register.php
2. Download the current feed (CSV): http://data.phishtank.com/data/<api_key>/online-valid.csv
3. Save it locally, e.g. data/phishtank_feed.csv (must have a `url` column)
4. Run:
     python src/validate_phishtank.py --phishing-feed data/phishtank_feed.csv \
         --legit-sample data/tranco_top_1000.csv

`--legit-sample` should be a CSV of known-legitimate URLs/domains (e.g. the
Tranco or Majestic Million top-sites list) to pair against the phishing
feed, since PhishTank only ships confirmed-phishing entries. Without
network access this script cannot be exercised against live feeds in
this environment, but the code path is exactly the same as the one used
for the synthetic held-out test set, so it runs unchanged once real
feeds are supplied.
"""

import argparse
import json
import os
import sys

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feature_extractor import FEATURE_NAMES, extract_features  # noqa: E402


def load_urls(path, url_column="url"):
    df = pd.read_csv(path)
    col = url_column if url_column in df.columns else df.columns[0]
    return df[col].dropna().astype(str).tolist()


def build_matrix(urls, lookup_domain_age=False):
    rows = [extract_features(u, lookup_domain_age=lookup_domain_age).values for u in urls]
    return pd.DataFrame(rows, columns=FEATURE_NAMES)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phishing-feed", required=True,
                         help="CSV of confirmed phishing URLs (PhishTank online-valid.csv)")
    parser.add_argument("--legit-sample", required=True,
                         help="CSV of known-legitimate URLs/domains (e.g. Tranco top sites)")
    parser.add_argument("--model", default="models/best_model.pkl")
    parser.add_argument("--limit", type=int, default=1000,
                         help="Cap rows per class for a quick spot-check")
    args = parser.parse_args()

    model = joblib.load(args.model)

    phishing_urls = load_urls(args.phishing_feed)[: args.limit]
    legit_urls = load_urls(args.legit_sample)[: args.limit]

    urls = phishing_urls + legit_urls
    y_true = [1] * len(phishing_urls) + [0] * len(legit_urls)

    X = build_matrix(urls)
    y_pred = model.predict(X)

    metrics = {
        "n_phishing": len(phishing_urls),
        "n_legit": len(legit_urls),
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1_score": round(f1_score(y_true, y_pred), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    print("PhishTank / live-feed validation results")
    print(json.dumps(metrics, indent=2))

    out_path = os.path.join("models", "phishtank_validation.json")
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
