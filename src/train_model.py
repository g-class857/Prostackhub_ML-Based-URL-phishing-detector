"""
train_model.py
---------------
Loads data/phishing_dataset.csv, extracts structural URL features,
trains Random Forest and XGBoost classifiers, evaluates both, and
persists the best-performing model + metadata for the web app.

Usage:
    python src/train_model.py --data data/phishing_dataset.csv
"""

import argparse
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feature_extractor import FEATURE_NAMES, extract_features  # noqa: E402

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("[warn] xgboost not installed — falling back to sklearn's "
          "GradientBoostingClassifier for the comparison model. "
          "Run `pip install xgboost` for the exact stack requested.")
    from sklearn.ensemble import GradientBoostingClassifier


def load_dataset(path: str, max_rows: int | None = None) -> pd.DataFrame:
    """Load the CSV and drop rows that can't be used at all (missing url or
    label). Malformed-but-present URLs are NOT dropped here — feature_extractor
    handles those safely — this only removes truly unusable rows.
    """
    df = pd.read_csv(path)
    before = len(df)

    df = df.dropna(subset=["url", "label"])
    df["url"] = df["url"].astype(str)
    dropped = before - len(df)
    if dropped:
        print(f"  [warn] dropped {dropped} rows with missing url/label")

    if max_rows and len(df) > max_rows:
        print(f"  [info] --max-rows {max_rows}: subsampling from {len(df)} rows "
              f"(stratified by label) for a faster run")
        # Use train_test_split purely as a stratified sampler (discard the
        # "test" half) rather than groupby().apply(), whose column-handling
        # behavior has changed across pandas versions (2.x vs 3.x).
        df, _ = train_test_split(
            df, train_size=max_rows, random_state=42, stratify=df["label"]
        )

    return df.reset_index(drop=True)


def build_feature_matrix(df: pd.DataFrame, progress_every: int = 100_000) -> pd.DataFrame:
    """Extract structural features for every URL in the dataframe.
    WHOIS domain-age lookup is skipped here (lookup_domain_age=False)
    because it requires a live network call per-domain and would make
    training on large datasets impractically slow / network-bound.
    The trained model therefore treats domain_age_days==-1 ("unknown")
    as its own signal, and the live app can optionally enrich a single
    URL with a real WHOIS lookup at prediction time.

    Prints periodic progress since this loop is the dominant cost on
    large (500k+ row) datasets and otherwise gives no feedback for
    several minutes.
    """
    n = len(df)
    rows = [None] * n
    t0 = time.time()
    for i, url in enumerate(df["url"].values):
        rows[i] = extract_features(url, lookup_domain_age=False).values
        if progress_every and (i + 1) % progress_every == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (n - (i + 1)) / rate if rate > 0 else float("inf")
            print(f"  ... {i + 1:,}/{n:,} rows "
                  f"({rate:,.0f} urls/sec, ~{remaining/60:.1f} min remaining)")
    return pd.DataFrame(rows, columns=FEATURE_NAMES)


def evaluate(model, X_test, y_test, name):
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "model": name,
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds), 4),
        "recall": round(recall_score(y_test, preds), 4),
        "f1_score": round(f1_score(y_test, preds), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
    }
    return metrics


def print_metrics(metrics):
    cm = np.array(metrics["confusion_matrix"])
    print(f"\n=== {metrics['model']} ===")
    print(f"Accuracy : {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall   : {metrics['recall']:.4f}")
    print(f"F1-score : {metrics['f1_score']:.4f}")
    print(f"ROC-AUC  : {metrics['roc_auc']:.4f}")
    print("Confusion matrix [ [TN FP] [FN TP] ]:")
    print(cm)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/phishing_dataset.csv")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--max-rows", type=int, default=None,
                         help="Optional cap on total rows (stratified subsample), "
                              "useful for a quick smoke test on very large datasets.")
    parser.add_argument("--tune", action="store_true",
                         help="Run a small GridSearchCV to optimize the "
                              "winning model (slower, better recall/precision trade-off).")
    parser.add_argument("--tune-sample-size", type=int, default=150_000,
                         help="On large datasets, GridSearchCV searches on a "
                              "stratified subsample of this size (not the full "
                              "training set) for speed, then the best "
                              "hyperparameters are refit on the FULL training "
                              "set. Ignored if the training set is smaller than this.")
    args = parser.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)

    print(f"Loading dataset from {args.data} ...")
    df = load_dataset(args.data, max_rows=args.max_rows)
    print(f"  {len(df)} rows | phishing={int(df['label'].sum())} "
          f"legitimate={int((df['label'] == 0).sum())}")

    print("Extracting structural features for every URL "
          "(this is the slow step on large datasets — progress below) ...")
    t0 = time.time()
    X = build_feature_matrix(df)
    y = df["label"].values
    print(f"  done in {time.time() - t0:.1f}s -> feature matrix {X.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    # ---- Random Forest ----------------------------------------------
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=14, min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_metrics = evaluate(rf, X_test, y_test, "RandomForest")
    print_metrics(rf_metrics)

    # ---- XGBoost (or GradientBoosting fallback) ----------------------
    if XGBOOST_AVAILABLE:
        gb = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.08,
            subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
            random_state=42, n_jobs=-1,
        )
        gb_name = "XGBoost"
    else:
        gb = GradientBoostingClassifier(
            n_estimators=250, max_depth=4, learning_rate=0.08, random_state=42,
        )
        gb_name = "GradientBoosting (xgboost fallback)"
    gb.fit(X_train, y_train)
    gb_metrics = evaluate(gb, X_test, y_test, gb_name)
    print_metrics(gb_metrics)

    # ---- Pick the best model on F1 (balances false positives/negatives) ----
    candidates = [(rf, rf_metrics), (gb, gb_metrics)]
    best_model, best_metrics = max(candidates, key=lambda c: c[1]["f1_score"])
    print(f"\n>>> Best model: {best_metrics['model']} "
          f"(F1={best_metrics['f1_score']}, Precision={best_metrics['precision']}, "
          f"Recall={best_metrics['recall']})")

    # ---- Optional hyperparameter tuning of the winner to cut false positives
    if args.tune:
        n_train = len(X_train)
        if n_train > args.tune_sample_size:
            print(f"\nTraining set has {n_train:,} rows — GridSearchCV would be "
                  f"very slow on the full set (each grid point refits multiple "
                  f"times). Searching hyperparameters on a stratified "
                  f"{args.tune_sample_size:,}-row subsample instead, then "
                  f"refitting the winning configuration on the FULL "
                  f"{n_train:,}-row training set. Override with --tune-sample-size.")
            search_idx, _ = train_test_split(
                np.arange(n_train), train_size=args.tune_sample_size,
                random_state=42, stratify=y_train,
            )
            X_search, y_search = X_train.iloc[search_idx], y_train[search_idx]
        else:
            X_search, y_search = X_train, y_train

        print("Running GridSearchCV to reduce false positives "
              "(optimizing for precision) ...")
        t_tune = time.time()
        if best_metrics["model"] == "RandomForest":
            param_grid = {
                "n_estimators": [200, 400],
                "max_depth": [10, 16, None],
                "min_samples_leaf": [1, 2, 4],
            }
            # n_jobs=1 on the base estimator during the search to avoid
            # nested parallelism fighting with GridSearchCV's own n_jobs=-1.
            base = RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=1)
        else:
            param_grid = {
                "n_estimators": [200, 300],
                "max_depth": [4, 6, 8],
                "learning_rate": [0.05, 0.1],
            }
            base = type(best_model)(random_state=42, n_jobs=1) if XGBOOST_AVAILABLE else \
                GradientBoostingClassifier(random_state=42)

        search = GridSearchCV(base, param_grid, scoring="precision", cv=3,
                               n_jobs=-1, refit=False)
        search.fit(X_search, y_search)
        print(f"  search done in {time.time() - t_tune:.1f}s | "
              f"best params: {search.best_params_}")

        # Refit a fresh model with the winning hyperparameters on the FULL
        # training set (not just the search subsample) for the final model.
        tuned_model = clone(base).set_params(**search.best_params_)
        if hasattr(tuned_model, "n_jobs"):
            tuned_model.set_params(n_jobs=-1)
        tuned_model.fit(X_train, y_train)

        tuned_metrics = evaluate(tuned_model, X_test, y_test,
                                  best_metrics["model"] + " (tuned)")
        print_metrics(tuned_metrics)
        if tuned_metrics["f1_score"] >= best_metrics["f1_score"]:
            best_model, best_metrics = tuned_model, tuned_metrics
            print(">>> Tuned model adopted as final model.")
        else:
            print(">>> Tuning did not improve F1 — keeping the untuned model.")

    # ---- Persist model + metadata -------------------------------------
    model_path = os.path.join(args.models_dir, "best_model.pkl")
    joblib.dump(best_model, model_path)

    importances = getattr(best_model, "feature_importances_", None)
    feature_importance = {}
    if importances is not None:
        feature_importance = dict(
            sorted(zip(FEATURE_NAMES, [float(i) for i in importances]),
                   key=lambda kv: kv[1], reverse=True)
        )

    results = {
        "RandomForest": rf_metrics,
        gb_metrics["model"]: gb_metrics,
    }
    if best_metrics["model"] not in results:
        results[best_metrics["model"]] = best_metrics
    results["selected_model"] = best_metrics["model"]

    metadata = {
        "feature_names": FEATURE_NAMES,
        "feature_importance": feature_importance,
        "results": results,
    }
    with open(os.path.join(args.models_dir, "metrics.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    with open(os.path.join(args.models_dir, "feature_names.json"), "w") as f:
        json.dump(FEATURE_NAMES, f, indent=2)

    print(f"\nSaved model      -> {model_path}")
    print(f"Saved metrics    -> {args.models_dir}/metrics.json")
    print(f"Saved feature schema -> {args.models_dir}/feature_names.json")

    print("\nTop 10 most important features:")
    for name, score in list(feature_importance.items())[:10]:
        print(f"  {name:28s} {score:.4f}")


if __name__ == "__main__":
    main()
