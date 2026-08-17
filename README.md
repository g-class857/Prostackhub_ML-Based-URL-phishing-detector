# 🛡️ Phishing URL Detector (Machine Learning)

Classify URLs as **phishing** or **legitimate** from their structure alone —
no page content or screenshots needed — using a scikit-learn model served
through a Flask or Streamlit web app with per-prediction explainability.

```
Paste a URL  →  extract 22 structural features  →  Random Forest / XGBoost
            →  phishing probability + plain-English "why"
```

## Features analyzed

| Category | Examples |
|---|---|
| Length & structure | URL/hostname/path length, dot/slash/hyphen counts |
| Host risk signals | raw IP address as host, `@` symbol, URL shorteners |
| Content signals | suspicious keywords (`verify`, `login`, `suspend`...), digit/letter ratio |
| Domain structure | subdomain count & length, TLD length |
| Transport & age | HTTPS usage, WHOIS domain age (days) |

See `docs/Rule_Documentation.pdf` for the full list with the phishing-vs-legitimate
rationale behind every feature.

## Project layout

```
phishing-url-detector/
├── data/
│   ├── generate_dataset.py     # builds training data (synthetic OR real Kaggle CSV)
│   └── phishing_dataset.csv    # generated training set
├── src/
│   ├── feature_extractor.py    # the 22-feature extraction pipeline (shared by training + app)
│   ├── train_model.py          # trains + compares RandomForest vs XGBoost, saves best model
│   └── validate_phishtank.py   # validates the saved model against a live PhishTank feed
├── models/
│   ├── best_model.pkl          # trained classifier (joblib)
│   ├── metrics.json            # accuracy/precision/recall/F1/ROC-AUC + feature importances
│   └── feature_names.json      # feature schema, kept in lockstep with the model
├── app/
│   ├── flask_app.py            # Flask web app (templates/ + static/)
│   └── streamlit_app.py        # Streamlit alternative UI
├── tests/
│   └── test_feature_extractor.py
├── docs/
│   └── Rule_Documentation.pdf  # deliverable: full feature/rule documentation
└── requirements.txt
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate   # optional but recommended
pip install -r requirements.txt

# 1. Build the training dataset
python data/generate_dataset.py --n-per-class 3000

# 2. Train + evaluate Random Forest vs XGBoost, save the best model
python src/train_model.py --data data/phishing_dataset.csv --tune

# 3a. Run the Flask app
python app/flask_app.py            # -> http://127.0.0.1:5000

# 3b. ...or run the Streamlit app instead
streamlit run app/streamlit_app.py
```

## Using the real Kaggle dataset (recommended for production)

This repo ships with a **synthetic dataset generator**
(`data/generate_dataset.py`) so the full pipeline runs end-to-end out of the
box, without requiring a Kaggle account or internet access to reproduce.
For a production-grade model, swap in real data:

1. Download a Kaggle phishing URL dataset, e.g.
   [Phishing Site URLs](https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls)
   or [Phishing Dataset](https://www.kaggle.com/datasets/eswarchandt/phishing-dataset).
2. Save it as `data/raw_kaggle_dataset.csv` with columns `url,label`
   (`label` = 1 for phishing, 0 for legitimate).
3. Rebuild the dataset:
   ```bash
   python data/generate_dataset.py --source kaggle --input data/raw_kaggle_dataset.csv
   ```
4. Re-run `python src/train_model.py --data data/phishing_dataset.csv --tune`.

## Validating against PhishTank

`src/validate_phishtank.py` scores the saved model against a **live feed of
confirmed phishing URLs** (PhishTank) paired with a known-legitimate sample
(e.g. Tranco top sites), to check generalization beyond the training
distribution:

```bash
python src/validate_phishtank.py \
    --phishing-feed data/phishtank_feed.csv \
    --legit-sample data/tranco_top_1000.csv
```

Get a free PhishTank feed/API key at https://www.phishtank.com/api_register.php.

## Model comparison & optimization

`train_model.py`:
- Trains **RandomForestClassifier** and **XGBoost** (falls back to
  scikit-learn's `GradientBoostingClassifier` automatically if `xgboost`
  isn't installed, so the pipeline never breaks in restricted environments).
- Evaluates both with **Accuracy, Precision, Recall, F1-score, ROC-AUC and
  a confusion matrix**.
- Picks the higher-F1 model, then (with `--tune`) runs a small
  `GridSearchCV` **optimized for precision** on the winner to cut down
  false positives — the costliest error type for a security tool, since a
  false "phishing" verdict on a legitimate site erodes user trust.
- Persists `models/best_model.pkl`, `models/metrics.json` (used by both web
  apps to render live stats + explainability) and `models/feature_names.json`.

### Training on large datasets (500k+ rows)

The feature-extraction loop prints progress every 100k rows and typically
runs at ~25–30k URLs/sec, so a multi-million-row dataset takes on the order
of a minute or two for feature extraction — that step is not the
bottleneck. Two flags help on real Kaggle-scale data:

- `--max-rows N` — subsamples (stratified by label) down to N rows before
  training, for a quick smoke-test run before committing to a full pass.
- `--tune-sample-size N` (default 150,000) — `--tune`'s `GridSearchCV`
  refits the model several times per grid point, which is impractically
  slow directly on a multi-million-row training set. Above this size, the
  hyperparameter *search* runs on a stratified subsample, and only the
  *winning* configuration is refit on the full training set — giving you
  tuned hyperparameters without paying for a full grid search on the
  entire dataset.

Malformed or missing URLs are handled defensively rather than crashing the
run: rows with a null/empty `url` or `label` are dropped up front (with a
count printed), and `feature_extractor.extract_features()` never raises —
even on pathological input (broken IPv6-style brackets, stray characters,
non-string cells) it degrades to string-only features instead of aborting
a multi-million-row training run over one bad row.

## Explainability

Every prediction returned by the app includes a **Signal Log** / factor
table: the top contributing features for that specific URL, each paired
with the model's global feature importance and a plain-English reason
(e.g. *"Uses a raw IP address instead of a domain name"*). This is built
from `model.feature_importances_` combined with which risk flags actually
fired on the submitted URL — not just a static global ranking — so the
explanation is specific to what the user just scanned.

## Known limitations (be transparent with end users)

- **WHOIS domain-age lookups require outbound network access** and a
  responsive registrar; they're skipped during training (would be
  impractically slow across thousands of rows) and time out gracefully to
  `-1` ("unknown") in the app if the WHOIS server is unreachable or the
  domain is privacy-protected.
- **`tldextract`'s public-suffix parsing is more accurate than the built-in
  fallback parser** used automatically when the package isn't installed;
  install it for best subdomain/TLD accuracy on tricky suffixes like
  `.co.uk`.
- The shipped `phishing_dataset.csv` is **synthetic** (see above) — treat
  the reported metrics as a pipeline sanity check, not a production
  benchmark, until retrained on the real Kaggle dataset + validated on
  PhishTank.
- This is a **structural/lexical** classifier only — it does not fetch or
  render the destination page, so it cannot catch phishing pages hosted on
  otherwise-reputable, compromised domains.

## Testing

```bash
python -m unittest tests/test_feature_extractor.py -v
```
