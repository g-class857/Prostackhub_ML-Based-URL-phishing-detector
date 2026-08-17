"""
generate_dataset.py
--------------------
Builds a labeled training set of (url, label) pairs, label = 1 (phishing)
or 0 (legitimate).

IMPORTANT — read before using in production
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This script *synthesizes* a realistic-but-artificial dataset using the
same structural patterns documented in the Kaggle "Phishing Site URLs"
dataset (hyphenated brand impersonation, IP-address hosts, excessive
subdomains, URL shorteners, long obfuscated paths, etc.) so that the
full pipeline (feature extraction -> training -> evaluation -> app) can
be built, run, and demoed end-to-end without requiring an internet
download inside this environment.

For a production model you should replace this step with the real
dataset:

    1. Download from Kaggle, e.g.:
       https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls
       or https://www.kaggle.com/datasets/eswarchandt/phishing-dataset
    2. Save as data/raw_kaggle_dataset.csv with columns: url,label
       (label = 1 for phishing, 0 for legitimate)
    3. Run:  python data/generate_dataset.py --source kaggle \
                 --input data/raw_kaggle_dataset.csv
    4. Optionally validate/augment with live PhishTank feed:
       http://data.phishtank.com/data/online-valid.csv
       (requires a free PhishTank API key for the live feed)

Both paths write the same normalized data/phishing_dataset.csv consumed
by src/train_model.py, so nothing downstream needs to change.
"""

import argparse
import csv
import random

random.seed(42)

LEGIT_BRANDS = [
    "google", "youtube", "facebook", "amazon", "wikipedia", "twitter",
    "instagram", "linkedin", "netflix", "microsoft", "apple", "github",
    "reddit", "spotify", "yahoo", "ebay", "paypal", "dropbox", "adobe",
    "zoom", "salesforce", "walmart", "target", "chase", "wellsfargo",
    "bankofamerica", "irs", "usps", "fedex", "ups",
]
LEGIT_TLDS = ["com", "org", "net", "gov", "edu", "co.uk", "io"]
LEGIT_PATHS = [
    "", "/", "/about", "/products", "/help/support", "/login",
    "/account/settings", "/news/latest", "/search?q=weather",
    "/watch?v=abc123", "/blog/2026/updates", "/docs/api/v2",
    "/store/category/electronics", "/careers", "/contact-us",
]

BRAND_TARGETS = [
    "paypal", "apple", "amazon", "microsoft", "google", "netflix", "chase",
    "wellsfargo", "bankofamerica", "facebook", "instagram", "irs", "usps",
    "dhl", "fedex", "coinbase", "binance", "outlook", "office365", "adobe",
]
SUSPICIOUS_TOKENS = [
    "login", "signin", "verify", "verification", "update", "secure",
    "account", "confirm", "suspended", "unlock", "billing", "invoice",
    "webscr", "security-alert", "reset-password", "wallet-recovery",
]
SUSPICIOUS_TLDS = ["ru", "tk", "ml", "ga", "cf", "top", "xyz", "click", "gq", "info"]
SHORTENERS = ["bit.ly", "tinyurl.com", "t.co", "goo.gl", "is.gd", "cutt.ly"]


def random_hex(n):
    return "".join(random.choice("0123456789abcdef") for _ in range(n))


def random_ip():
    return ".".join(str(random.randint(1, 255)) for _ in range(4))


def make_legit_url():
    brand = random.choice(LEGIT_BRANDS)
    tld = random.choice(LEGIT_TLDS)
    path = random.choice(LEGIT_PATHS)
    use_www = random.random() < 0.5
    host = f"{'www.' if use_www else ''}{brand}.{tld}"
    # Realistic noise: ~4% of legitimate sites still lack HTTPS (misconfigured
    # or legacy), and a small share use marketing subdomains/hyphens that
    # overlap with phishing-style patterns. This keeps the classes from being
    # perfectly separable, which is what real-world traffic looks like.
    scheme = "http" if random.random() < 0.04 else "https"
    if random.random() < 0.08:
        host = f"promo-{brand}.{tld}"  # legit marketing subdomain w/ hyphen
    if random.random() < 0.05:
        path += f"?ref={random_hex(6)}"  # tracking params add digits/specials
    return f"{scheme}://{host}{path}"


def make_phishing_url():
    style = random.choice([
        "hyphen_brand", "ip_host", "long_subdomain", "shortener",
        "suspicious_tld", "at_symbol", "long_obfuscated_path",
    ])
    brand = random.choice(BRAND_TARGETS)
    token = random.choice(SUSPICIOUS_TOKENS)

    if style == "hyphen_brand":
        fake_tld = random.choice(SUSPICIOUS_TLDS + LEGIT_TLDS)
        host = f"{brand}-{token}-{random_hex(4)}.{fake_tld}"
        url = f"http://{host}/{token}"

    elif style == "ip_host":
        ip = random_ip()
        url = f"http://{ip}/{brand}/{token}.php"

    elif style == "long_subdomain":
        fake_tld = random.choice(SUSPICIOUS_TLDS)
        sub = f"{brand}.{token}.secure-{random_hex(3)}"
        url = f"http://{sub}.{random_hex(5)}.{fake_tld}/{token}"

    elif style == "shortener":
        short = random.choice(SHORTENERS)
        url = f"http://{short}/{random_hex(6)}"

    elif style == "suspicious_tld":
        fake_tld = random.choice(SUSPICIOUS_TLDS)
        url = f"http://{brand}{token}.{fake_tld}/{token}/{random_hex(4)}"

    elif style == "at_symbol":
        url = f"http://{brand}.{token}@{random_ip()}/{token}"

    else:  # long_obfuscated_path
        fake_tld = random.choice(SUSPICIOUS_TLDS)
        segments = "/".join(random_hex(6) for _ in range(4))
        url = f"http://{brand}-{token}.{fake_tld}/{segments}?id={random_hex(8)}"

    # Realistic noise: ~15% of modern phishing kits use free HTTPS certs
    # (Let's Encrypt etc.), so "has_https" alone should not be a perfect tell.
    if random.random() < 0.15:
        url = url.replace("http://", "https://", 1)
    return url


def synth_dataset(n_per_class=2500, label_noise=0.02):
    """label_noise: fraction of labels randomly flipped to simulate
    mislabeled/ambiguous real-world samples (near-duplicate brand domains,
    compromised legitimate sites, etc.) so the model can't reach a
    trivial 100% on held-out data."""
    rows = []
    for _ in range(n_per_class):
        rows.append([make_legit_url(), 0])
    for _ in range(n_per_class):
        rows.append([make_phishing_url(), 1])
    random.shuffle(rows)
    n_flip = int(len(rows) * label_noise)
    for idx in random.sample(range(len(rows)), n_flip):
        rows[idx][1] = 1 - rows[idx][1]
    return [tuple(r) for r in rows]


def load_kaggle_csv(path, phishing_label=None):
    """Load one raw dataset CSV and normalize it to (url, label) rows.

    Handles the column-name variants seen across different Kaggle phishing
    datasets (url/URL/URLs, label/Label/Result/status/Type/class) and
    common label encodings (1/0, "phishing"/"legitimate", "bad"/"good",
    -1/1, etc). Datasets differ in which value means "phishing" — most use
    1 = phishing, but some (e.g. the classic UCI-style dumps redistributed
    on Kaggle) use -1 = phishing, 1 = legitimate. If auto-detection can't
    tell (ambiguous 0/1-only files with no other clue), pass
    --phishing-label to force it for that file.
    """
    URL_COLS = ["url", "URL", "URLs", "urls", "Domain", "domain", "website", "link"]
    LABEL_COLS = ["label", "Label", "Result", "result", "status", "Status",
                  "Type", "type", "class", "Class", "phishing", "CLASS_LABEL"]

    PHISHING_WORDS = {"phishing", "bad", "malicious", "malware", "spam", "1", "-1", "yes", "true"}
    LEGIT_WORDS = {"legitimate", "good", "benign", "safe", "0", "no", "false", "ham"}

    rows = []
    skipped = 0
    with open(path, newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        url_col = next((c for c in URL_COLS if c in fieldnames), None)
        label_col = next((c for c in LABEL_COLS if c in fieldnames), None)

        if url_col is None or label_col is None:
            print(f"  [skip] {path}: couldn't find url/label columns "
                  f"(found columns: {fieldnames}). Rename columns to "
                  f"'url' and 'label', or add them to URL_COLS/LABEL_COLS "
                  f"in this script.")
            return []

        # Figure out this file's label encoding. If --phishing-label was
        # passed, force it; otherwise infer from the value text, falling
        # back to "1 means phishing" (the majority convention).
        raw_values = set()
        sample_rows = []
        for r in reader:
            sample_rows.append(r)
            raw_values.add(str(r.get(label_col, "")).strip().lower())
        if phishing_label is not None:
            phishing_marker = str(phishing_label).strip().lower()
        elif raw_values & PHISHING_WORDS and raw_values & LEGIT_WORDS:
            phishing_marker = None  # word-based, handled per-row below
        elif "-1" in raw_values:
            phishing_marker = "-1"  # classic UCI encoding: -1 = phishing
        else:
            phishing_marker = "1"  # common convention: 1 = phishing

        for r in sample_rows:
            url = (r.get(url_col) or "").strip()
            label_raw = str(r.get(label_col, "")).strip().lower()
            if not url or not label_raw:
                skipped += 1
                continue
            if phishing_marker is None:
                if label_raw in PHISHING_WORDS:
                    label = 1
                elif label_raw in LEGIT_WORDS:
                    label = 0
                else:
                    skipped += 1
                    continue
            else:
                label = 1 if label_raw == phishing_marker else 0
            if not url.lower().startswith(("http://", "https://")):
                url = "http://" + url
            rows.append((url, label))

    n_phish = sum(1 for _, l in rows if l == 1)
    print(f"  [ok] {path}: {len(rows)} rows loaded "
          f"({n_phish} phishing / {len(rows) - n_phish} legitimate), "
          f"{skipped} skipped, columns url='{url_col}' label='{label_col}'")
    return rows


def discover_kaggle_csvs(inputs):
    """Expand a list of --input args (files, directories, or glob patterns)
    into a flat, deduped list of CSV file paths."""
    paths = []
    for item in inputs:
        if os.path.isdir(item):
            paths.extend(sorted(glob.glob(os.path.join(item, "*.csv"))))
        elif any(ch in item for ch in "*?["):
            paths.extend(sorted(glob.glob(item)))
        else:
            paths.append(item)
    # de-dupe while preserving order, drop our own output file if it's caught by a glob
    seen, unique = set(), []
    for p in paths:
        ap = os.path.abspath(p)
        if ap not in seen and os.path.basename(p) != "phishing_dataset.csv":
            seen.add(ap)
            unique.append(p)
    return unique


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["synthetic", "kaggle"], default="synthetic")
    parser.add_argument(
        "--input", nargs="+", default=None,
        help="One or more raw dataset CSVs, directories, or glob patterns. "
             "E.g. --input data/raw/  or  --input data/kaggle1.csv data/kaggle2.csv "
             "or  --input 'data/raw/*.csv'",
    )
    parser.add_argument(
        "--phishing-label", default=None,
        help="Force which raw label value means 'phishing' for ALL input "
             "files (e.g. '1' or '-1'). Leave unset to auto-detect per file.",
    )
    parser.add_argument("--n-per-class", type=int, default=2500)
    parser.add_argument("--output", default="data/phishing_dataset.csv")
    parser.add_argument(
        "--no-dedupe", action="store_true",
        help="Keep duplicate URLs across/within files instead of dropping them.",
    )
    args = parser.parse_args()

    if args.source == "kaggle":
        if not args.input:
            raise SystemExit(
                "--input is required when --source kaggle "
                "(a file, a directory of CSVs, or a glob pattern)"
            )
        csv_paths = discover_kaggle_csvs(args.input)
        if not csv_paths:
            raise SystemExit(f"No CSV files found for --input {args.input}")
        print(f"Found {len(csv_paths)} CSV file(s):")

        rows = []
        for path in csv_paths:
            rows.extend(load_kaggle_csv(path, phishing_label=args.phishing_label))

        total_before = len(rows)
        if not args.no_dedupe:
            seen_urls = set()
            deduped = []
            for url, label in rows:
                key = url.strip().lower().rstrip("/")
                if key in seen_urls:
                    continue
                seen_urls.add(key)
                deduped.append((url, label))
            rows = deduped
            print(f"Merged {len(csv_paths)} file(s): {total_before} rows -> "
                  f"{len(rows)} after de-duplication.")
        else:
            print(f"Merged {len(csv_paths)} file(s): {total_before} rows (no de-duplication).")

        n_phish = sum(1 for _, l in rows if l == 1)
        n_legit = len(rows) - n_phish
        print(f"Combined dataset: {n_phish} phishing / {n_legit} legitimate")
        if n_phish and n_legit and (max(n_phish, n_legit) / min(n_phish, n_legit) > 3):
            print("  [warn] classes are quite imbalanced — train_model.py uses "
                  "class_weight='balanced', but consider capping the majority "
                  "class if this feels too skewed for your use case.")
        random.shuffle(rows)
    else:
        rows = synth_dataset(args.n_per_class)
        print(f"Generated {len(rows)} synthetic rows "
              f"({args.n_per_class} legitimate / {args.n_per_class} phishing).")
        print("NOTE: replace with the real Kaggle dataset for production use — "
              "see the module docstring for instructions.")

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["url", "label"])
        writer.writerows(rows)
    print(f"Saved dataset -> {args.output} ({len(rows)} total rows)")


if __name__ == "__main__":
    main()
