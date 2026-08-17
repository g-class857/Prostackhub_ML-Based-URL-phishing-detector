"""
extract_legit_urls.py
----------------------
Filters a labeled dataset down to just the legitimate (non-phishing) rows
and writes them to a new CSV. Useful for building a --legit-sample file
for src/validate_phishtank.py, or for any other task that needs a clean
set of known-legitimate URLs pulled out of a bigger labeled dataset.

Usage:
    # Keep all original columns, just filter to label == 0
    python extract_legit_urls.py data/full_dataset.csv data/legit_only.csv

    # Only write the url column (matches what validate_phishtank.py expects)
    python extract_legit_urls.py data/full_dataset.csv data/legit_only.csv --url-only

    # If your legit rows are labeled differently (e.g. "good"/"legitimate" instead of 0)
    python extract_legit_urls.py data/full_dataset.csv data/legit_only.csv --legit-value legitimate

    # Cap how many rows get written (e.g. for a quick validation sample)
    python extract_legit_urls.py data/full_dataset.csv data/legit_only.csv --url-only --limit 1000
"""

import argparse

import pandas as pd

LEGIT_ALIASES = {"0", "0.0", "legitimate", "legit", "good", "benign"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", help="Path to the source labeled dataset")
    parser.add_argument("output_csv", help="Path to write the filtered legit-only CSV")
    parser.add_argument("--label-column", default="label",
                         help="Name of the label column (default: label)")
    parser.add_argument("--legit-value", default=None,
                         help="Value that marks a row as legitimate (default: "
                              "auto-detects 0 / 'legitimate' / 'good' / 'benign')")
    parser.add_argument("--url-only", action="store_true",
                         help="Write only the url column instead of all columns")
    parser.add_argument("--limit", type=int, default=None,
                         help="Optional cap on number of rows written")
    parser.add_argument("--shuffle", action="store_true",
                         help="Shuffle rows before applying --limit (otherwise "
                              "takes the first N as they appear in the file)")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    if args.label_column not in df.columns:
        raise SystemExit(f"[error] column '{args.label_column}' not found. "
                          f"Columns present: {list(df.columns)}")

    if args.legit_value is not None:
        legit_mask = df[args.label_column].astype(str).str.lower() == args.legit_value.lower()
    else:
        legit_mask = df[args.label_column].astype(str).str.lower().isin(LEGIT_ALIASES)

    legit_df = df[legit_mask].copy()

    if legit_df.empty:
        raise SystemExit(
            "[error] no rows matched as legitimate. Check --label-column / "
            "--legit-value, or inspect unique values with:\n"
            f"  python -c \"import pandas as pd; print(pd.read_csv('{args.input_csv}')"
            f"['{args.label_column}'].unique())\""
        )

    if args.shuffle:
        legit_df = legit_df.sample(frac=1, random_state=42)

    if args.limit:
        legit_df = legit_df.head(args.limit)

    if args.url_only:
        if "url" not in legit_df.columns:
            raise SystemExit(f"[error] no 'url' column found. Columns present: {list(df.columns)}")
        legit_df = legit_df[["url"]]

    legit_df.to_csv(args.output_csv, index=False)

    print(f"Input:  {args.input_csv} ({len(df):,} rows)")
    print(f"Output: {args.output_csv} ({len(legit_df):,} legitimate rows)")


if __name__ == "__main__":
    main()
