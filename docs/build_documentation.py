"""
build_documentation.py
-----------------------
Generates docs/Rule_Documentation.pdf: the deliverable documenting every
feature/rule the model uses, the reasoning behind it, and current model
performance. Reads live numbers from models/metrics.json so the PDF stays
in sync with whatever model was last trained.

Run: python docs/build_documentation.py
"""

import json
import os

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICS_PATH = os.path.join(BASE_DIR, "models", "metrics.json")
OUT_PATH = os.path.join(BASE_DIR, "docs", "Rule_Documentation.pdf")

INK = colors.HexColor("#0B1220")
CYAN = colors.HexColor("#2C6FB0")
MUTED = colors.HexColor("#55607A")
SAFE = colors.HexColor("#1E9E73")
DANGER = colors.HexColor("#C23B4E")
LIGHT_ROW = colors.HexColor("#F3F6FB")

FEATURE_DOCS = [
    ("url_length", "Length", "Total character count of the full URL.",
     "Phishing URLs are frequently padded with extra characters, tokens or "
     "encoded parameters to bury the real destination and evade quick visual "
     "inspection."),
    ("hostname_length", "Length", "Character count of the hostname only.",
     "A long hostname often signals a crafted lookalike domain (brand name + "
     "extra words) rather than a short, memorable legitimate one."),
    ("path_length", "Length", "Character count of the URL path.",
     "Deep, padded paths are used to bury fake login pages several levels "
     "into a disposable domain."),
    ("num_dots", "Structure", "Count of '.' characters in the URL.",
     "Excess dots usually indicate a deep, fake subdomain chain designed to "
     "make a URL look like it belongs to a trusted brand, e.g. "
     "paypal.com.verify-login.ru."),
    ("num_hyphens", "Structure", "Count of '-' characters in the URL.",
     "Hyphens are commonly inserted to combine a brand name with a "
     "suspicious keyword, e.g. paypal-secure-login.com."),
    ("num_underscores", "Structure", "Count of '_' characters in the URL.",
     "Underscores are rare in legitimate, human-facing brand domains; a high "
     "count can indicate auto-generated or obfuscated hosts."),
    ("num_slashes", "Structure", "Count of '/' characters in the URL.",
     "Extra path segments are used to nest fake pages and to make the URL "
     "look more 'official' or deeply structured than it is."),
    ("num_digits", "Structure", "Count of numeric characters in the URL.",
     "A high digit count (random IDs, encoded strings, IP octets) is atypical "
     "of clean, marketing-friendly legitimate brand URLs."),
    ("num_special_chars", "Structure", "Count of characters outside [A-Za-z0-9./:_-].",
     "Unusual symbols (%, =, &, unicode look-alikes) are uncommon in "
     "human-typed brand URLs and often appear in obfuscation attempts."),
    ("num_query_params", "Structure", "Number of '&'-separated query parameters.",
     "A large number of query parameters can be used to pass tracking or "
     "session-hijacking data, or simply to lengthen/obscure the URL."),
    ("has_at_symbol", "Host risk", "Whether the URL contains an '@' symbol.",
     "Browsers ignore everything before an '@' when resolving the host, so "
     "attackers use it to disguise the real destination behind a "
     "trusted-looking prefix (a classic, well-documented phishing trick)."),
    ("has_ip_address", "Host risk", "Whether the host is a raw IP address "
     "instead of a domain name.",
     "Legitimate, brand-operated sites virtually always use a registered "
     "domain name. A raw IP host is a strong phishing indicator."),
    ("has_https", "Transport", "Whether the URL uses the https:// scheme.",
     "Lack of HTTPS is a red flag, though this signal has weakened over time "
     "since free certificate authorities (e.g. Let's Encrypt) let phishing "
     "sites obtain HTTPS too — the model weighs it, but not exclusively."),
    ("has_http_in_path", "Structure", "Whether the literal string 'http' "
     "appears inside the URL path.",
     "Often indicates an embedded/open-redirect URL nested inside the "
     "visible one, a technique used to bounce victims through a trusted "
     "domain before landing on the phishing page."),
    ("num_subdomains", "Domain structure", "Number of subdomain labels "
     "before the registered domain.",
     "Multiple chained subdomains (login.secure.paypal.verify-x.ru) are a "
     "very common brand-impersonation pattern."),
    ("subdomain_length", "Domain structure", "Character count of the full "
     "subdomain portion.",
     "A long subdomain string usually means several impersonation keywords "
     "have been chained together."),
    ("is_shortened", "Host risk", "Whether the host matches a known URL "
     "shortening service (bit.ly, tinyurl.com, etc.).",
     "Shorteners hide the real destination until after a click and are "
     "heavily abused in phishing/smishing campaigns to bypass filters and "
     "curiosity-gate victims."),
    ("has_suspicious_words", "Content", "Whether the URL contains "
     "phishing-associated keywords (login, verify, secure, suspended, "
     "billing, wallet, etc.).",
     "These words are the backbone of social-engineering pretexts used to "
     "create urgency ('your account will be suspended') and are "
     "disproportionately present in confirmed phishing URLs."),
    ("has_double_slash_redirect", "Structure", "Whether '//' appears deep "
     "in the URL (position > 7), beyond the scheme separator.",
     "A '//' appearing after the host can indicate an open-redirect "
     "exploit used to launder a malicious destination through a trusted "
     "domain."),
    ("domain_age_days", "Metadata", "Domain age in days from WHOIS "
     "creation date (-1 if unavailable).",
     "The overwhelming majority of phishing domains are registered days to "
     "weeks before use and abandoned shortly after — domain age is one of "
     "the single strongest phishing predictors when available."),
    ("tld_length", "Domain structure", "Character count of the top-level "
     "domain / public suffix.",
     "Cheap, unusual TLDs (.tk, .ml, .top, .xyz) are disproportionately "
     "used for disposable phishing infrastructure and often differ in "
     "length/character profile from common TLDs."),
    ("digit_letter_ratio", "Structure", "Ratio of digit count to letter "
     "count in the URL.",
     "A high digit-to-letter ratio is atypical of legitimate, brand-focused "
     "domains and often appears in randomly generated or IP-based hosts."),
]

CATEGORY_COLORS = {
    "Length": colors.HexColor("#2C6FB0"),
    "Structure": colors.HexColor("#7A5CC2"),
    "Host risk": colors.HexColor("#C23B4E"),
    "Transport": colors.HexColor("#1E9E73"),
    "Domain structure": colors.HexColor("#B07B1E"),
    "Content": colors.HexColor("#C2662A"),
    "Metadata": colors.HexColor("#4A5772"),
}


def load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            return json.load(f)
    return {}


def build_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        "TitleBig", parent=styles["Title"], fontSize=26, textColor=INK,
        spaceAfter=6, fontName="Helvetica-Bold",
    ))
    styles.add(ParagraphStyle(
        "SubTitle", parent=styles["Normal"], fontSize=12, textColor=MUTED,
        spaceAfter=18,
    ))
    styles.add(ParagraphStyle(
        "H2", parent=styles["Heading2"], fontSize=15, textColor=INK,
        spaceBefore=18, spaceAfter=8, fontName="Helvetica-Bold",
    ))
    styles.add(ParagraphStyle(
        "Body", parent=styles["Normal"], fontSize=9.7, leading=14, textColor=INK,
    ))
    styles.add(ParagraphStyle(
        "BodyMuted", parent=styles["Normal"], fontSize=9.3, leading=13.5,
        textColor=MUTED,
    ))
    styles.add(ParagraphStyle(
        "FeatureName", parent=styles["Normal"], fontSize=10, leading=13,
        textColor=INK, fontName="Helvetica-Bold",
    ))
    styles.add(ParagraphStyle(
        "Cell", parent=styles["Normal"], fontSize=8.6, leading=11.5, textColor=INK,
    ))
    return styles


def category_badge(text):
    color = CATEGORY_COLORS.get(text, MUTED)
    return f'<font color="#{color.hexval()[2:]}"><b>{text.upper()}</b></font>'


def build_pdf():
    metrics = load_metrics()
    results = metrics.get("results", {})
    selected = results.get("selected_model", "N/A")
    best = results.get(selected, {})

    styles = build_styles()
    doc = SimpleDocTemplate(
        OUT_PATH, pagesize=LETTER,
        topMargin=0.75 * inch, bottomMargin=0.75 * inch,
        leftMargin=0.75 * inch, rightMargin=0.75 * inch,
        title="Phishing URL Detector — Rule Documentation",
    )
    story = []

    # ---- Cover / intro ----
    story.append(Paragraph("Phishing URL Detector", styles["TitleBig"]))
    story.append(Paragraph(
        "Rule &amp; Feature Documentation — how the classifier decides "
        "phishing vs. legitimate, and why each signal was chosen.",
        styles["SubTitle"],
    ))

    story.append(Paragraph("1. Overview", styles["H2"]))
    story.append(Paragraph(
        "This system classifies a URL as <b>phishing</b> or <b>legitimate</b> "
        "using only its structure and metadata &mdash; no page content, "
        "screenshots, or third-party blocklists are required at prediction "
        "time. A URL is converted into 22 numeric features, which are fed "
        "into a trained scikit-learn classifier (Random Forest, compared "
        "against XGBoost during training). The sections below document the "
        "exact rules/features used, the reasoning behind each one, current "
        "model performance, and the top factors the model actually relies on.",
        styles["Body"],
    ))

    # ---- Model performance ----
    story.append(Paragraph("2. Model Performance", styles["H2"]))
    if best:
        story.append(Paragraph(
            f"Selected model: <b>{selected}</b> (chosen as the higher-F1 "
            f"model between Random Forest and XGBoost, then tuned with "
            f"GridSearchCV optimized for precision to reduce false positives).",
            styles["Body"],
        ))
        perf_table_data = [
            ["Metric", "Score"],
            ["Accuracy", f"{best['accuracy']*100:.2f}%"],
            ["Precision", f"{best['precision']*100:.2f}%"],
            ["Recall", f"{best['recall']*100:.2f}%"],
            ["F1-score", f"{best['f1_score']*100:.2f}%"],
            ["ROC-AUC", f"{best['roc_auc']*100:.2f}%"],
        ]
        t = Table(perf_table_data, colWidths=[2.2 * inch, 2.2 * inch])
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), INK),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9.5),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_ROW]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D8DEE9")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(Spacer(1, 8))
        story.append(t)

        cm = best.get("confusion_matrix")
        if cm:
            story.append(Spacer(1, 14))
            story.append(Paragraph("Confusion matrix (held-out test set):", styles["Body"]))
            cm_data = [
                ["", "Predicted: Legitimate", "Predicted: Phishing"],
                ["Actual: Legitimate", str(cm[0][0]), str(cm[0][1])],
                ["Actual: Phishing", str(cm[1][0]), str(cm[1][1])],
            ]
            t2 = Table(cm_data, colWidths=[1.6 * inch, 1.9 * inch, 1.9 * inch])
            t2.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EDF1F8")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D8DEE9")),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]))
            story.append(Spacer(1, 6))
            story.append(t2)

        story.append(Spacer(1, 10))
        story.append(Paragraph(
            "<b>Note:</b> these numbers come from a training run on the "
            "bundled synthetic dataset (data/generate_dataset.py), used so "
            "the full pipeline can be built and demoed end-to-end without a "
            "Kaggle download. Retrain on the real Kaggle Phishing dataset "
            "and re-run src/validate_phishtank.py against a live PhishTank "
            "feed before treating these as production numbers "
            "(see README.md for both steps).",
            styles["BodyMuted"],
        ))
    else:
        story.append(Paragraph(
            "No trained model metrics found. Run `python src/train_model.py` "
            "and regenerate this document with `python docs/build_documentation.py`.",
            styles["Body"],
        ))

    story.append(PageBreak())

    # ---- Feature / rule documentation ----
    story.append(Paragraph("3. Feature &amp; Rule Reference", styles["H2"]))
    story.append(Paragraph(
        "Every feature below is computed by src/feature_extractor.py and is "
        "identical at training time and prediction time. Categories: "
        "<b>Length</b> (raw size signals), <b>Structure</b> (character/"
        "symbol patterns), <b>Host risk</b> (host-level red flags), "
        "<b>Domain structure</b> (subdomain/TLD parsing), <b>Content</b> "
        "(keyword signals), <b>Transport</b> (HTTPS), and <b>Metadata</b> "
        "(WHOIS domain age).",
        styles["Body"],
    ))
    story.append(Spacer(1, 10))

    importance = metrics.get("feature_importance", {})

    header = ["Feature", "Category", "What it measures / rationale", "Weight"]
    rows = [header]
    for fname, category, what, why in FEATURE_DOCS:
        cell = Paragraph(f"<b>{fname}</b>", styles["Cell"])
        cat_cell = Paragraph(category_badge(category), styles["Cell"])
        desc_cell = Paragraph(f"{what} {why}", styles["Cell"])
        weight = importance.get(fname)
        weight_txt = f"{weight:.3f}" if weight is not None else "—"
        weight_cell = Paragraph(weight_txt, styles["Cell"])
        rows.append([cell, cat_cell, desc_cell, weight_cell])

    t3 = Table(rows, colWidths=[1.1 * inch, 0.95 * inch, 3.65 * inch, 0.6 * inch], repeatRows=1)
    t3.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_ROW]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#D8DEE9")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t3)

    story.append(PageBreak())

    # ---- Decision logic ----
    story.append(Paragraph("4. How a Verdict Is Produced", styles["H2"]))
    steps = [
        "The submitted URL is normalized (a scheme is added if missing) and "
        "parsed into hostname, path and query components.",
        "All 22 features above are computed from that parsed URL (and, "
        "optionally, a live WHOIS lookup for domain age).",
        "The feature vector is passed to the trained classifier, which "
        "outputs a phishing probability between 0 and 1.",
        "A probability &ge; 0.50 is labeled <b>Phishing</b>; below that, "
        "<b>Legitimate</b>. The exact probability is always shown, not just "
        "the binary label, so users can gauge confidence.",
        "For explainability, the app re-checks which individual risk flags "
        "fired for that specific URL (e.g. IP host, '@' symbol, suspicious "
        "keyword) and pairs each with the model's global feature "
        "importance, producing a per-URL, human-readable factor list "
        "rather than a single opaque score.",
    ]
    story.append(ListFlowable(
        [ListItem(Paragraph(s, styles["Body"]), leftIndent=6) for s in steps],
        bulletType="1", start=1, leftIndent=14,
    ))

    story.append(Paragraph("5. Optimization for False Positives", styles["H2"]))
    story.append(Paragraph(
        "For a security tool, a false <i>phishing</i> verdict on a legitimate "
        "site (a false positive) erodes user trust faster than an occasional "
        "missed phishing URL. src/train_model.py reflects this by: (a) "
        "training with class_weight='balanced' / comparable regularization, "
        "(b) selecting the best model by F1-score rather than accuracy alone "
        "(accuracy alone can hide a poor minority-class score on a "
        "500/500-balanced test set), and (c) an optional GridSearchCV pass "
        "(<code>--tune</code>) that explicitly re-optimizes the winning "
        "model's hyperparameters for <b>precision</b>, then only adopts the "
        "tuned model if it does not regress overall F1.",
        styles["Body"],
    ))

    story.append(Paragraph("6. Validation Beyond the Training Distribution", styles["H2"]))
    story.append(Paragraph(
        "src/validate_phishtank.py scores the saved model against a live "
        "feed of confirmed, currently-active phishing URLs from PhishTank, "
        "paired with a sample of known-legitimate domains (e.g. the Tranco "
        "top-sites list). This checks whether the model generalizes to "
        "phishing campaigns it never saw during training, rather than just "
        "re-measuring performance on data drawn from the same generation "
        "process used to build the training set.",
        styles["Body"],
    ))

    story.append(Paragraph("7. Known Limitations", styles["H2"]))
    limits = [
        "Structural/lexical only: does not fetch or render the destination "
        "page, so it cannot catch phishing hosted on a compromised, "
        "otherwise-reputable domain.",
        "WHOIS domain-age requires network access and a responsive "
        "registrar; it degrades gracefully to 'unknown' (-1) rather than "
        "failing the whole prediction.",
        "The bundled dataset is synthetic; retrain on the real Kaggle "
        "dataset before production use (see README.md).",
        "HTTPS usage is a weakening signal on its own, since free "
        "certificate authorities let phishing sites obtain valid TLS too — "
        "the model treats it as one signal among many, not a determining one.",
    ]
    story.append(ListFlowable(
        [ListItem(Paragraph(s, styles["Body"]), leftIndent=6) for s in limits],
        bulletType="bullet", leftIndent=14,
    ))

    doc.build(story)
    print(f"Saved -> {OUT_PATH}")


if __name__ == "__main__":
    build_pdf()
