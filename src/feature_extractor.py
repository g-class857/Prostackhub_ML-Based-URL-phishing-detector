"""
feature_extractor.py
---------------------
Extracts structural, lexical and metadata features from a URL that are
predictive of phishing behaviour. Used identically at training time and
at inference time (Flask/Streamlit app) so the model always sees the
same feature schema.

Design notes
~~~~~~~~~~~~
* Domain parsing prefers `tldextract` (accurate public-suffix-aware
  parsing) but falls back to a small built-in parser if the package or
  network access to refresh its suffix list is unavailable. This keeps
  the app usable offline / in restricted environments.
* WHOIS domain-age lookups require an outbound network call and a
  registrar that answers WHOIS queries. They are wrapped in a timeout +
  try/except and return -1 (unknown) on failure so a single slow/blocked
  lookup never crashes a prediction request.
* Every feature is documented in docs/Rule_Documentation.pdf with the
  phishing-vs-legitimate rationale behind it.
"""

from __future__ import annotations

import ipaddress
import re
import socket
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict
from urllib.parse import urlparse

# --------------------------------------------------------------------------
# Optional dependencies — degrade gracefully if not installed / no network.
# --------------------------------------------------------------------------
try:
    import tldextract

    _TLDEXTRACT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TLDEXTRACT_AVAILABLE = False

try:
    import whois as pywhois  # python-whois

    _WHOIS_AVAILABLE = True
except ImportError:  # pragma: no cover
    _WHOIS_AVAILABLE = False


SUSPICIOUS_WORDS = [
    "login", "signin", "verify", "update", "secure", "account", "banking",
    "confirm", "webscr", "ebayisapi", "paypal", "password", "credential",
    "suspend", "urgent", "wallet", "unlock", "billing", "invoice",
]

SHORTENER_DOMAINS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd", "buff.ly",
    "adf.ly", "bit.do", "cutt.ly", "rebrand.ly", "shorte.st", "s.id",
}

# Ordered feature list — this exact order is what the model is trained on.
FEATURE_NAMES = [
    "url_length",
    "hostname_length",
    "path_length",
    "num_dots",
    "num_hyphens",
    "num_underscores",
    "num_slashes",
    "num_digits",
    "num_special_chars",
    "num_query_params",
    "has_at_symbol",
    "has_ip_address",
    "has_https",
    "has_http_in_path",
    "num_subdomains",
    "subdomain_length",
    "is_shortened",
    "has_suspicious_words",
    "has_double_slash_redirect",
    "domain_age_days",
    "tld_length",
    "digit_letter_ratio",
]


def _fallback_domain_parts(hostname: str) -> tuple[str, str, str]:
    """Very small fallback if tldextract isn't installed: naive split on dots.
    Not PSL-aware (won't handle co.uk perfectly) but keeps the pipeline
    functional offline.
    """
    parts = hostname.split(".")
    if len(parts) <= 2:
        return "", parts[0] if parts else "", ".".join(parts[1:])
    return ".".join(parts[:-2]), parts[-2], parts[-1]


def _get_domain_parts(hostname: str):
    if _TLDEXTRACT_AVAILABLE:
        ext = tldextract.extract(hostname)
        return ext.subdomain, ext.domain, ext.suffix
    return _fallback_domain_parts(hostname)


def _is_ip_address(hostname: str) -> bool:
    host = hostname.split(":")[0]
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def get_domain_age_days(hostname: str, timeout: int = 3) -> int:
    """Look up domain creation date via WHOIS. Returns -1 if unavailable
    (no network, WHOIS server blocked, rate-limited, or python-whois not
    installed). Callers should treat -1 as 'unknown', not 'zero'.
    """
    if not _WHOIS_AVAILABLE:
        return -1
    try:
        socket.setdefaulttimeout(timeout)
        w = pywhois.whois(hostname)
        created = w.creation_date
        if isinstance(created, list):
            created = created[0]
        if created is None:
            return -1
        return max((datetime.now() - created).days, 0)
    except Exception:
        return -1


def _safe_parse(raw_url: str):
    """Parse a URL into (scheme, hostname, path, query), tolerating the
    malformed / pathological URLs that show up in large real-world dumps
    (bad IPv6-style brackets, stray characters, multiple '@'/'#', etc.).
    `urllib.parse.urlparse` raises ValueError on some of these — rather
    than crash a multi-million-row training run on one bad row, fall back
    to a tolerant regex split, and as a last resort treat the whole string
    as an opaque path so every feature still gets *some* value.
    """
    try:
        p = urlparse(raw_url)
        return p.scheme, (p.hostname or ""), (p.path or ""), (p.query or "")
    except Exception:
        pass

    try:
        m = re.match(r"^([a-zA-Z][a-zA-Z0-9+.\-]*)://([^/?#]*)([^?#]*)(?:\?([^#]*))?", raw_url)
        if m:
            scheme = m.group(1)
            authority = m.group(2) or ""
            path = m.group(3) or ""
            query = m.group(4) or ""
            hostname = authority.split("@")[-1].split(":")[0].strip("[]")
            return scheme, hostname, path, query
    except Exception:
        pass

    return "", "", raw_url, ""


def _get_domain_parts_safe(hostname: str):
    try:
        return _get_domain_parts(hostname)
    except Exception:
        return "", "", ""


def _is_ip_address_safe(hostname: str) -> bool:
    try:
        return _is_ip_address(hostname)
    except Exception:
        return False


@dataclass
class URLFeatures:
    values: Dict[str, float] = field(default_factory=dict)
    explanation: Dict[str, str] = field(default_factory=dict)

    def as_vector(self):
        return [self.values[name] for name in FEATURE_NAMES]


def extract_features(url, lookup_domain_age: bool = True) -> URLFeatures:
    """Extract the full feature set for a single URL.

    Parameters
    ----------
    url: raw URL string as typed/pasted by the user.
    lookup_domain_age: if False, skips the (network-bound) WHOIS call and
        reports -1. Useful for fast batch training on large datasets.

    This function is intentionally defensive: on any parsing failure it
    still returns a complete, correctly-shaped URLFeatures rather than
    raising, so a single malformed row in a multi-million-row dataset
    never aborts a training run. Malformed input degrades to
    string-only features (length/character counts still computed
    correctly) with parse-dependent fields (hostname/IP/subdomain/TLD)
    defaulted safely.
    """
    if url is None:
        raw_url = ""
    else:
        raw_url = str(url).strip()

    if raw_url and not re.match(r"^[a-zA-Z][a-zA-Z0-9+.\-]*://", raw_url):
        raw_url = "http://" + raw_url  # allow bare "example.com" input

    scheme, hostname, path, query = _safe_parse(raw_url)

    subdomain, domain, suffix = _get_domain_parts_safe(hostname)
    num_subdomains = len([s for s in subdomain.split(".") if s]) if subdomain else 0

    digits = sum(c.isdigit() for c in raw_url)
    letters = sum(c.isalpha() for c in raw_url)
    special_chars = len(re.findall(r"[^a-zA-Z0-9./:_\-]", raw_url))

    domain_age = get_domain_age_days(hostname) if (lookup_domain_age and hostname) else -1

    values = {
        "url_length": len(raw_url),
        "hostname_length": len(hostname),
        "path_length": len(path),
        "num_dots": raw_url.count("."),
        "num_hyphens": raw_url.count("-"),
        "num_underscores": raw_url.count("_"),
        "num_slashes": raw_url.count("/"),
        "num_digits": digits,
        "num_special_chars": special_chars,
        "num_query_params": len(query.split("&")) if query else 0,
        "has_at_symbol": int("@" in raw_url),
        "has_ip_address": int(_is_ip_address_safe(hostname)),
        "has_https": int(scheme.lower() == "https"),
        "has_http_in_path": int("http" in path.lower()),
        "num_subdomains": num_subdomains,
        "subdomain_length": len(subdomain),
        "is_shortened": int(hostname.lower() in SHORTENER_DOMAINS),
        "has_suspicious_words": int(
            any(w in raw_url.lower() for w in SUSPICIOUS_WORDS)
        ),
        "has_double_slash_redirect": int(raw_url.rfind("//") > 7),
        "domain_age_days": domain_age,
        "tld_length": len(suffix),
        "digit_letter_ratio": round(digits / max(letters, 1), 4),
    }

    explanation = {
        "url_length": "Very long URLs are often used to hide the real destination or obfuscate a fake domain.",
        "hostname_length": "Unusually long hostnames can indicate a crafted lookalike domain.",
        "num_dots": "Excess dots usually mean deep/fake subdomain chains (e.g. paypal.com.verify.ru).",
        "num_hyphens": "Phishing domains frequently insert hyphens to mimic brand names (e.g. paypal-secure-login.com).",
        "num_slashes": "Extra path segments can be used to bury a fake login page.",
        "num_special_chars": "Unusual symbols are uncommon in legitimate, human-typed brand URLs.",
        "has_at_symbol": "'@' lets browsers ignore everything before it, hiding the true host — a classic phishing trick.",
        "has_ip_address": "Legitimate sites almost always use a domain name, not a raw IP address.",
        "has_https": "Lack of HTTPS/TLS is a red flag, though many phishing sites now use free HTTPS certs too.",
        "num_subdomains": "Many chained subdomains are commonly used to impersonate a trusted brand.",
        "is_shortened": "URL shorteners can hide the real destination and are commonly abused in phishing campaigns.",
        "has_suspicious_words": "Words like 'verify', 'login', 'secure', 'suspend' are common phishing/social-engineering triggers.",
        "has_double_slash_redirect": "A '//' appearing deep in the URL can indicate an open-redirect trick.",
        "domain_age_days": "Newly registered domains (days/weeks old) are disproportionately used for phishing.",
        "digit_letter_ratio": "A high ratio of digits to letters is atypical of legitimate brand domains.",
    }

    return URLFeatures(values=values, explanation=explanation)


if __name__ == "__main__":
    for test_url in [
        "https://www.google.com/search?q=test",
        "http://192.168.1.1/paypal-login/verify-account",
        "http://secure-paypal-login.account-verify.ru/signin",
    ]:
        feats = extract_features(test_url, lookup_domain_age=False)
        print(test_url)
        for k in FEATURE_NAMES:
            print(f"  {k}: {feats.values[k]}")
        print()
