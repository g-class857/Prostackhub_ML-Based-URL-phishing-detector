"""
Basic unit tests for the feature extraction module.
Run with: python -m pytest tests/ -v   (or) python tests/test_feature_extractor.py
"""

import os
import sys
import unittest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feature_extractor import FEATURE_NAMES, extract_features


class TestFeatureExtractor(unittest.TestCase):

    def test_all_features_present(self):
        feats = extract_features("https://example.com", lookup_domain_age=False)
        for name in FEATURE_NAMES:
            self.assertIn(name, feats.values)

    def test_https_flag(self):
        secure = extract_features("https://example.com", lookup_domain_age=False)
        insecure = extract_features("http://example.com", lookup_domain_age=False)
        self.assertEqual(secure.values["has_https"], 1)
        self.assertEqual(insecure.values["has_https"], 0)

    def test_ip_address_detection(self):
        feats = extract_features("http://192.168.1.1/login", lookup_domain_age=False)
        self.assertEqual(feats.values["has_ip_address"], 1)

        feats2 = extract_features("http://example.com/login", lookup_domain_age=False)
        self.assertEqual(feats2.values["has_ip_address"], 0)

    def test_at_symbol_detection(self):
        feats = extract_features("http://real.com@fake.com/login", lookup_domain_age=False)
        self.assertEqual(feats.values["has_at_symbol"], 1)

    def test_suspicious_words(self):
        feats = extract_features("http://example.com/verify-account", lookup_domain_age=False)
        self.assertEqual(feats.values["has_suspicious_words"], 1)

        feats2 = extract_features("http://example.com/products", lookup_domain_age=False)
        self.assertEqual(feats2.values["has_suspicious_words"], 0)

    def test_shortener_detection(self):
        feats = extract_features("http://bit.ly/abc123", lookup_domain_age=False)
        self.assertEqual(feats.values["is_shortened"], 1)

    def test_subdomain_count(self):
        feats = extract_features("http://a.b.c.example.com", lookup_domain_age=False)
        self.assertGreaterEqual(feats.values["num_subdomains"], 1)

    def test_bare_domain_input(self):
        # Should not raise even without a scheme
        feats = extract_features("example.com/login", lookup_domain_age=False)
        self.assertIn("url_length", feats.values)

    def test_malformed_urls_never_crash(self):
        malformed = [
            "http://[::1",
            "http://[gibberish]/path",
            "http://example.com]:80/x",
            None,
            "",
            "   ",
            "ht!tp://not a real url at all @@@",
            "http://user:pass@[::ffff:1.2.3.4]/path?q=1",
        ]
        for bad_url in malformed:
            feats = extract_features(bad_url, lookup_domain_age=False)
            for name in FEATURE_NAMES:
                self.assertIn(name, feats.values)

    def test_non_string_input_coerced(self):
        # Some real-world CSVs load stray numeric/NaN-like cells into the
        # url column; extract_features should coerce rather than crash.
        feats = extract_features(12345, lookup_domain_age=False)
        self.assertIn("url_length", feats.values)

    def test_as_vector_order(self):
        feats = extract_features("https://example.com", lookup_domain_age=False)
        vector = feats.as_vector()
        self.assertEqual(len(vector), len(FEATURE_NAMES))


if __name__ == "__main__":
    unittest.main()
