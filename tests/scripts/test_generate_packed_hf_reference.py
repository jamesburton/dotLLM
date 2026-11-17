"""Standard-library-only checks: python -m unittest discover -s tests/scripts -p test_generate_packed_hf_reference.py"""
import hashlib
import io
import unittest
from unittest.mock import patch

from generate_packed_hf_reference import producer_module


class ProducerIntegrityTests(unittest.TestCase):
    def test_verified_source_executes(self):
        source = b"value = 42"
        digest = hashlib.sha256(source).hexdigest()
        with patch("urllib.request.urlopen", return_value=io.BytesIO(source)):
            module, _, actual = producer_module("owner/repo", "commit", "file.py", digest)
        self.assertEqual(42, module.value)
        self.assertEqual(digest, actual)

    def test_changed_source_is_rejected_before_execution(self):
        source = b"raise AssertionError('unverified source executed')"
        with patch("urllib.request.urlopen", return_value=io.BytesIO(source)):
            with self.assertRaisesRegex(ValueError, "Producer SHA256 mismatch"):
                producer_module("owner/repo", "commit", "file.py", "0" * 64)
