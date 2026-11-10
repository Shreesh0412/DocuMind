import os
import sys
import types
import unittest
from unittest.mock import patch

from src.config import get_config_value, get_settings


class TestSettings(unittest.TestCase):
    def test_environment_value_takes_precedence_over_streamlit_secret(self):
        fake_streamlit = types.SimpleNamespace(secrets={"GEMINI_API_KEY": "cloud-key"})
        with patch.dict(os.environ, {"GEMINI_API_KEY": "local-key"}, clear=True):
            with patch.dict(sys.modules, {"streamlit": fake_streamlit}):
                self.assertEqual(get_config_value("GEMINI_API_KEY"), "local-key")

    def test_reads_value_from_streamlit_secrets(self):
        fake_streamlit = types.SimpleNamespace(
            secrets={"GEMINI_API_KEY": "cloud-key", "DOCUMIND_DEMO_MODE": "true"}
        )
        with patch.dict(os.environ, {}, clear=True):
            with patch.dict(sys.modules, {"streamlit": fake_streamlit}):
                self.assertEqual(get_config_value("GEMINI_API_KEY"), "cloud-key")
                self.assertTrue(get_settings().demo_mode)

    def test_returns_default_when_streamlit_is_not_installed(self):
        with patch.dict(os.environ, {}, clear=True):
            with patch.dict(sys.modules, {"streamlit": None}):
                self.assertEqual(get_config_value("UNSET_DOCUMIND_SETTING", "fallback"), "fallback")

    def test_default_settings_are_safe_for_local_development(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = get_settings()
        self.assertFalse(settings.demo_mode)
        self.assertEqual(settings.max_pdf_file_size_mb, 20)
        self.assertEqual(settings.max_pdf_pages, 500)
        self.assertEqual(settings.max_batch_files, 5)
        self.assertEqual(settings.max_batch_size_mb, 50)
        self.assertEqual(settings.max_chunks_per_batch, 5000)

    def test_demo_mode_parses_true(self):
        with patch.dict(os.environ, {"DOCUMIND_DEMO_MODE": "true"}, clear=True):
            self.assertTrue(get_settings().demo_mode)

    def test_invalid_boolean_is_rejected(self):
        with patch.dict(os.environ, {"DOCUMIND_DEMO_MODE": "sometimes"}, clear=True):
            with self.assertRaisesRegex(ValueError, "must be true or false"):
                get_settings()

    def test_invalid_numeric_limit_is_rejected(self):
        with patch.dict(os.environ, {"MAX_PDF_PAGES": "0"}, clear=True):
            with self.assertRaisesRegex(ValueError, "MAX_PDF_PAGES"):
                get_settings()

    def test_individual_limit_cannot_exceed_batch_limit(self):
        with patch.dict(
            os.environ,
            {
                "MAX_PDF_FILE_SIZE_MB": "30",
                "MAX_BATCH_SIZE_MB": "20",
            },
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "cannot exceed"):
                get_settings()


if __name__ == "__main__":
    unittest.main()
