"""Minimal automated runtime smoke checks for the personal Streamlit dashboard."""
from __future__ import annotations

import os
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


APP = str(Path(__file__).with_name("app.py"))


class DashboardSmokeTests(unittest.TestCase):
    def test_missing_password_fails_closed(self):
        previous = os.environ.pop("DASHBOARD_PASSWORD", None)
        try:
            app = AppTest.from_file(APP, default_timeout=15).run()
            errors = [str(item.message) for item in app.error]
            self.assertTrue(
                any("Access locked" in msg for msg in errors),
                f"Expected password configuration lock, got: {errors}",
            )
        finally:
            if previous is not None:
                os.environ["DASHBOARD_PASSWORD"] = previous

    def test_configured_password_requires_login(self):
        previous = os.environ.get("DASHBOARD_PASSWORD")
        os.environ["DASHBOARD_PASSWORD"] = "test-only-placeholder-not-for-production"
        try:
            app = AppTest.from_file(APP, default_timeout=15).run()
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(any("Private dashboard password" in str(x.label) for x in app.text_input))
            self.assertFalse(any("betting intelligence" in str(x.value) for x in app.title))
        finally:
            if previous is None:
                os.environ.pop("DASHBOARD_PASSWORD", None)
            else:
                os.environ["DASHBOARD_PASSWORD"] = previous


if __name__ == "__main__":
    unittest.main()
