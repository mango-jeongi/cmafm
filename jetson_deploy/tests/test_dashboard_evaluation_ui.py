"""Verify that completed TensorRT accuracy results render in the main dashboard."""

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


REPO_ROOT = Path(__file__).resolve().parents[2]


class DashboardEvaluationUITest(unittest.TestCase):
    def test_evaluation_tab_displays_completed_metrics(self):
        app = AppTest.from_file(
            str(REPO_ROOT / "src" / "fusion" / "dashboard.py"),
            default_timeout=60,
        )
        app.run()

        self.assertFalse(app.exception)
        self.assertIn("📈 Model Evaluation", [tab.label for tab in app.tabs])
        metrics = {metric.label: metric.value for metric in app.metric}
        self.assertEqual(metrics["mAP @ 0.5"], "85.60%")
        self.assertEqual(metrics["mAP @ 0.5:0.95"], "56.62%")
        self.assertEqual(metrics["Mean Precision"], "89.00%")
        self.assertEqual(metrics["Mean Recall"], "79.91%")

        selector = next(
            selectbox for selectbox in app.selectbox
            if selectbox.label == "Evaluation Dataset"
        )
        self.assertEqual(
            selector.options,
            ["M3FD validation (6 classes)", "FLIR aligned test (People + Car)"],
        )
        selector.select("FLIR aligned test (People + Car)").run()
        self.assertFalse(app.exception)
        metrics = {metric.label: metric.value for metric in app.metric}
        self.assertEqual(metrics["mAP @ 0.5"], "89.67%")
        self.assertEqual(metrics["mAP @ 0.5:0.95"], "53.97%")
        self.assertEqual(metrics["Mean Precision"], "87.30%")
        self.assertEqual(metrics["Mean Recall"], "82.16%")


if __name__ == "__main__":
    unittest.main()
