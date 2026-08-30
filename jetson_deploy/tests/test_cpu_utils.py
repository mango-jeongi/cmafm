"""CPU-only regression tests for paired preprocessing and YOLO postprocessing."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

DEPLOY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEPLOY_DIR))

from src.postprocess import non_max_suppression
from src.preprocess import preprocess_pair, scale_boxes


class PreprocessTests(unittest.TestCase):
    def test_aligned_pair_letterboxes_to_fixed_shape(self):
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        thermal = np.zeros((480, 640), dtype=np.uint8)
        rgb_tensor, thermal_tensor, meta = preprocess_pair(rgb, thermal)
        self.assertEqual(rgb_tensor.shape, (1, 3, 640, 640))
        self.assertEqual(thermal_tensor.shape, rgb_tensor.shape)
        self.assertEqual(rgb_tensor.dtype, np.float32)
        self.assertEqual(meta.pad, (0.0, 80.0))

    def test_misaligned_pair_is_rejected(self):
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        thermal = np.zeros((479, 640), dtype=np.uint8)
        with self.assertRaises(ValueError):
            preprocess_pair(rgb, thermal)

    def test_boxes_are_scaled_back_after_letterbox(self):
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        thermal = np.zeros((480, 640), dtype=np.uint8)
        _, _, meta = preprocess_pair(rgb, thermal)
        scaled = scale_boxes(np.asarray([[0, 80, 640, 560]], dtype=np.float32), meta)
        np.testing.assert_allclose(scaled, [[0, 0, 640, 480]])


class PostprocessTests(unittest.TestCase):
    def test_class_aware_nms_suppresses_overlapping_boxes(self):
        prediction = np.zeros((1, 2, 11), dtype=np.float32)
        prediction[0, 0, :5] = [320, 320, 100, 100, 0.9]
        prediction[0, 0, 5] = 0.8
        prediction[0, 1, :5] = [322, 322, 100, 100, 0.8]
        prediction[0, 1, 5] = 0.7
        detections = non_max_suppression(prediction)
        self.assertEqual(detections.shape, (1, 6))
        self.assertAlmostEqual(float(detections[0, 4]), 0.72, places=5)

    def test_different_classes_are_not_suppressed(self):
        prediction = np.zeros((1, 2, 11), dtype=np.float32)
        prediction[0, :, :5] = [[320, 320, 100, 100, 0.9], [320, 320, 100, 100, 0.9]]
        prediction[0, 0, 5] = 0.9
        prediction[0, 1, 6] = 0.9
        detections = non_max_suppression(prediction)
        self.assertEqual(detections.shape, (2, 6))


if __name__ == "__main__":
    unittest.main()

