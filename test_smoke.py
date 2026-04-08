"""
Smoke tests for non-hardware depth estimation modules.

These tests verify that basic imports and utility functions work
without requiring Kinect hardware or a webcam.
"""
import unittest
import sys
import os
import numpy as np


class TestImports(unittest.TestCase):
    """Test that core dependencies can be imported."""

    def test_import_cv2(self):
        import cv2
        self.assertTrue(hasattr(cv2, 'VideoCapture'))

    def test_import_numpy(self):
        import numpy as np
        self.assertTrue(hasattr(np, 'array'))

    def test_import_ultralytics(self):
        try:
            from ultralytics import YOLO
        except ImportError:
            self.skipTest("ultralytics not installed")

    def test_import_matplotlib(self):
        import matplotlib.pyplot as plt
        self.assertTrue(hasattr(plt, 'figure'))


class TestDepthUtilities(unittest.TestCase):
    """Test utility functions that don't require hardware."""

    def test_bilateral_filter(self):
        """Test that bilateral filtering works on a dummy depth map."""
        import cv2
        dummy_depth = np.random.rand(480, 640).astype(np.float32)
        d = 7
        sigma_color = 100
        sigma_space = 100
        filtered = cv2.bilateralFilter(dummy_depth, d, sigma_color, sigma_space)
        self.assertEqual(filtered.shape, dummy_depth.shape)

    def test_colormap_application(self):
        """Test depth colormap rendering."""
        import cv2
        dummy_depth = np.random.rand(480, 640).astype(np.float32)
        depth_display = cv2.applyColorMap(
            (dummy_depth * 255).astype(np.uint8),
            cv2.COLORMAP_VIRIDIS
        )
        self.assertEqual(depth_display.shape, (480, 640, 3))

    def test_hstack_display(self):
        """Test that horizontal stacking of images works for display."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        depth_display = np.zeros((480, 640, 3), dtype=np.uint8)
        combined = np.hstack((img, depth_display))
        self.assertEqual(combined.shape, (480, 1280, 3))


class TestSkeletonConnections(unittest.TestCase):
    """Test skeleton connection definitions."""

    def test_connections_are_valid_tuples(self):
        """Verify SKELETON_CONNECTIONS format is correct."""
        SKELETON_CONNECTIONS = [
            (0, 1), (0, 2), (1, 3), (2, 4),
            (5, 6), (5, 11), (6, 12), (11, 12),
            (5, 7), (7, 9), (6, 8), (8, 10),
            (11, 13), (13, 15), (12, 14), (14, 16),
            (0, 5), (0, 6)
        ]
        for conn in SKELETON_CONNECTIONS:
            self.assertEqual(len(conn), 2)
            self.assertIsInstance(conn[0], int)
            self.assertIsInstance(conn[1], int)
            self.assertGreaterEqual(conn[0], 0)
            self.assertGreaterEqual(conn[1], 0)
            self.assertLessEqual(conn[0], 16)
            self.assertLessEqual(conn[1], 16)


if __name__ == '__main__':
    unittest.main()
