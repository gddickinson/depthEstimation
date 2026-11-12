"""
WARNING: This module is experimental / work-in-progress.
SLAM-like mapping using Kinect depth data is incomplete and
may not produce reliable results. Use at your own risk.
"""

import numpy as np
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel, QTextEdit
from PyQt5.QtCore import QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from kinect_wrapper import PyKinectCapture
from kinect_depth.processing import depth_to_points
from ultralytics import YOLO
import traceback

class KinectSLAMApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kinect SLAM-like Mapping")
        self.setGeometry(100, 100, 800, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.start_button = QPushButton("Start Mapping")
        self.start_button.clicked.connect(self.start_mapping)
        layout.addWidget(self.start_button)

        self.status_label = QLabel("Status: Ready")
        layout.addWidget(self.status_label)

        self.image_label = QLabel()
        layout.addWidget(self.image_label)

        self.debug_text = QTextEdit()
        self.debug_text.setReadOnly(True)
        layout.addWidget(self.debug_text)

        self.kinect = PyKinectCapture()
        try:
            self.kinect.init()
            self.log_debug("Kinect initialized successfully")
        except Exception as e:
            self.log_debug(f"Error initializing Kinect: {str(e)}")

        self.yolo_model = YOLO('yolov8n.pt')

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_map)

        self.frame_count = 0
        self.max_points = 1000000
        self.points = np.zeros((0, 3))
        self.colors = np.zeros((0, 3))

    def log_debug(self, message):
        self.debug_text.append(message)
        print(message)

    def start_mapping(self):
        self.timer.start(100)
        self.status_label.setText("Status: Mapping")
        self.log_debug("Mapping started")

    def update_map(self):
        try:
            rgb, depth = self.kinect.get_frames()
            if rgb is None or depth is None:
                self.log_debug("Failed to get frames from Kinect")
                return

            self.log_debug(f"Received frame {self.frame_count}: RGB shape {rgb.shape}, Depth shape {depth.shape}")

            # Create point cloud (pixels without a depth reading are dropped)
            xyz, valid = depth_to_points(depth)
            colors = rgb.reshape((-1, 3))[valid]

            # Append to existing point cloud
            self.points = np.vstack((self.points, xyz))
            self.colors = np.vstack((self.colors, colors))

            self.log_debug(f"Total points: {len(self.points)}")

            # Downsample
            if len(self.points) > self.max_points:
                indices = np.random.choice(len(self.points), self.max_points, replace=False)
                self.points = self.points[indices]
                self.colors = self.colors[indices]
                self.log_debug(f"Downsampled to {len(self.points)} points")

            # Create a 2D projection of the point cloud
            proj_img = np.zeros((480, 640, 3), dtype=np.uint8)
            for point, color in zip(self.points, self.colors):
                x, y, z = point
                if 0 <= int(x + 320) < 640 and 0 <= int(240 - y) < 480:
                    proj_img[int(240 - y), int(x + 320)] = color

            # Perform object detection
            results = self.yolo_model(rgb)
            self.log_debug(f"Detected {len(results)} objects")

            # Draw detection boxes on the projection
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(proj_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Display the projection
            h, w, ch = proj_img.shape
            bytes_per_line = ch * w
            q_image = QImage(proj_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(q_image))

            self.frame_count += 1
            self.status_label.setText(f"Status: Mapping (Frame {self.frame_count})")

        except Exception as e:
            self.handle_error(str(e))

    def handle_error(self, error_message):
        self.status_label.setText(f"Error: {error_message}")
        self.log_debug(f"Error occurred: {error_message}")
        self.log_debug(traceback.format_exc())

    def closeEvent(self, event):
        self.log_debug("Closing application")
        self.kinect.close()
        event.accept()

def main():
    app = QApplication([])
    window = KinectSLAMApp()
    window.show()
    app.exec_()

if __name__ == "__main__":
    main()
