import sys
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import dlib
from kinect_wrapper import PyKinectCapture
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout
from PyQt5.QtCore import QTimer
import pyqtgraph as pg

# Constants
frameWidth = 640
frameHeight = 480

# Load models
print("Loading models...")
yolo_pose_model = YOLO('yolov8n-pose.pt')
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Define skeleton connections
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head and shoulders
    (5, 6), (5, 11), (6, 12), (11, 12),  # Body
    (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (11, 13), (13, 15), (12, 14), (14, 16)  # Legs
]

class DepthEstimationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kinect Depth Estimation")
        self.setGeometry(100, 100, 1200, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)

        # Create image display widgets
        self.rgb_view = pg.ImageView()
        self.depth_view = pg.ImageView()
        self.skeleton_view = pg.PlotWidget()

        layout.addWidget(self.rgb_view)
        layout.addWidget(self.depth_view)
        layout.addWidget(self.skeleton_view)

        self.kinect = PyKinectCapture()
        self.kinect.init()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Update every 30 ms

    def update_frame(self):
        rgb, depth = self.kinect.get_frames()
        if rgb is None or depth is None:
            return

        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # Perform YOLO pose detection
        results = yolo_pose_model(rgb)

        if len(results) > 0 and results[0].keypoints is not None:
            keypoints = results[0].keypoints.data[0].cpu().numpy()
            self.draw_skeleton(keypoints)

        self.rgb_view.setImage(rgb.transpose(1, 0, 2))
        self.depth_view.setImage(depth.T, levels=(np.min(depth), np.max(depth)))

    def draw_skeleton(self, keypoints):
        self.skeleton_view.clear()

        # Draw bones
        for connection in SKELETON_CONNECTIONS:
            if keypoints[connection[0]][2] > 0.5 and keypoints[connection[1]][2] > 0.5:
                x = [keypoints[connection[0]][0], keypoints[connection[1]][0]]
                y = [keypoints[connection[0]][1], keypoints[connection[1]][1]]
                self.skeleton_view.plot(x, y, pen=pg.mkPen(color=(255, 255, 255), width=2))

        # Draw joints
        for point in keypoints:
            if point[2] > 0.5:
                self.skeleton_view.plot([point[0]], [point[1]], pen=None, symbol='o', symbolSize=6, symbolBrush=(255, 0, 0))

        self.skeleton_view.setYRange(0, frameHeight)
        self.skeleton_view.setXRange(0, frameWidth)
        self.skeleton_view.invertY(True)

    def closeEvent(self, event):
        self.kinect.release()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = DepthEstimationApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
