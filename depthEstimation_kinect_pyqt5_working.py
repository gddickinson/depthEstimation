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
yolo_seg_model = YOLO('yolov8n-seg.pt')
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

class PersonTracker:
    def __init__(self):
        self.tracker = cv2.TrackerKCF_create()
        self.bbox = None
        self.tracking_started = False

    def start_track(self, frame, bbox):
        self.bbox = bbox
        self.tracker = cv2.TrackerKCF_create()
        self.tracker.init(frame, tuple(bbox))
        self.tracking_started = True

    def update(self, frame):
        if self.tracking_started:
            success, bbox = self.tracker.update(frame)
            if success:
                self.bbox = bbox
            return success, bbox
        return False, None

person_tracker = PersonTracker()

def detect_facial_landmarks(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = face_detector(gray)
    landmarks = []
    for face in faces:
        shape = landmark_predictor(gray, face)
        landmarks.append(shape)
    return landmarks

def create_stick_figure(keypoints, image_shape):
    stick_figure = np.zeros(image_shape[:2], dtype=np.uint8)
    connections = [
        (5, 7), (7, 9), (6, 8), (8, 10), (5, 6),
        (5, 11), (6, 12), (11, 13), (13, 15), (12, 14), (14, 16)
    ]
    for connection in connections:
        start_point = keypoints[connection[0]]
        end_point = keypoints[connection[1]]
        if start_point[2] > 0.5 and end_point[2] > 0.5:
            cv2.line(stick_figure,
                     (int(start_point[0]), int(start_point[1])),
                     (int(end_point[0]), int(end_point[1])),
                     255, 2)
    for point in keypoints:
        if point[2] > 0.5:
            cv2.circle(stick_figure, (int(point[0]), int(point[1])), 3, 255, -1)
    return stick_figure

class DepthEstimationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kinect Depth Estimation")
        self.setGeometry(100, 100, 1200, 400)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)

        # Create image display widgets
        self.rgb_view = pg.ImageView()
        self.depth_view = pg.ImageView()
        self.stick_figure_view = pg.ImageView()

        layout.addWidget(self.rgb_view)
        layout.addWidget(self.depth_view)
        layout.addWidget(self.stick_figure_view)

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

        results = yolo_seg_model(rgb)

        person_bbox = None
        person_keypoints = None
        if len(results) > 0 and len(results[0].boxes) > 0:
            person_bbox = results[0].boxes.data[0][:4].cpu().numpy().astype(int)
            if not person_tracker.tracking_started:
                person_tracker.start_track(rgb, person_bbox)
            else:
                success, tracked_bbox = person_tracker.update(rgb)
                if success:
                    person_bbox = tracked_bbox
            person_keypoints = results[0].keypoints.data[0].cpu().numpy() if results[0].keypoints is not None else None

        landmarks = detect_facial_landmarks(rgb)

        if len(results) > 0 and results[0].masks is not None:
            mask = results[0].masks.data[0].cpu().numpy()
            mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]))
            rgb[mask > 0.5] = rgb[mask > 0.5] * 0.5 + np.array([0, 0, 255]) * 0.5

        if person_bbox is not None:
            person_depth = np.median(depth[int(person_bbox[1]):int(person_bbox[3]), int(person_bbox[0]):int(person_bbox[2])])
            cv2.putText(rgb, f"Avg Depth: {person_depth:.2f}mm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        stick_figure = None
        if person_keypoints is not None:
            stick_figure = create_stick_figure(person_keypoints, rgb.shape)

        self.rgb_view.setImage(rgb.transpose(1, 0, 2))
        self.depth_view.setImage(depth.T, levels=(np.min(depth), np.max(depth)))
        if stick_figure is not None:
            self.stick_figure_view.setImage(stick_figure.T)

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
