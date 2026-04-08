import sys
import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import dlib
import mediapipe as mp
from kinect_wrapper import PyKinectCapture
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton
from PyQt5.QtCore import QTimer
import pyqtgraph as pg
from scipy.spatial import ConvexHull
from scipy.spatial import procrustes

# Constants
frameWidth = 640
frameHeight = 480
HEAD_RADIUS = 60  # Radius of the head sphere in pixels

# Load models
print("Loading models...")
yolo_pose_model = YOLO('yolov8n-pose.pt')
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# MediaPipe hands setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Define skeleton connections
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # Head and shoulders
    (5, 6), (5, 11), (6, 12), (11, 12),  # Body
    (5, 7), (7, 9), (6, 8), (8, 10),  # Arms
    (11, 13), (13, 15), (12, 14), (14, 16),  # Legs
    (0, 5), (0, 6)  # Neck connections
]

FACIAL_LANDMARKS_CONNECTORS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16),  # Jaw line
    (17, 18), (18, 19), (19, 20), (20, 21),  # Right eyebrow
    (22, 23), (23, 24), (24, 25), (25, 26),  # Left eyebrow
    (27, 28), (28, 29), (29, 30),  # Nose bridge
    (30, 31), (31, 32), (32, 33), (33, 34), (34, 35),  # Lower nose
    (36, 37), (37, 38), (38, 39), (39, 40), (40, 41), (41, 36),  # Right eye
    (42, 43), (43, 44), (44, 45), (45, 46), (46, 47), (47, 42),  # Left eye
    (48, 49), (49, 50), (50, 51), (51, 52), (52, 53), (53, 54), (54, 55), (55, 56), (56, 57), (57, 58), (58, 59), (59, 48),  # Outer lip
    (60, 61), (61, 62), (62, 63), (63, 64), (64, 65), (65, 66), (66, 67), (67, 60)  # Inner lip
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
]

class DepthEstimationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kinect Depth Estimation")
        self.setGeometry(100, 100, 1200, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        display_layout = QHBoxLayout()
        main_layout.addLayout(display_layout)

        # Create image display widgets
        self.rgb_view = pg.ImageView()
        self.depth_view = pg.ImageView()
        self.skeleton_view = pg.PlotWidget()

        display_layout.addWidget(self.rgb_view)
        display_layout.addWidget(self.depth_view)
        display_layout.addWidget(self.skeleton_view)

        # Initialize MediaPipe Hands
        self.hands = mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        self.hand_landmarks = None

        # Add exit button
        self.exit_button = QPushButton("Exit")
        self.exit_button.clicked.connect(self.close)
        main_layout.addWidget(self.exit_button)

        self.kinect = PyKinectCapture()
        self.kinect.init()

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Update every 30 ms

        self.current_frame = None

    def update_frame(self):
        rgb, depth = self.kinect.get_frames()
        if rgb is None or depth is None:
            return

        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        self.current_frame = rgb  # Store the current frame

        # Perform YOLO pose detection
        results = yolo_pose_model(rgb)

        # Perform MediaPipe Hands detection
        rgb.flags.writeable = False
        hand_results = self.hands.process(rgb)
        rgb.flags.writeable = True

        self.hand_landmarks = hand_results.multi_hand_landmarks

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
        joint_positions = []
        for point in keypoints:
            if point[2] > 0.5:
                joint_positions.append((point[0], point[1]))

        # Draw hands
        if self.hand_landmarks:
            for hand_landmarks in self.hand_landmarks:
                self.draw_hand(hand_landmarks)

        scatter = pg.ScatterPlotItem(pos=joint_positions, size=6, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0))
        self.skeleton_view.addItem(scatter)

        # Draw skull-like head shape
        if keypoints[0][2] > 0.5:  # If nose keypoint is detected
            gray = cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2GRAY)
            faces = face_detector(gray)
            for face in faces:
                landmarks = landmark_predictor(gray, face)
                landmark_points = np.array([(p.x, p.y) for p in landmarks.parts()])

                # Estimate head pose
                transformed_skull = self.estimate_head_pose(landmark_points)

                # Draw the skull shape
                skull_curve = pg.PlotCurveItem(x=transformed_skull[:, 0],
                                               y=transformed_skull[:, 1],
                                               pen=pg.mkPen(color=(0, 255, 0), width=2))
                self.skeleton_view.addItem(skull_curve)

                # Draw facial landmarks
                for start, end in FACIAL_LANDMARKS_CONNECTORS:
                    start_point = landmark_points[start]
                    end_point = landmark_points[end]
                    self.skeleton_view.plot([start_point[0], end_point[0]],
                                            [start_point[1], end_point[1]],
                                            pen=pg.mkPen(color=(255, 255, 0), width=1))

        # Draw neck
        if keypoints[0][2] > 0.5 and keypoints[5][2] > 0.5 and keypoints[6][2] > 0.5:
            neck_start = ((keypoints[5][0] + keypoints[6][0]) / 2, (keypoints[5][1] + keypoints[6][1]) / 2)
            neck_end = (keypoints[0][0], keypoints[0][1])
            self.skeleton_view.plot([neck_start[0], neck_end[0]], [neck_start[1], neck_end[1]],
                                    pen=pg.mkPen(color=(255, 255, 255), width=2))

        self.skeleton_view.setYRange(0, frameHeight)
        self.skeleton_view.setXRange(0, frameWidth)
        self.skeleton_view.invertY(True)

    def create_smooth_curve(self, points, num_points=100):
        # Create a smooth curve from the given points
        hull = ConvexHull(points)
        hull_points = [points[i] for i in hull.vertices]
        hull_points.append(hull_points[0])  # Close the loop

        # Interpolate points along the curve
        smooth_points = []
        for i in range(len(hull_points) - 1):
            for t in np.linspace(0, 1, num_points // len(hull_points)):
                x = hull_points[i][0] * (1 - t) + hull_points[i+1][0] * t
                y = hull_points[i][1] * (1 - t) + hull_points[i+1][1] * t
                smooth_points.append((x, y))

        return smooth_points

    def estimate_head_pose(self, landmarks):
        # Define the 3D model points
        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left Mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ]) / 4.5  # Scale down the model

        # Get the corresponding 2D points from landmarks
        image_points = np.array([
            landmarks[30],  # Nose tip
            landmarks[8],   # Chin
            landmarks[36],  # Left eye left corner
            landmarks[45],  # Right eye right corner
            landmarks[48],  # Left Mouth corner
            landmarks[54]   # Right mouth corner
        ], dtype="double")

        # Define the skull shape (now with the same number of points as image_points)
        skull_shape = np.array([
            (0, 0),      # Nose tip
            (0, -60),    # Chin
            (-30, 30),   # Left eye left corner
            (30, 30),    # Right eye right corner
            (-20, -20),  # Left Mouth corner
            (20, -20)    # Right mouth corner
        ])

        # Estimate the transformation
        _, transformed_skull, _ = procrustes(image_points, skull_shape)

        # Create a more detailed skull shape
        detailed_skull = np.array([
            (0, -60), (-50, -40), (-60, 0), (-50, 40), (-30, 60),
            (0, 70), (30, 60), (50, 40), (60, 0), (50, -40)
        ])

        # Apply the same transformation to the detailed skull
        scale = np.linalg.norm(transformed_skull[1] - transformed_skull[0]) / np.linalg.norm(skull_shape[1] - skull_shape[0])
        rotation = np.arctan2(transformed_skull[1, 1] - transformed_skull[0, 1], transformed_skull[1, 0] - transformed_skull[0, 0]) - \
                   np.arctan2(skull_shape[1, 1] - skull_shape[0, 1], skull_shape[1, 0] - skull_shape[0, 0])
        translation = transformed_skull[0] - skull_shape[0]

        transformed_detailed_skull = scale * detailed_skull.dot(np.array([[np.cos(rotation), -np.sin(rotation)],
                                                                          [np.sin(rotation), np.cos(rotation)]])) + translation

        return transformed_detailed_skull

    def draw_hand(self, hand_landmarks):
        hand_points = []
        for landmark in hand_landmarks.landmark:
            x = landmark.x * frameWidth
            y = landmark.y * frameHeight
            hand_points.append((x, y))

        # Draw hand connections
        for connection in HAND_CONNECTIONS:
            start_point = hand_points[connection[0]]
            end_point = hand_points[connection[1]]
            self.skeleton_view.plot([start_point[0], end_point[0]],
                                    [start_point[1], end_point[1]],
                                    pen=pg.mkPen(color=(0, 255, 0), width=1))

        # Draw hand keypoints
        scatter = pg.ScatterPlotItem(pos=hand_points, size=4, pen=pg.mkPen(None), brush=pg.mkBrush(0, 255, 0))
        self.skeleton_view.addItem(scatter)

    def closeEvent(self, event):
        self.kinect.close()
        self.hands.close()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = DepthEstimationApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

