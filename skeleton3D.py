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
import pyqtgraph.opengl as gl
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
    (5, 7), (7, 9),   # Left arm
    (6, 8), (8, 10),  # Right arm
    (11, 13), (13, 15),  # Left leg
    (12, 14), (14, 16),  # Right leg
    (5, 6), (5, 11), (6, 12),  # Body
    (0, 5), (0, 6)    # Neck
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
        self.setWindowTitle("Kinect 3D Depth Estimation")
        self.setGeometry(100, 100, 1200, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        display_layout = QHBoxLayout()
        main_layout.addLayout(display_layout)

        # Create image display widgets
        self.rgb_view = pg.ImageView()
        self.depth_view = pg.ImageView()

        # Create 3D view
        self.plot3d = gl.GLViewWidget()
        self.plot3d.opts['distance'] = 500
        self.plot3d.setBackgroundColor('k')

        display_layout.addWidget(self.rgb_view)
        display_layout.addWidget(self.depth_view)
        display_layout.addWidget(self.plot3d)

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
        self.current_frame = rgb

        # Perform YOLO pose detection
        results = yolo_pose_model(rgb)

        # Perform MediaPipe Hands detection
        rgb.flags.writeable = False
        hand_results = self.hands.process(rgb)
        rgb.flags.writeable = True

        self.hand_landmarks = hand_results.multi_hand_landmarks

        if len(results) > 0 and results[0].keypoints is not None:
            keypoints = results[0].keypoints.data[0].cpu().numpy()
            self.draw_3d_skeleton(keypoints, depth)

        self.rgb_view.setImage(rgb.transpose(1, 0, 2))
        self.depth_view.setImage(depth.T, levels=(np.min(depth), np.max(depth)))

    def draw_3d_skeleton(self, keypoints, depth):
        self.plot3d.clear()

        # Convert 2D keypoints to 3D using depth information
        keypoints_3d = []
        for point in keypoints:
            if point[2] > 0.5:  # Only consider keypoints with high confidence
                x, y = int(point[0]), int(point[1])
                if 0 <= y < depth.shape[0] and 0 <= x < depth.shape[1]:
                    z = depth[y, x]
                    keypoints_3d.append((x, y, z))
                else:
                    keypoints_3d.append(None)
            else:
                keypoints_3d.append(None)

        # Draw bones
        for connection in SKELETON_CONNECTIONS:
            start = keypoints_3d[connection[0]]
            end = keypoints_3d[connection[1]]
            if start is not None and end is not None:
                line = gl.GLLinePlotItem(pos=np.array([start, end]), color=(1, 1, 1, 1), width=2)
                self.plot3d.addItem(line)

        # Draw joints
        valid_keypoints = [kp for kp in keypoints_3d if kp is not None]
        if valid_keypoints:
            scatter = gl.GLScatterPlotItem(pos=np.array(valid_keypoints), size=10, color=(1, 0, 0, 1))
            self.plot3d.addItem(scatter)

        # Draw head
        if keypoints_3d[0] is not None:  # If nose keypoint is detected
            gray = cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2GRAY)
            faces = face_detector(gray)
            for face in faces:
                landmarks = landmark_predictor(gray, face)
                landmark_points = np.array([(p.x, p.y) for p in landmarks.parts()])

                # Get 3D positions for landmarks
                landmark_points_3d = []
                for point in landmark_points:
                    x, y = int(point[0]), int(point[1])
                    if 0 <= y < depth.shape[0] and 0 <= x < depth.shape[1]:
                        z = depth[y, x]
                        landmark_points_3d.append((x, y, z))
                    else:
                        landmark_points_3d.append(None)

                # Draw facial landmarks
                for start, end in FACIAL_LANDMARKS_CONNECTORS:
                    start_point = landmark_points_3d[start]
                    end_point = landmark_points_3d[end]
                    if start_point is not None and end_point is not None:
                        line = gl.GLLinePlotItem(pos=np.array([start_point, end_point]), color=(1, 1, 0, 1), width=1)
                        self.plot3d.addItem(line)

                # Create head mesh
                head_mesh = self.create_head_mesh(landmark_points_3d)
                if head_mesh is not None:
                    self.plot3d.addItem(head_mesh)

        # Draw hands
        if self.hand_landmarks:
            for hand_landmarks in self.hand_landmarks:
                self.draw_3d_hand(hand_landmarks, depth)

    def create_head_mesh(self, landmark_points_3d):
        valid_points = [p for p in landmark_points_3d if p is not None]
        if len(valid_points) < 3:
            return None

        vertices = np.array(valid_points)
        faces = []
        for i in range(1, len(vertices) - 1):
            faces.append([0, i, i+1])

        colors = np.ones((len(vertices), 4))
        colors[:, 3] = 0.5  # Set alpha for transparency

        mesh = gl.GLMeshItem(vertexes=vertices, faces=faces, smooth=True, drawEdges=True, edgeColor=(0, 1, 0, 1))
        mesh.setGLOptions('additive')
        return mesh

    def draw_3d_hand(self, hand_landmarks, depth):
        hand_points_3d = []
        for landmark in hand_landmarks.landmark:
            x = int(landmark.x * frameWidth)
            y = int(landmark.y * frameHeight)
            if 0 <= y < depth.shape[0] and 0 <= x < depth.shape[1]:
                z = depth[y, x]
                hand_points_3d.append((x, y, z))
            else:
                hand_points_3d.append(None)

        # Draw hand connections
        for connection in HAND_CONNECTIONS:
            start_point = hand_points_3d[connection[0]]
            end_point = hand_points_3d[connection[1]]
            if start_point is not None and end_point is not None:
                line = gl.GLLinePlotItem(pos=np.array([start_point, end_point]), color=(0, 1, 0, 1), width=1)
                self.plot3d.addItem(line)

        # Draw hand keypoints
        valid_points = [p for p in hand_points_3d if p is not None]
        if valid_points:
            scatter = gl.GLScatterPlotItem(pos=np.array(valid_points), size=5, color=(0, 1, 0, 1))
            self.plot3d.addItem(scatter)

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
