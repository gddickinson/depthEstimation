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
from filterpy.kalman import KalmanFilter

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

class KeypointTracker:
    def __init__(self, initial_pos, initial_confidence=1.0):
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.x = np.array([initial_pos[0], initial_pos[1], 0., 0.])  # position and velocity
        self.kf.F = np.array([[1, 0, 1, 0],
                              [0, 1, 0, 1],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])  # state transition matrix
        self.kf.H = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0]])  # measurement function
        self.kf.P *= 1000.  # covariance matrix
        self.kf.R = np.eye(2) * 0.1  # measurement noise
        self.kf.Q = np.eye(4) * 0.1  # process noise
        self.confidence = initial_confidence
        self.confidence_decay = 0.9  # Confidence decay factor when no measurement

    def update(self, measurement, confidence=None):
        self.kf.predict()
        if measurement is not None:
            self.kf.update(measurement)
            if confidence is not None:
                self.confidence = confidence
        else:
            self.confidence *= self.confidence_decay
        return self.kf.x[:2], self.confidence

    @property
    def state(self):
        return np.append(self.kf.x[:2], self.confidence)

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

        self.skeleton_trackers = None
        self.hand_trackers = None
        self.face_tracker = None

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
            self.update_skeleton_trackers(keypoints)
        else:
            self.update_skeleton_trackers(None)

        if self.hand_landmarks:
            self.update_hand_trackers(self.hand_landmarks)
        else:
            self.update_hand_trackers(None)

        # Perform face detection and update face tracker
        gray = cv2.cvtColor(self.current_frame, cv2.COLOR_RGB2GRAY)
        faces = face_detector(gray)
        if faces:
            landmarks = landmark_predictor(gray, faces[0])
            landmark_points = np.array([(p.x, p.y) for p in landmarks.parts()])
            self.update_face_tracker(landmark_points)
        else:
            self.update_face_tracker(None)

        self.draw_tracked_skeleton()

        self.rgb_view.setImage(rgb.transpose(1, 0, 2))
        self.depth_view.setImage(depth.T, levels=(np.min(depth), np.max(depth)))

    def update_skeleton_trackers(self, keypoints):
        if self.skeleton_trackers is None:
            if keypoints is not None:
                self.skeleton_trackers = [KeypointTracker(kp[:2], kp[2]) for kp in keypoints]
            else:
                return

        if keypoints is not None:
            for i, (tracker, keypoint) in enumerate(zip(self.skeleton_trackers, keypoints)):
                if keypoint[2] > 0.5:  # If keypoint is detected with high confidence
                    tracker.update(keypoint[:2], keypoint[2])
                else:
                    tracker.update(None)
        else:
            for tracker in self.skeleton_trackers:
                tracker.update(None)

    def update_hand_trackers(self, hand_landmarks):
        if self.hand_trackers is None:
            self.hand_trackers = []

        if hand_landmarks:
            for hand in hand_landmarks:
                hand_points = [(landmark.x * frameWidth, landmark.y * frameHeight) for landmark in hand.landmark]
                if len(self.hand_trackers) < 2:
                    self.hand_trackers.append([KeypointTracker(point) for point in hand_points])
                else:
                    for tracker, point in zip(self.hand_trackers[-1], hand_points):
                        tracker.update(point)
        else:
            for trackers in self.hand_trackers:
                for tracker in trackers:
                    tracker.update(None)

    def update_face_tracker(self, landmark_points):
        if self.face_tracker is None:
            if landmark_points is not None:
                self.face_tracker = [KeypointTracker(point) for point in landmark_points]
            else:
                return

        if landmark_points is not None:
            for tracker, point in zip(self.face_tracker, landmark_points):
                tracker.update(point)
        else:
            for tracker in self.face_tracker:
                tracker.update(None)

    def draw_tracked_skeleton(self):
        self.skeleton_view.clear()

        if self.skeleton_trackers:
            keypoints = np.array([tracker.state for tracker in self.skeleton_trackers])

            # Fix feet positions
            if keypoints[15][2] > 0.5 and keypoints[16][2] > 0.5:  # If both feet are detected
                left_foot = keypoints[15][:2]
                right_foot = keypoints[16][:2]
            elif keypoints[13][2] > 0.5 and keypoints[14][2] > 0.5:  # If knees are detected
                left_foot = keypoints[13][:2] + (keypoints[13][:2] - keypoints[11][:2])
                right_foot = keypoints[14][:2] + (keypoints[14][:2] - keypoints[12][:2])
            else:
                left_foot = keypoints[11][:2] + np.array([0, 100])  # Default position below hip
                right_foot = keypoints[12][:2] + np.array([0, 100])

            keypoints[15][:2] = left_foot
            keypoints[16][:2] = right_foot

            self.draw_skeleton(keypoints)

            # Draw hands
            left_wrist = keypoints[9][:2]
            right_wrist = keypoints[10][:2]
            left_elbow = keypoints[7][:2]
            right_elbow = keypoints[8][:2]

            if self.hand_trackers and len(self.hand_trackers) > 0:
                for hand_trackers in self.hand_trackers:
                    hand_points = np.array([tracker.state for tracker in hand_trackers])

                    # Determine if this is left or right hand
                    is_left_hand = np.linalg.norm(hand_points[0][:2] - left_wrist) < np.linalg.norm(hand_points[0][:2] - right_wrist)

                    if is_left_hand:
                        if hand_points[0][2] > 0.5:  # If hand is detected with high confidence
                            self.draw_hand(hand_points)
                        else:
                            self.draw_default_hand(left_wrist, left_elbow)
                    else:
                        if hand_points[0][2] > 0.5:  # If hand is detected with high confidence
                            self.draw_hand(hand_points)
                        else:
                            self.draw_default_hand(right_wrist, right_elbow)
            else:
                # Draw default hands if no hands are detected
                self.draw_default_hand(left_wrist, left_elbow)
                self.draw_default_hand(right_wrist, right_elbow)

        if self.face_tracker:
            landmark_points = np.array([tracker.state for tracker in self.face_tracker])
            self.draw_face(landmark_points)

    def draw_skeleton(self, keypoints):
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

        scatter = pg.ScatterPlotItem(pos=joint_positions, size=6, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0))
        self.skeleton_view.addItem(scatter)

        self.skeleton_view.setYRange(0, frameHeight)
        self.skeleton_view.setXRange(0, frameWidth)
        self.skeleton_view.invertY(True)

    def draw_hand(self, hand_points):
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

    def draw_default_hand(self, wrist, elbow):
        # Calculate default hand position
        hand_direction = wrist - elbow
        hand_length = np.linalg.norm(hand_direction) * 0.7  # Adjust this factor to change hand size
        hand_direction_normalized = hand_direction / np.linalg.norm(hand_direction)
        hand_end = wrist + hand_direction_normalized * hand_length

        # Draw a simple hand representation
        self.skeleton_view.plot([wrist[0], hand_end[0]], [wrist[1], hand_end[1]],
                                pen=pg.mkPen(color=(0, 255, 0), width=2))
        scatter = pg.ScatterPlotItem(pos=[hand_end], size=8, pen=pg.mkPen(None), brush=pg.mkBrush(0, 255, 0))
        self.skeleton_view.addItem(scatter)

    def draw_face(self, landmark_points):
        # Draw facial landmarks
        for start, end in FACIAL_LANDMARKS_CONNECTORS:
            start_point = landmark_points[start]
            end_point = landmark_points[end]
            self.skeleton_view.plot([start_point[0], end_point[0]],
                                    [start_point[1], end_point[1]],
                                    pen=pg.mkPen(color=(255, 255, 0), width=1))

        # Draw skull-like head shape
        transformed_skull = self.estimate_head_pose(landmark_points)
        skull_curve = pg.PlotCurveItem(x=transformed_skull[:, 0],
                                       y=transformed_skull[:, 1],
                                       pen=pg.mkPen(color=(0, 255, 0), width=2))
        self.skeleton_view.addItem(skull_curve)

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
