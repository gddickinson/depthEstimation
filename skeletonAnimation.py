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
from scipy.spatial.transform import Rotation as R
from filterpy.kalman import KalmanFilter

# Constants
frameWidth = 640
frameHeight = 480

# Load models
print("Loading models...")
yolo_pose_model = YOLO('yolov8n-pose.pt')
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# MediaPipe hands setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

class SkeletonJoint:
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.children = []
        self.position = np.zeros(3)
        self.rotation = R.from_quat([0, 0, 0, 1])
        self.kalman = KalmanFilter(dim_x=6, dim_z=3)  # 3D position and velocity

        # Initialize Kalman filter
        self.kalman.x = np.zeros(6)
        self.kalman.F = np.array([[1,0,0,1,0,0],
                                  [0,1,0,0,1,0],
                                  [0,0,1,0,0,1],
                                  [0,0,0,1,0,0],
                                  [0,0,0,0,1,0],
                                  [0,0,0,0,0,1]])
        self.kalman.H = np.array([[1,0,0,0,0,0],
                                  [0,1,0,0,0,0],
                                  [0,0,1,0,0,0]])
        self.kalman.P *= 1000.
        self.kalman.R = np.eye(3) * 0.1

    def update(self, measurement=None):
        if measurement is not None:
            self.kalman.update(measurement)
        else:
            self.kalman.predict()

        self.position = self.kalman.x[:3]

    def apply_inverse_kinematics(self, end_effector_position):
        # Simplified IK: just move towards the target
        direction = end_effector_position - self.position
        self.position += direction * 0.1  # Adjust step size as needed

class Skeleton:
    def __init__(self):
        self.joints = {}
        self.create_skeleton_structure()
        self.mesh = self.create_skeleton_mesh()

    def create_skeleton_structure(self):
        # Create joints and their hierarchy
        self.joints['root'] = SkeletonJoint('root')
        self.joints['spine'] = SkeletonJoint('spine', self.joints['root'])
        self.joints['neck'] = SkeletonJoint('neck', self.joints['spine'])
        self.joints['head'] = SkeletonJoint('head', self.joints['neck'])
        self.joints['left_shoulder'] = SkeletonJoint('left_shoulder', self.joints['neck'])
        self.joints['left_elbow'] = SkeletonJoint('left_elbow', self.joints['left_shoulder'])
        self.joints['left_hand'] = SkeletonJoint('left_hand', self.joints['left_elbow'])
        self.joints['right_shoulder'] = SkeletonJoint('right_shoulder', self.joints['neck'])
        self.joints['right_elbow'] = SkeletonJoint('right_elbow', self.joints['right_shoulder'])
        self.joints['right_hand'] = SkeletonJoint('right_hand', self.joints['right_elbow'])
        self.joints['left_hip'] = SkeletonJoint('left_hip', self.joints['root'])
        self.joints['left_knee'] = SkeletonJoint('left_knee', self.joints['left_hip'])
        self.joints['left_foot'] = SkeletonJoint('left_foot', self.joints['left_knee'])
        self.joints['right_hip'] = SkeletonJoint('right_hip', self.joints['root'])
        self.joints['right_knee'] = SkeletonJoint('right_knee', self.joints['right_hip'])
        self.joints['right_foot'] = SkeletonJoint('right_foot', self.joints['right_knee'])

    def create_skeleton_mesh(self):
        # Define connections for a more human-like skeleton
        connections = [
            ('head', 'neck'), ('neck', 'spine'), ('spine', 'root'),
            ('neck', 'left_shoulder'), ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_hand'),
            ('neck', 'right_shoulder'), ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_hand'),
            ('root', 'left_hip'), ('left_hip', 'left_knee'), ('left_knee', 'left_foot'),
            ('root', 'right_hip'), ('right_hip', 'right_knee'), ('right_knee', 'right_foot')
        ]

        lines = []
        colors = []
        for start, end in connections:
            lines.append([self.joints[start].position, self.joints[end].position])
            colors.append([1, 1, 1, 1])

        mesh = gl.GLLinePlotItem(pos=np.array(lines), color=colors, width=5, antialias=True, mode='lines')
        return mesh

    def update(self, keypoints):
        # Update joint positions based on keypoints
        for joint_name, keypoint in keypoints.items():
            if joint_name in self.joints:
                self.joints[joint_name].update(keypoint)

        # Infer positions of joints not directly detected
        self.infer_missing_joints(keypoints)

        # Update mesh
        self.update_mesh()

    def infer_missing_joints(self, keypoints):
        # Infer spine position if not detected
        if 'spine' not in keypoints:
            neck = self.joints['neck'].position
            root = self.joints['root'].position
            self.joints['spine'].position = (neck + root) / 2

        # Infer positions of other potentially missing joints
        if 'left_hip' not in keypoints and 'right_hip' in keypoints:
            self.joints['left_hip'].position = self.joints['right_hip'].position * np.array([-1, 1, 1])
        if 'right_hip' not in keypoints and 'left_hip' in keypoints:
            self.joints['right_hip'].position = self.joints['left_hip'].position * np.array([-1, 1, 1])

        # Similar inferences can be made for other symmetric joints

    def update_mesh(self):
        connections = [
            ('head', 'neck'), ('neck', 'spine'), ('spine', 'root'),
            ('neck', 'left_shoulder'), ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_hand'),
            ('neck', 'right_shoulder'), ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_hand'),
            ('root', 'left_hip'), ('left_hip', 'left_knee'), ('left_knee', 'left_foot'),
            ('root', 'right_hip'), ('right_hip', 'right_knee'), ('right_knee', 'right_foot')
        ]

        lines = []
        for start, end in connections:
            lines.append([self.joints[start].position, self.joints[end].position])

        self.mesh.setData(pos=np.array(lines))

class SkeletonAnimationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Skeleton Animation")
        self.setGeometry(100, 100, 1200, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        display_layout = QHBoxLayout()
        main_layout.addLayout(display_layout)

        # Create image display widgets
        self.rgb_view = pg.ImageView()
        self.depth_view = pg.ImageView()
        self.skeleton_view = gl.GLViewWidget()

        display_layout.addWidget(self.rgb_view)
        display_layout.addWidget(self.depth_view)
        display_layout.addWidget(self.skeleton_view)

        # Add exit button
        self.exit_button = QPushButton("Exit")
        self.exit_button.clicked.connect(self.close)
        main_layout.addWidget(self.exit_button)

        self.kinect = PyKinectCapture()
        self.kinect.init()

        # Initialize MediaPipe Hands
        self.hands = mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        self.skeleton = Skeleton()
        self.skeleton_view.addItem(self.skeleton.mesh)

        # Set up camera
        self.skeleton_view.setCameraPosition(distance=2, elevation=30, azimuth=45)

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

        keypoints = {}

        if len(results) > 0 and results[0].keypoints is not None:
            yolo_keypoints = results[0].keypoints.data[0].cpu().numpy()
            # Map YOLO keypoints to our skeleton structure
            keypoint_mapping = {
                0: 'head', 5: 'left_shoulder', 6: 'right_shoulder',
                7: 'left_elbow', 8: 'right_elbow', 9: 'left_hand',
                10: 'right_hand', 11: 'left_hip', 12: 'right_hip',
                13: 'left_knee', 14: 'right_knee', 15: 'left_foot',
                16: 'right_foot'
            }
            for i, kp in enumerate(yolo_keypoints):
                if i in keypoint_mapping and kp[2] > 0.5:
                    keypoints[keypoint_mapping[i]] = kp[:3]

        # Infer root and neck positions if not provided by YOLO
        if 'left_hip' in keypoints and 'right_hip' in keypoints:
            keypoints['root'] = (keypoints['left_hip'] + keypoints['right_hip']) / 2
        if 'left_shoulder' in keypoints and 'right_shoulder' in keypoints:
            keypoints['neck'] = (keypoints['left_shoulder'] + keypoints['right_shoulder']) / 2

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # You might want to differentiate between left and right hand here
                wrist = hand_landmarks.landmark[0]
                keypoints['left_hand'] = np.array([wrist.x * frameWidth, wrist.y * frameHeight, wrist.z])

        self.skeleton.update(keypoints)

        self.rgb_view.setImage(rgb.transpose(1, 0, 2))
        self.depth_view.setImage(depth.T, levels=(np.min(depth), np.max(depth)))

    def closeEvent(self, event):
        self.kinect.close()
        self.hands.close()
        event.accept()

def main():
    app = QApplication(sys.argv)
    window = SkeletonAnimationApp()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
