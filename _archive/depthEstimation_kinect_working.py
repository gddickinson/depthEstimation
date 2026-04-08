import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Conv2D, UpSampling2D
from tensorflow.keras.models import Model
import dlib
from kinect_wrapper import PyKinectCapture

# Constants
frameWidth = 640
frameHeight = 480
AVERAGE_FACE_WIDTH = 0.15  # 15cm
AVERAGE_PERSON_HEIGHT = 1.7  # 170cm

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

def create_wireframe_model(landmarks, person_keypoints, depth_map):
    model_3d = []
    height, width = depth_map.shape

    def safe_index(y, x):
        return min(max(y, 0), height - 1), min(max(x, 0), width - 1)

    if landmarks:
        for point in landmarks.parts():
            y, x = safe_index(point.y, point.x)
            z = depth_map[y, x]  # Use Kinect depth directly
            model_3d.append((point.x, point.y, z))

    if person_keypoints is not None:
        for point in person_keypoints:
            if point[2] > 0.5:  # Check confidence
                y, x = safe_index(int(point[1]), int(point[0]))
                z = depth_map[y, x]  # Use Kinect depth directly
                model_3d.append((point[0], point[1], z))

    return np.array(model_3d)

def draw_wireframe(img, wireframe_model):
    if len(wireframe_model) == 0:
        return  # No points to draw

    # Draw face wireframe
    face_points = wireframe_model[:68] if len(wireframe_model) >= 68 else wireframe_model
    for i in range(len(face_points)):
        cv2.circle(img, (int(face_points[i][0]), int(face_points[i][1])), 2, (0, 255, 0), -1)

    # Draw body wireframe
    body_points = wireframe_model[68:] if len(wireframe_model) > 68 else []
    if len(body_points) >= 17:  # Assuming we have at least 17 body keypoints
        # Connect keypoints to form a simple stick figure
        connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Arms
            (5, 6), (5, 11), (6, 12), (11, 12),  # Body
            (11, 13), (12, 14), (13, 15), (14, 16)  # Legs
        ]
        for connection in connections:
            if connection[0] < len(body_points) and connection[1] < len(body_points):
                start_point = tuple(map(int, body_points[connection[0]][:2]))
                end_point = tuple(map(int, body_points[connection[1]][:2]))
                cv2.line(img, start_point, end_point, (0, 0, 255), 2)

def main():
    kinect = PyKinectCapture()
    kinect.init()

    frame_count = 0
    test_image_saved = False

    while True:
        rgb, depth = kinect.get_frames()
        if rgb is None or depth is None:
            continue

        # Convert RGB from BGR to RGB color space
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # Perform YOLO segmentation
        results = yolo_seg_model(rgb)

        person_bbox = None
        person_keypoints = None
        if len(results) > 0 and len(results[0].boxes) > 0:
            person_bbox = results[0].boxes.data[0][:4].cpu().numpy().astype(int)

            # Start or update tracking
            if not person_tracker.tracking_started:
                person_tracker.start_track(rgb, person_bbox)
            else:
                success, tracked_bbox = person_tracker.update(rgb)
                if success:
                    person_bbox = tracked_bbox

            # Get person keypoints
            person_keypoints = results[0].keypoints.data[0].cpu().numpy() if results[0].keypoints is not None else None

        landmarks = detect_facial_landmarks(rgb)

        # Create wireframe model using Kinect depth
        wireframe_model = create_wireframe_model(landmarks[0] if landmarks else None, person_keypoints, depth)

        # Draw wireframe on the image
        draw_wireframe(rgb, wireframe_model)

        # Draw segmentation mask
        if len(results) > 0 and results[0].masks is not None:
            mask = results[0].masks.data[0].cpu().numpy()
            mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]))
            rgb[mask > 0.5] = rgb[mask > 0.5] * 0.5 + np.array([0, 0, 255]) * 0.5

        # Update person tracker
        if person_bbox is not None:
            person_depth = np.median(depth[int(person_bbox[1]):int(person_bbox[3]), int(person_bbox[0]):int(person_bbox[2])])
            person_tracker.update(rgb)

        avg_person_depth = np.median(depth[depth > 0]) if person_tracker.bbox is not None else None
        if avg_person_depth is not None:
            cv2.putText(rgb, f"Avg Depth: {avg_person_depth:.2f}mm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Normalize depth for display
        depth_display = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        depth_colormap = cv2.applyColorMap(depth_display, cv2.COLORMAP_JET)

        combined_display = np.hstack((rgb, depth_colormap))
        cv2.imshow("Kinect Depth Estimation", combined_display)

        # Save test image
        frame_count += 1
        if frame_count == 30 and not test_image_saved:
            cv2.imwrite("kinect_depth_estimation_test.jpg", combined_display)
            print("Test image saved as 'kinect_depth_estimation_test.jpg'")
            test_image_saved = True

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    kinect.release()

if __name__ == "__main__":
    main()
