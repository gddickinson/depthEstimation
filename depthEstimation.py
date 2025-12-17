import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Conv2D, UpSampling2D
from tensorflow.keras.models import Model
import dlib

# Constants
frameWidth = 640
frameHeight = 480
AVERAGE_FACE_WIDTH = 0.15  # 15cm
AVERAGE_PERSON_HEIGHT = 1.7  # 170cm

# Load models
print("Loading models...")
yolo_seg_model = YOLO('yolov8n-seg.pt')
person_det_model = YOLO('yolov8n-pose.pt')
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def create_depth_model():
    base_model = MobileNetV2(input_shape=(frameHeight, frameWidth, 3), include_top=False, weights='imagenet')
    x = base_model.output
    x = Conv2D(256, 3, padding='same', activation='relu')(x)
    x = UpSampling2D()(x)
    x = Conv2D(128, 3, padding='same', activation='relu')(x)
    x = UpSampling2D()(x)
    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    x = UpSampling2D()(x)
    x = Conv2D(32, 3, padding='same', activation='relu')(x)
    x = UpSampling2D()(x)
    outputs = Conv2D(1, 3, padding='same', activation='sigmoid')(x)
    return Model(inputs=base_model.input, outputs=outputs)

depth_model = create_depth_model()

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

def process_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def detect_objects_and_segment(frame):
    results = yolo_seg_model(frame)
    return results[0]

def detect_person_details(frame):
    results = person_det_model(frame)
    return results[0]

def detect_facial_landmarks(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = face_detector(gray)
    landmarks = []
    for face in faces:
        shape = landmark_predictor(gray, face)
        landmarks.append(shape)
    return landmarks

def estimate_depth(frame):
    input_image = frame.astype(np.float32) / 255.0
    depth_map = depth_model.predict(np.expand_dims(input_image, axis=0))[0, :, :, 0]
    return depth_map

def refine_depth_with_person_info(depth_map, landmarks, person_bbox, person_keypoints):
    refined_depth = depth_map.copy()
    height, width = refined_depth.shape

    def safe_index(y, x):
        return min(max(y, 0), height - 1), min(max(x, 0), width - 1)

    if person_bbox is not None:
        # Adjust bbox to fit within depth map
        x1, y1, x2, y2 = person_bbox
        x1, y1 = safe_index(y1, x1)
        x2, y2 = safe_index(y2, x2)

        # Create a mask for the person
        person_mask = np.zeros_like(refined_depth, dtype=np.uint8)
        cv2.rectangle(person_mask, (x1, y1), (x2, y2), 1, -1)

        # Estimate person's depth
        person_depth = np.median(refined_depth[person_mask > 0])

        # Adjust depth for the person
        refined_depth[person_mask > 0] = person_depth

    if landmarks:
        # Refine face area
        face_points = np.array([(safe_index(p.y, p.x)) for p in landmarks.parts()])
        face_mask = np.zeros_like(refined_depth, dtype=np.uint8)
        cv2.fillConvexPoly(face_mask, face_points, 1)

        # Estimate face depth
        face_depth = np.median(refined_depth[face_mask > 0])

        # Adjust depth for the face
        refined_depth[face_mask > 0] = face_depth * 0.95  # Make face slightly closer

    if person_keypoints is not None:
        # Refine body parts based on keypoints
        for i, keypoint in enumerate(person_keypoints):
            if keypoint[2] > 0.5:  # Check confidence
                y, x = safe_index(int(keypoint[1]), int(keypoint[0]))
                cv2.circle(refined_depth, (x, y), 5, refined_depth[y, x] * 0.9, -1)  # Make keypoints slightly closer

    return refined_depth

def create_wireframe_model(landmarks, person_keypoints, depth_map):
    model_3d = []
    height, width = depth_map.shape

    def safe_index(y, x):
        return min(max(y, 0), height - 1), min(max(x, 0), width - 1)

    if landmarks:
        for point in landmarks.parts():
            y, x = safe_index(point.y, point.x)
            z = -depth_map[y, x]
            model_3d.append((point.x, point.y, z))

    if person_keypoints is not None:
        for point in person_keypoints:
            if point[2] > 0.5:  # Check confidence
                y, x = safe_index(int(point[1]), int(point[0]))
                z = -depth_map[y, x]
                model_3d.append((point[0], point[1], z))

    return np.array(model_3d)

def draw_wireframe(img, wireframe_model):
    # Draw face wireframe
    face_points = wireframe_model[:68]  # Assuming first 68 points are facial landmarks
    for i in range(68):
        cv2.circle(img, (int(face_points[i][0]), int(face_points[i][1])), 2, (0, 255, 0), -1)

    # Draw body wireframe
    body_points = wireframe_model[68:]
    if len(body_points) >= 17:  # Assuming we have at least 17 body keypoints
        # Connect keypoints to form a simple stick figure
        connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Arms
            (5, 6), (5, 11), (6, 12), (11, 12),  # Body
            (11, 13), (12, 14), (13, 15), (14, 16)  # Legs
        ]
        for connection in connections:
            start_point = tuple(map(int, body_points[connection[0]][:2]))
            end_point = tuple(map(int, body_points[connection[1]][:2]))
            cv2.line(img, start_point, end_point, (0, 0, 255), 2)

def apply_edge_aware_filter(depth_map):
    d = 7
    sigma_color = 100
    sigma_space = 100
    filtered_depth = cv2.bilateralFilter(depth_map.astype(np.float32), d, sigma_color, sigma_space)
    return filtered_depth

def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, frameWidth)
    cap.set(4, frameHeight)

    frame_count = 0
    test_image_saved = False

    while cap.isOpened():
        success, img = cap.read()
        if not success:
            break

        processed_img = process_image(img)

        # Perform YOLO segmentation
        results = yolo_seg_model(processed_img)

        person_bbox = None
        person_keypoints = None
        if len(results) > 0 and len(results[0].boxes) > 0:
            person_bbox = results[0].boxes.data[0][:4].cpu().numpy().astype(int)

            # Start or update tracking
            if not person_tracker.tracking_started:
                person_tracker.start_track(img, person_bbox)
            else:
                success, tracked_bbox = person_tracker.update(img)
                if success:
                    person_bbox = tracked_bbox

            # Get person keypoints
            person_keypoints = results[0].keypoints.data[0].cpu().numpy() if results[0].keypoints is not None else None

        landmarks = detect_facial_landmarks(processed_img)

        depth_map = estimate_depth(processed_img)

        refined_depth_map = refine_depth_with_person_info(depth_map, landmarks[0] if landmarks else None, person_bbox, person_keypoints)

        # Apply edge-aware filtering
        filtered_depth_map = apply_edge_aware_filter(refined_depth_map)

        # Create wireframe model
        wireframe_model = create_wireframe_model(landmarks[0] if landmarks else None, person_keypoints, filtered_depth_map)

        # Draw wireframe on the image
        draw_wireframe(img, wireframe_model)

        # Draw segmentation mask
        if len(results) > 0 and results[0].masks is not None:
            mask = results[0].masks.data[0].cpu().numpy()
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
            img[mask > 0.5] = img[mask > 0.5] * 0.5 + np.array([0, 0, 255]) * 0.5

        # Update person tracker
        if person_bbox is not None:
            person_depth = np.median(filtered_depth_map[int(person_bbox[1]):int(person_bbox[3]), int(person_bbox[0]):int(person_bbox[2])])
            person_tracker.update(img)

        avg_person_depth = person_tracker.bbox[2] - person_tracker.bbox[0] if person_tracker.bbox is not None else None
        if avg_person_depth is not None:
            cv2.putText(img, f"Avg Depth: {avg_person_depth:.2f}m", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Ensure depth_display has the same dimensions as img
        depth_display = cv2.applyColorMap((filtered_depth_map * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        depth_display = cv2.resize(depth_display, (img.shape[1], img.shape[0]))

        combined_display = np.hstack((img, depth_display))
        cv2.imshow("Person Depth Estimation", combined_display)

        # Save test image
        frame_count += 1
        if frame_count == 30 and not test_image_saved:
            cv2.imwrite("depth_estimation_test.jpg", combined_display)
            print("Test image saved as 'depth_estimation_test.jpg'")
            test_image_saved = True

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
