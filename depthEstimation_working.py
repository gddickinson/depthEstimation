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

def refine_depth_with_objects_and_face(depth_map, objects, landmarks):
    if objects.masks is not None:
        for seg in objects.masks.data:
            mask = seg.cpu().numpy()
            mask = cv2.resize(mask, (depth_map.shape[1], depth_map.shape[0]))
            mask = (mask > 0.5).astype(np.uint8)  # Convert to uint8
            obj_depth = np.median(depth_map[mask > 0])
            depth_map[mask > 0] = obj_depth * 0.9  # Make objects slightly closer

    for landmark in landmarks:
        face_points = np.array([(p.x, p.y) for p in landmark.parts()])
        face_mask = np.zeros(depth_map.shape, dtype=np.uint8)
        cv2.fillConvexPoly(face_mask, face_points, 1)
        face_depth = np.median(depth_map[face_mask > 0])
        depth_map[face_mask > 0] = face_depth * 0.8  # Make face more prominent

    return depth_map

def estimate_depth(frame):
    input_image = frame.astype(np.float32) / 255.0
    depth_map = depth_model.predict(np.expand_dims(input_image, axis=0))[0, :, :, 0]
    return depth_map

def apply_edge_aware_filter(image, depth_map):
    d = 7
    sigma_color = 100
    sigma_space = 100
    filtered_depth = cv2.bilateralFilter(depth_map.astype(np.float32), d, sigma_color, sigma_space)
    return filtered_depth

def main():
    print("Initializing camera...")
    cap = cv2.VideoCapture(0)
    cap.set(3, frameWidth)
    cap.set(4, frameHeight)
    cap.set(10, 150)

    frame_count = 0
    test_image_saved = False

    while cap.isOpened():
        success, img = cap.read()
        if success:
            processed_img = process_image(img)

            print("Detecting objects and segmenting...")
            objects = detect_objects_and_segment(processed_img)

            print("Detecting person details...")
            person_details = detect_person_details(processed_img)

            print("Detecting facial landmarks...")
            landmarks = detect_facial_landmarks(processed_img)

            print("Estimating depth...")
            depth_map = estimate_depth(processed_img)

            print("Refining depth map...")
            refined_depth_map = refine_depth_with_objects_and_face(depth_map, objects, landmarks)

            print("Applying edge-aware filter...")
            filtered_depth_map = apply_edge_aware_filter(processed_img, refined_depth_map)

            depth_map_resized = cv2.resize(filtered_depth_map, (frameWidth, frameHeight))
            depth_display = cv2.applyColorMap((depth_map_resized * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)

            display_img = cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR)

            for landmark in landmarks:
                for point in landmark.parts():
                    cv2.circle(display_img, (point.x, point.y), 3, (0, 255, 0), -1)

            combined_display = np.hstack((display_img, depth_display))
            cv2.imshow("Original and Depth", combined_display)

            frame_count += 1
            if frame_count == 10 and not test_image_saved:
                cv2.imwrite("depth_estimation_test_enhanced.jpg", combined_display)
                print("Test image saved as 'depth_estimation_test_enhanced.jpg'")
                test_image_saved = True

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Process completed.")

if __name__ == "__main__":
    main()
