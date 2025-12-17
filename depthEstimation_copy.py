import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, UpSampling2D, Input
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Constants for image dimensions
WIDTH = 320
HEIGHT = 240

# Create depth model (as before)
def create_depth_model(input_shape):
    inputs = Input(shape=input_shape)

    x = Conv2D(32, 3, strides=2, padding='same', activation='relu')(inputs)
    x = Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)
    x = Conv2D(128, 3, strides=2, padding='same', activation='relu')(x)

    x = UpSampling2D()(x)
    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    x = UpSampling2D()(x)
    x = Conv2D(32, 3, padding='same', activation='relu')(x)
    x = UpSampling2D()(x)
    outputs = Conv2D(1, 3, padding='same', activation='linear')(x)

    return Model(inputs=inputs, outputs=outputs)

def process_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (WIDTH, HEIGHT))
    image = image.astype(np.float32) / 255.0
    return np.expand_dims(image, axis=0)

def detect_objects(frame):
    results = yolo_model(frame)
    return results[0].boxes.data.cpu().numpy()

def refine_depth_with_objects(depth_map, objects, original_shape):
    scale_y, scale_x = depth_map.shape[0] / original_shape[0], depth_map.shape[1] / original_shape[1]
    for obj in objects:
        x1, y1, x2, y2, conf, class_id = obj
        x1, y1, x2, y2 = int(x1*scale_x), int(y1*scale_y), int(x2*scale_x), int(y2*scale_y)
        obj_depth = np.mean(depth_map[y1:y2, x1:x2])
        depth_map[y1:y2, x1:x2] = obj_depth
    return depth_map

def capture_multiple_frames(cap, num_frames=5):
    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if ret:
            frames.append(cv2.resize(frame, (WIDTH, HEIGHT)))
    return frames

def average_depth_maps(depth_maps):
    return np.mean(depth_maps, axis=0)

def detect_edges(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    return edges

def create_confidence_map(frame, depth_map, edges):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    confidence = (gray.astype(float) / 255) * 0.5 + (edges.astype(float) / 255) * 0.5
    return confidence

def visualize_results(frame, depth_map, confidence_map, objects):
    fig, axs = plt.subplots(2, 2, figsize=(20, 20))

    axs[0, 0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title('Original Image')
    axs[0, 0].axis('off')

    depth_display = axs[0, 1].imshow(depth_map, cmap='viridis')
    axs[0, 1].set_title('Depth Map')
    axs[0, 1].axis('off')
    plt.colorbar(depth_display, ax=axs[0, 1])

    conf_display = axs[1, 0].imshow(confidence_map, cmap='hot')
    axs[1, 0].set_title('Confidence Map')
    axs[1, 0].axis('off')
    plt.colorbar(conf_display, ax=axs[1, 0])

    axs[1, 1].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    axs[1, 1].set_title('Detected Objects')
    for obj in objects:
        x1, y1, x2, y2, conf, class_id = obj
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='r', linewidth=2)
        axs[1, 1].add_patch(rect)
    axs[1, 1].axis('off')

    plt.tight_layout()
    plt.savefig('enhanced_depth_estimation_results.png')
    plt.show()

def main():
    print("Initializing...")
    cap = cv2.VideoCapture(1)
    depth_model = create_depth_model((HEIGHT, WIDTH, 3))

    global yolo_model
    yolo_model = YOLO("yolov8n-seg.pt")

    print("Capturing frames...")
    frames = capture_multiple_frames(cap)

    depth_maps = []
    for frame in frames:
        input_image = process_image(frame)
        depth_map = depth_model.predict(input_image)[0, :, :, 0]
        depth_maps.append(depth_map)

    print("Processing depth maps...")
    avg_depth_map = average_depth_maps(depth_maps)

    print("Detecting objects...")
    objects = detect_objects(frames[-1])

    print("Refining depth map...")
    refined_depth_map = refine_depth_with_objects(avg_depth_map, objects, frames[-1].shape[:2])

    print("Detecting edges...")
    edges = detect_edges(frames[-1])

    print("Creating confidence map...")
    confidence_map = create_confidence_map(frames[-1], refined_depth_map, edges)

    print("Final refinement...")
    final_depth_map = refined_depth_map * confidence_map

    print("Visualizing results...")
    visualize_results(frames[-1], final_depth_map, confidence_map, objects)

    cap.release()
    print("Process completed.")

if __name__ == "__main__":
    main()
