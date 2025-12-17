import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, UpSampling2D, Input
from tensorflow.keras.models import Model
from mpl_toolkits.axes_grid1 import make_axes_locatable

print("Starting the script...")

def create_depth_model(input_shape):
    inputs = Input(shape=input_shape)

    # Simple encoder
    x = Conv2D(32, 3, strides=2, padding='same', activation='relu')(inputs)
    x = Conv2D(64, 3, strides=2, padding='same', activation='relu')(x)
    x = Conv2D(128, 3, strides=2, padding='same', activation='relu')(x)

    # Simple decoder
    x = UpSampling2D()(x)
    x = Conv2D(64, 3, padding='same', activation='relu')(x)
    x = UpSampling2D()(x)
    x = Conv2D(32, 3, padding='same', activation='relu')(x)
    x = UpSampling2D()(x)
    outputs = Conv2D(1, 3, padding='same', activation='linear')(x)

    return Model(inputs=inputs, outputs=outputs)

print("Creating the model...")
model = create_depth_model((240, 320, 3))
print("Model created successfully.")

# Open the camera
print("Opening the camera...")
frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(1)  # Use index 1 as it worked in your previous script
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, 150)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Camera opened successfully. Capturing frames...")

# Capture a few frames to ensure camera is initialized
for _ in range(10):
    success, frame = cap.read()
    if not success:
        print("Failed to capture frame.")
        cap.release()
        exit()

print("Frame captured successfully.")
cap.release()

print(f"Frame shape: {frame.shape}")

def process_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (320, 240))  # Resize to match model input
    image = image.astype(np.float32) / 255.0
    return np.expand_dims(image, axis=0)

def edge_aware_filter(image, depth):
    # Guided filter
    depth = cv2.ximgproc.guidedFilter(image, depth, radius=8, eps=1e-4)
    return depth

def post_process(depth_map, original_image):
    depth_map = cv2.resize(depth_map, (frameWidth, frameHeight))
    depth_map = edge_aware_filter(cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY), depth_map)
    return depth_map

print("Processing frame...")
# Prepare image for the model
input_image = process_image(frame)

# Perform inference
print("Performing inference...")
depth_map = model.predict(input_image)[0, :, :, 0]

# Post-process the depth map
depth_map = post_process(depth_map, frame)

# Assume a maximum depth of 5 meters
max_depth = 5  # meters

# Normalize depth map for visualization
depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
depth_map_meters = depth_map * max_depth

# Plot and display the results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Original Image
ax1.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
ax1.set_title("Original Image")
ax1.axis('off')

# Depth Map
im = ax2.imshow(depth_map_meters, cmap='viridis')
ax2.set_title("Depth Map")
ax2.axis('off')

# Add colorbar
divider = make_axes_locatable(ax2)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(im, cax=cax)
cbar.set_label('Estimated Depth (meters)', rotation=270, labelpad=15)

plt.tight_layout()
plt.savefig("comparison_with_scale.png", dpi=300, bbox_inches='tight')
plt.show()

print("Images saved and displayed. Script finished.")
