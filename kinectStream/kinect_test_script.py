import kinect_wrapper
import cv2
import numpy as np
import time

def process_depth(depth):
    """Processes the depth data assuming it's already in 8-bit format"""
    if depth.dtype != np.uint8:
        depth = depth.astype(np.uint8)

    # Apply a colormap for visualization
    depth_colormap = cv2.applyColorMap(depth, cv2.COLORMAP_JET)

    return depth_colormap

def get_video():
    return kinect_wrapper.get_rgb_frame()

def get_depth():
    return kinect_wrapper.get_depth_frame()

def main():
    if kinect_wrapper.initialize() < 0:
        print("Failed to initialize Kinect")
        return

    cv2.namedWindow('Depth')
    cv2.namedWindow('Video')
    print('Press ESC in window to stop')

    frame_count = 0
    start_time = time.time()
    last_depth = None

    try:
        while True:
            frame_start = time.time()

            raw_depth = get_depth()
            video = get_video()

            frame_count += 1

            # Check if depth data has changed
            if last_depth is not None and np.array_equal(raw_depth, last_depth):
                print(f"Warning: Depth frame {frame_count} is identical to the previous frame")
            last_depth = raw_depth.copy()

            if frame_count % 30 == 0:  # Print debug info every 30 frames
                print(f"Frame {frame_count}:")
                print(f"  Video shape: {video.shape}, dtype: {video.dtype}")
                print(f"  Depth shape: {raw_depth.shape}, dtype: {raw_depth.dtype}")
                print(f"  Video min: {video.min()}, max: {video.max()}")
                print(f"  Depth min: {raw_depth.min()}, max: {raw_depth.max()}")
                print(f"  Depth unique values: {np.unique(raw_depth).size}")
                print(f"  FPS: {frame_count / (time.time() - start_time):.2f}")

            # Process depth data
            depth_colormap = process_depth(raw_depth)

            # Convert RGB from RGB to BGR for OpenCV
            video = cv2.cvtColor(video, cv2.COLOR_RGB2BGR)

            # Flip both video and depth horizontally (mirror mode)
            video = cv2.flip(video, 1)
            depth_colormap = cv2.flip(depth_colormap, 1)

            # Display the images
            cv2.imshow('Depth', depth_colormap)
            cv2.imshow('Video', video)

            # Save images every 60 frames
            if frame_count % 60 == 0:
                cv2.imwrite(f"rgb_image_{frame_count}.jpg", video)
                cv2.imwrite(f"depth_image_{frame_count}.png", depth_colormap)
                comparison = np.hstack((video, depth_colormap))
                cv2.imwrite(f"comparison_{frame_count}.jpg", comparison)
                frame_end = time.time()
                print(f"Frame processing time: {(frame_end - frame_start) * 1000:.2f} ms")

            if cv2.waitKey(1) == 27:  # Exit on ESC
                break



    finally:
        kinect_wrapper.shutdown()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
