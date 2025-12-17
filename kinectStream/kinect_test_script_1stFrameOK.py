import kinect_wrapper
import cv2
import numpy as np
import matplotlib.pyplot as plt

def pretty_depth(depth):
    """Converts depth into a 'nicer' format for display
    This is abstracted to allow for experimentation with normalization"""
    np.clip(depth, 0, 2**10 - 1, depth)
    depth >>= 2
    depth = depth.astype(np.uint8)
    return depth

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

    try:
        i = 0
        while i<1:
            raw_depth = get_depth()
            video = get_video()

            # Print info about the frames
            print(f"Video shape: {video.shape}, dtype: {video.dtype}")
            print(f"Depth shape: {raw_depth.shape}, dtype: {raw_depth.dtype}")
            print(f"Video min: {video.min()}, max: {video.max()}")
            print(f"Depth min: {raw_depth.min()}, max: {raw_depth.max()}")

            # Process depth data
            depth = pretty_depth(raw_depth.copy())

            # Convert RGB from RGB to BGR for OpenCV
            video = cv2.cvtColor(video, cv2.COLOR_RGB2BGR)

            # Apply colormap to depth image
            depth_colormap = cv2.applyColorMap(depth, cv2.COLORMAP_JET)

            # Display the images
            cv2.imshow('Depth', depth_colormap)
            cv2.imshow('Video', video)

            # Save the images
            cv2.imwrite("rgb_image.jpg", video)
            cv2.imwrite("depth_image.png", depth_colormap)

            # Create a side-by-side comparison
            comparison = np.hstack((video, depth_colormap))
            cv2.imshow("RGB and Depth Comparison", comparison)
            cv2.imwrite("comparison.jpg", comparison)

            if cv2.waitKey(10) == 27:  # Exit on ESC
                break

            i+=1

    finally:
        kinect_wrapper.shutdown()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
