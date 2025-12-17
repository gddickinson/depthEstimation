import cv2
import numpy as np

def list_camera_devices():
    index = 0
    arr = []
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            arr.append(index)
        cap.release()
        index += 1
    return arr

def show_camera_feed(index, window_name):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"Could not open camera at index {index}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Can't receive frame from camera at index {index}. Exiting...")
            break

        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyWindow(window_name)

def main():
    devices = list_camera_devices()
    print(f"Available camera indices: {devices}")

    for index in devices:
        print(f"\nTesting camera at index {index}")
        print("Press 'q' to close the window and move to the next camera.")
        show_camera_feed(index, f"Camera {index}")

    print("\nDevice testing completed.")

if __name__ == "__main__":
    main()
