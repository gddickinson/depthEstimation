#include "kinect_capture.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <atomic>
#include <iomanip>

#define FREENECT_FRAME_W 640
#define FREENECT_FRAME_H 480
#define FREENECT_VIDEO_RGB_SIZE (FREENECT_FRAME_W * FREENECT_FRAME_H * 3)
#define FREENECT_DEPTH_SIZE (FREENECT_FRAME_W * FREENECT_FRAME_H * sizeof(uint16_t))

std::atomic<bool> timeout_occurred(false);

void timeout_thread(int timeout_seconds) {
    std::this_thread::sleep_for(std::chrono::seconds(timeout_seconds));
    timeout_occurred.store(true);
    std::cout << "Timeout occurred after " << timeout_seconds << " seconds." << std::endl;
}

bool get_frame_data(uint8_t* rgb_data, uint16_t* depth_data, int max_retries = 3) {
    for (int i = 0; i < max_retries; ++i) {
        std::cout << "Attempt " << (i + 1) << " to get frame data..." << std::endl;

        process_events();

        std::cout << "Getting RGB data..." << std::endl;
        if (get_rgb(rgb_data, 1000)) { // 1 second timeout for RGB
            std::cout << "RGB data received." << std::endl;
        } else {
            std::cout << "Timeout while getting RGB data." << std::endl;
            continue;
        }

        std::cout << "Getting depth data..." << std::endl;
        if (get_depth(depth_data, 1000)) { // 1 second timeout for depth
            std::cout << "Depth data received." << std::endl;
            return true;
        } else {
            std::cout << "Timeout while getting depth data." << std::endl;
        }
    }
    std::cout << "Failed to get frame data after " << max_retries << " attempts." << std::endl;
    return false;
}

void print_frame_info(const uint8_t* rgb_data, const uint16_t* depth_data) {
    std::cout << "RGB data (first 30 bytes): ";
    for (int i = 0; i < 30; i++) {
        std::cout << std::setw(3) << static_cast<int>(rgb_data[i]) << " ";
    }
    std::cout << std::endl;

    std::cout << "Depth data (first 30 values): ";
    for (int i = 0; i < 30; i++) {
        std::cout << std::setw(5) << depth_data[i] << " ";
    }
    std::cout << std::endl;
}

int main() {
    std::cout << "Initializing Kinect..." << std::endl;
    if (init_kinect() < 0) {
        std::cerr << "Failed to initialize Kinect" << std::endl;
        return -1;
    }
    std::cout << "Kinect initialized successfully." << std::endl;

    uint8_t* rgb_data = new uint8_t[FREENECT_VIDEO_RGB_SIZE];
    uint16_t* depth_data = new uint16_t[FREENECT_FRAME_H * FREENECT_FRAME_W];

    int frame_count = 0;
    int failed_frames = 0;
    auto start_time = std::chrono::high_resolution_clock::now();

    std::cout << "Starting frame capture loop..." << std::endl;

    // Start timeout thread
    std::thread timeout_t(timeout_thread, 60); // 60 seconds timeout

    while (frame_count < 100 && failed_frames < 10 && !timeout_occurred.load()) {
        if (get_frame_data(rgb_data, depth_data)) {
            frame_count++;
            failed_frames = 0; // Reset failed frames counter on success

            if (frame_count % 10 == 0) {
                auto current_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time);
                double fps = frame_count / (duration.count() / 1000.0);

                std::cout << "Frame " << frame_count << std::endl;
                std::cout << "FPS: " << fps << std::endl;

                print_frame_info(rgb_data, depth_data);
            }
        } else {
            failed_frames++;
            std::cout << "Failed to get frame. Failed frames: " << failed_frames << std::endl;
        }

        // Add a small delay to avoid overwhelming the CPU
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    if (timeout_occurred.load()) {
        std::cout << "Program timed out. Exiting..." << std::endl;
    } else if (failed_frames >= 10) {
        std::cout << "Too many failed frames. Exiting..." << std::endl;
    } else {
        std::cout << "Frame capture complete." << std::endl;
    }

    std::cout << "Cleaning up..." << std::endl;
    delete[] rgb_data;
    delete[] depth_data;
    shutdown_kinect();
    std::cout << "Cleanup complete. Exiting." << std::endl;

    // Wait for timeout thread to finish
    timeout_t.join();

    return 0;
}
