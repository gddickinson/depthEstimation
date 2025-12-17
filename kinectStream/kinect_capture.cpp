#include "kinect_capture.h"
#include <iostream>
#include <vector>
#include <cstring>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <thread>
#include <queue>

static freenect_context *f_ctx;
static freenect_device *f_dev;
static uint8_t *rgb_back, *rgb_mid, *rgb_front;
static uint16_t *depth_back, *depth_mid, *depth_front;
static std::mutex rgb_mutex, depth_mutex;
static std::condition_variable rgb_cv, depth_cv;
static bool got_rgb = false;
static bool got_depth = false;

#define FREENECT_FRAME_W 640
#define FREENECT_FRAME_H 480
#define FREENECT_VIDEO_RGB_SIZE (FREENECT_FRAME_W * FREENECT_FRAME_H * 3)
#define FREENECT_DEPTH_SIZE (FREENECT_FRAME_W * FREENECT_FRAME_H * sizeof(uint16_t))

struct FrameData {
    uint32_t timestamp;
    std::vector<uint8_t> data;
};

static std::queue<FrameData> rgb_queue;
static std::queue<FrameData> depth_queue;
static const size_t MAX_QUEUE_SIZE = 30;

static void depth_cb(freenect_device *dev, void *v_depth, uint32_t timestamp)
{
    std::lock_guard<std::mutex> lock(depth_mutex);
    uint16_t *depth = (uint16_t*)v_depth;

    FrameData frame;
    frame.timestamp = timestamp;
    frame.data.assign((uint8_t*)depth, (uint8_t*)depth + FREENECT_DEPTH_SIZE);

    if (depth_queue.size() >= MAX_QUEUE_SIZE) {
        depth_queue.pop();
    }
    depth_queue.push(frame);

    got_depth = true;
    depth_cv.notify_one();
    std::cout << "Depth callback: timestamp = " << timestamp << std::endl;
}

static void rgb_cb(freenect_device *dev, void *rgb, uint32_t timestamp)
{
    std::lock_guard<std::mutex> lock(rgb_mutex);

    FrameData frame;
    frame.timestamp = timestamp;
    frame.data.assign((uint8_t*)rgb, (uint8_t*)rgb + FREENECT_VIDEO_RGB_SIZE);

    if (rgb_queue.size() >= MAX_QUEUE_SIZE) {
        rgb_queue.pop();
    }
    rgb_queue.push(frame);

    got_rgb = true;
    rgb_cv.notify_one();
    std::cout << "RGB callback: timestamp = " << timestamp << std::endl;
}

int init_kinect()
{
    if (freenect_init(&f_ctx, NULL) < 0) {
        std::cerr << "freenect_init() failed" << std::endl;
        return -1;
    }

    freenect_set_log_level(f_ctx, FREENECT_LOG_DEBUG);
    freenect_select_subdevices(f_ctx, (freenect_device_flags)(FREENECT_DEVICE_MOTOR | FREENECT_DEVICE_CAMERA));

    int nr_devices = freenect_num_devices(f_ctx);
    std::cout << "Number of devices found: " << nr_devices << std::endl;

    if (nr_devices < 1) {
        std::cerr << "No devices found" << std::endl;
        freenect_shutdown(f_ctx);
        return -1;
    }

    if (freenect_open_device(f_ctx, &f_dev, 0) < 0) {
        std::cerr << "Could not open device" << std::endl;
        freenect_shutdown(f_ctx);
        return -1;
    }

    // Adjust USB transfer settings
    freenect_set_depth_buffer(f_dev, NULL);
    freenect_set_video_buffer(f_dev, NULL);

    freenect_set_depth_callback(f_dev, depth_cb);
    freenect_set_video_callback(f_dev, rgb_cb);

    freenect_frame_mode video_mode = freenect_find_video_mode(FREENECT_RESOLUTION_MEDIUM, FREENECT_VIDEO_RGB);
    if (!video_mode.is_valid) {
        std::cerr << "Invalid video mode" << std::endl;
        return -1;
    }
    if (freenect_set_video_mode(f_dev, video_mode) < 0) {
        std::cerr << "Failed to set video mode" << std::endl;
        return -1;
    }

    freenect_frame_mode depth_mode = freenect_find_depth_mode(FREENECT_RESOLUTION_MEDIUM, FREENECT_DEPTH_11BIT);
    if (!depth_mode.is_valid) {
        std::cerr << "Invalid depth mode" << std::endl;
        return -1;
    }
    if (freenect_set_depth_mode(f_dev, depth_mode) < 0) {
        std::cerr << "Failed to set depth mode" << std::endl;
        return -1;
    }

    if (freenect_start_depth(f_dev) < 0) {
        std::cerr << "Failed to start depth stream" << std::endl;
        return -1;
    }
    if (freenect_start_video(f_dev) < 0) {
        std::cerr << "Failed to start video stream" << std::endl;
        return -1;
    }

    return 0;
}

bool get_rgb(uint8_t *output, int timeout_ms)
{
    std::unique_lock<std::mutex> lock(rgb_mutex);
    if (rgb_cv.wait_for(lock, std::chrono::milliseconds(timeout_ms), []{ return !rgb_queue.empty(); })) {
        FrameData frame = rgb_queue.front();
        rgb_queue.pop();
        memcpy(output, frame.data.data(), FREENECT_VIDEO_RGB_SIZE);
        return true;
    }
    std::cerr << "RGB data timeout" << std::endl;
    return false;
}

bool get_depth(uint16_t *output, int timeout_ms)
{
    std::unique_lock<std::mutex> lock(depth_mutex);
    if (depth_cv.wait_for(lock, std::chrono::milliseconds(timeout_ms), []{ return !depth_queue.empty(); })) {
        FrameData frame = depth_queue.front();
        depth_queue.pop();
        memcpy(output, frame.data.data(), FREENECT_DEPTH_SIZE);
        return true;
    }
    std::cerr << "Depth data timeout" << std::endl;
    return false;
}

void shutdown_kinect()
{
    if (f_dev) {
        freenect_stop_depth(f_dev);
        freenect_stop_video(f_dev);
        freenect_close_device(f_dev);
    }
    if (f_ctx) {
        freenect_shutdown(f_ctx);
    }
}

void process_events()
{
    if (freenect_process_events(f_ctx) < 0) {
        std::cerr << "Error processing freenect events" << std::endl;
    }
}
