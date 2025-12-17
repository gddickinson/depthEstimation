#ifndef KINECT_CAPTURE_H
#define KINECT_CAPTURE_H

#include <cstdint>
#include <libfreenect.h>

#ifdef __cplusplus
extern "C" {
#endif

int init_kinect();
bool get_rgb(uint8_t *output, int timeout_ms);
bool get_depth(uint16_t *output, int timeout_ms);
void shutdown_kinect();
void process_events();

#ifdef __cplusplus
}
#endif

#endif // KINECT_CAPTURE_H
