# kinect_wrapper.pyx
import numpy as np
cimport numpy as np

cdef extern from "kinect_capture.h":
    int init_kinect()
    void get_rgb(unsigned char *output)
    void get_depth(unsigned short *output)
    void shutdown_kinect()

def initialize():
    return init_kinect()

def get_rgb_frame():
    cdef np.ndarray[np.uint8_t, ndim=3] frame = np.empty((480, 640, 3), dtype=np.uint8)
    get_rgb(<unsigned char*>frame.data)
    return frame

def get_depth_frame():
    cdef np.ndarray[np.uint16_t, ndim=2] frame = np.empty((480, 640), dtype=np.uint16)
    get_depth(<unsigned short*>frame.data)
    return frame

def shutdown():
    shutdown_kinect()
