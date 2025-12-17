from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import os

freenect_include = "/opt/homebrew/Cellar/libfreenect/0.7.5/include/libfreenect"
freenect_lib = "/opt/homebrew/Cellar/libfreenect/0.7.5/lib"
libusb_include = "/opt/homebrew/Cellar/libusb/1.0.27/include/libusb-1.0"
libusb_lib = "/opt/homebrew/Cellar/libusb/1.0.27/lib"

ext_modules = [
    Extension(
        "kinect_wrapper",
        sources=["kinect_wrapper.pyx", "kinect_capture.cpp"],  # Removed 'kinect_capture.h' from here
        libraries=["freenect", "usb-1.0"],
        include_dirs=[numpy.get_include(), freenect_include, libusb_include, "."],
        library_dirs=[freenect_lib, libusb_lib],
        extra_compile_args=["-std=c++11", f"-I{freenect_include}", f"-I{libusb_include}"],
        extra_link_args=[f"-L{freenect_lib}", f"-L{libusb_lib}", "-lfreenect", "-lusb-1.0"],
        language="c++"
    )
]

setup(
    name="kinect_wrapper",
    ext_modules=cythonize(ext_modules, language_level="3"),
    include_dirs=[numpy.get_include()]
)
