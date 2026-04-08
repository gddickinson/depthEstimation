"""
Build script for the Kinect wrapper Cython extension.

Requires:
  - libfreenect (installed via Homebrew or system package manager)
  - libusb
  - Cython
  - numpy

If libfreenect is not installed, the build will fail with a clear error message.
"""
import os
import sys

try:
    from setuptools import setup, Extension
except ImportError:
    sys.exit("Error: setuptools is required. Install with: pip install setuptools")

try:
    from Cython.Build import cythonize
except ImportError:
    sys.exit("Error: Cython is required to build kinect_wrapper. Install with: pip install cython")

try:
    import numpy as np
except ImportError:
    sys.exit("Error: numpy is required. Install with: pip install numpy")


def find_libfreenect():
    """Find libfreenect installation paths, checking common locations."""
    # Common Homebrew paths (Apple Silicon and Intel)
    search_paths = [
        "/opt/homebrew/Cellar/libfreenect",
        "/usr/local/Cellar/libfreenect",
        "/opt/homebrew/opt/libfreenect",
        "/usr/local/opt/libfreenect",
        "/usr",
        "/usr/local",
    ]

    for base in search_paths:
        if not os.path.exists(base):
            continue

        # Check for versioned directories (e.g., /opt/homebrew/Cellar/libfreenect/0.7.5)
        if "Cellar" in base:
            versions = sorted(os.listdir(base), reverse=True) if os.path.isdir(base) else []
            for ver in versions:
                include_dir = os.path.join(base, ver, "include", "libfreenect")
                lib_dir = os.path.join(base, ver, "lib")
                if os.path.exists(include_dir) and os.path.exists(lib_dir):
                    return include_dir, lib_dir
        else:
            include_dir = os.path.join(base, "include", "libfreenect")
            lib_dir = os.path.join(base, "lib")
            if os.path.exists(include_dir) and os.path.exists(lib_dir):
                return include_dir, lib_dir

    return None, None


def find_libusb():
    """Find libusb installation paths."""
    search_paths = [
        "/opt/homebrew/Cellar/libusb",
        "/usr/local/Cellar/libusb",
        "/opt/homebrew/opt/libusb",
        "/usr/local/opt/libusb",
        "/usr",
        "/usr/local",
    ]

    for base in search_paths:
        if not os.path.exists(base):
            continue

        if "Cellar" in base:
            versions = sorted(os.listdir(base), reverse=True) if os.path.isdir(base) else []
            for ver in versions:
                include_dir = os.path.join(base, ver, "include", "libusb-1.0")
                lib_dir = os.path.join(base, ver, "lib")
                if os.path.exists(include_dir) and os.path.exists(lib_dir):
                    return include_dir, lib_dir
        else:
            include_dir = os.path.join(base, "include", "libusb-1.0")
            lib_dir = os.path.join(base, "lib")
            if os.path.exists(include_dir) and os.path.exists(lib_dir):
                return include_dir, lib_dir

    return None, None


# Find library paths
freenect_include, freenect_lib = find_libfreenect()
usb_include, usb_lib = find_libusb()

if freenect_include is None or freenect_lib is None:
    sys.exit(
        "Error: libfreenect not found.\n"
        "Install it with: brew install libfreenect\n"
        "Or set FREENECT_INCLUDE_DIR and FREENECT_LIB_DIR environment variables."
    )

if usb_include is None or usb_lib is None:
    sys.exit(
        "Error: libusb not found.\n"
        "Install it with: brew install libusb\n"
        "Or set LIBUSB_INCLUDE_DIR and LIBUSB_LIB_DIR environment variables."
    )

# Allow environment variable overrides
freenect_include = os.environ.get("FREENECT_INCLUDE_DIR", freenect_include)
freenect_lib = os.environ.get("FREENECT_LIB_DIR", freenect_lib)
usb_include = os.environ.get("LIBUSB_INCLUDE_DIR", usb_include)
usb_lib = os.environ.get("LIBUSB_LIB_DIR", usb_lib)

print(f"Using libfreenect: include={freenect_include}, lib={freenect_lib}")
print(f"Using libusb: include={usb_include}, lib={usb_lib}")

extensions = [
    Extension(
        "kinect_wrapper",
        ["kinect_wrapper.pyx", "kinect_capture.cpp"],
        include_dirs=[
            np.get_include(),
            freenect_include,
            usb_include,
            ".",
        ],
        library_dirs=[freenect_lib, usb_lib],
        libraries=["freenect", "usb-1.0"],
        extra_compile_args=["-std=c++11", "-stdlib=libc++"],
        extra_link_args=["-stdlib=libc++"],
        language="c++",
    )
]

setup(
    name="kinect_wrapper",
    ext_modules=cythonize(extensions),
    install_requires=["numpy"],
)
