import ctypes
import platform

if platform.system() == "Linux":
    try:
        ctypes.cdll.LoadLibrary("/usr/lib/aarch64-linux-gnu/libgomp.so.1")
        print("libgomp loaded successfully.")
    except OSError as e:
        print(f"Error loading libgomp: {e}")
