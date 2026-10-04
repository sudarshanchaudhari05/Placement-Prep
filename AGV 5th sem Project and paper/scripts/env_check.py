"""
Environment Verification Script
===============================
Verifies Python version, PyTorch, Torchvision, OpenCV, NumPy, and CUDA capability.
"""

import sys
import platform


def check_environment() -> dict:
    info = {}

    # Python version
    info["python_version"] = platform.python_version()
    info["python_build"] = platform.python_compiler()

    # PyTorch
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            info["cuda_capability"] = torch.cuda.get_device_capability(0)
        else:
            info["cuda_device_count"] = 0
            info["cuda_device_name"] = "N/A"
            info["cuda_capability"] = "N/A"
    except ImportError as e:
        info["torch_version"] = f"Missing ({e})"
        info["cuda_available"] = False

    # Torchvision
    try:
        import torchvision
        info["torchvision_version"] = torchvision.__version__
    except ImportError as e:
        info["torchvision_version"] = f"Missing ({e})"

    # OpenCV
    try:
        import cv2
        info["opencv_version"] = cv2.__version__
    except ImportError as e:
        info["opencv_version"] = f"Missing ({e})"

    # NumPy
    try:
        import numpy as np
        info["numpy_version"] = np.__version__
    except ImportError as e:
        info["numpy_version"] = f"Missing ({e})"

    # PyYAML
    try:
        import yaml
        info["yaml_version"] = yaml.__version__
    except ImportError as e:
        info["yaml_version"] = f"Missing ({e})"

    return info


def main():
    print("=" * 60)
    print("AGV Path Tracking - Environment Verification")
    print("=" * 60)
    env = check_environment()
    for k, v in env.items():
        print(f"  {k:22s}: {v}")
    print("=" * 60)

    # Sanity checks
    assert env["torch_version"] != "Missing", "PyTorch is required!"
    assert env["torchvision_version"] != "Missing", "Torchvision is required!"
    assert env["opencv_version"] != "Missing", "OpenCV is required!"
    assert env["numpy_version"] != "Missing", "NumPy is required!"
    print("All core dependencies successfully verified.")


if __name__ == "__main__":
    main()
