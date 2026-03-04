"""
Device selection for YOLO model (GPU/CPU).
"""
import logging
from typing import Literal

logger = logging.getLogger(__name__)

DeviceType = Literal["cuda", "cpu"]


def select_device(prefer_gpu: bool = True) -> DeviceType:
    """
    Select the best available device for YOLO inference.
    
    Args:
        prefer_gpu: Whether to prefer GPU if available
    
    Returns:
        Device string: "cuda" or "cpu"
    """
    if not prefer_gpu:
        logger.info("CPU mode selected by configuration")
        print("✓ select_device completed: cpu (by config)")
        return "cpu"
    
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU available: {device_name}")
            print(f"✓ select_device completed: cuda ({device_name})")
            return "cuda"
        else:
            logger.info("No GPU available, using CPU")
            print("✓ select_device completed: cpu (no GPU)")
            return "cpu"
    except ImportError:
        logger.warning("PyTorch not available, defaulting to CPU")
        print("✓ select_device completed: cpu (no torch)")
        return "cpu"


def get_device_info() -> dict:
    """
    Get detailed information about the selected device.
    
    Returns:
        Dictionary with device information
    """
    info = {
        "device": "cpu",
        "gpu_available": False,
        "gpu_name": None,
        "gpu_memory_gb": None
    }
    
    try:
        import torch
        if torch.cuda.is_available():
            info["device"] = "cuda"
            info["gpu_available"] = True
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
    except ImportError:
        pass
    
    print(f"✓ get_device_info completed: {info['device']}")
    return info


print("✓ detection.device module loaded")
