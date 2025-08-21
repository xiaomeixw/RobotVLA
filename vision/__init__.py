"""
Universal Vision Interface for RobotVLA

This module provides adapters for integrating any vision model into the RobotVLA framework.
"""

from .adapters import (
    CLIPAdapter,
    DINOv2Adapter, 
    CustomVisionAdapter,
    VisionModelAdapter,
)
from .registry import VisionModelRegistry
from .utils import (
    preprocess_image,
    normalize_features,
    resize_image,
)

__all__ = [
    # Adapters
    "CLIPAdapter",
    "DINOv2Adapter",
    "CustomVisionAdapter", 
    "VisionModelAdapter",
    
    # Registry
    "VisionModelRegistry",
    
    # Utilities
    "preprocess_image",
    "normalize_features", 
    "resize_image",
] 