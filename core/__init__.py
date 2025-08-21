"""
RobotVLA Core Framework

This module provides the foundational interfaces and utilities for the RobotVLA system.
"""

from .config import RobotVLAConfig, VisionModelConfig, RobotConfig
from .pipeline import RobotVLAPipeline
from .types import (
    VisionFeatures,
    RobotAction,
    FunctionCall,
    ExecutionResult,
    TaskContext,
)

__version__ = "0.1.0"
__all__ = [
    # Configuration
    "RobotVLAConfig",
    "VisionModelConfig", 
    "RobotConfig",
    
    # Main Pipeline
    "RobotVLAPipeline",
    
    # Core Types
    "VisionFeatures",
    "RobotAction", 
    "FunctionCall",
    "ExecutionResult",
    "TaskContext",
] 