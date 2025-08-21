"""
Core data types and interfaces for RobotVLA system.

This module defines the fundamental data structures used throughout the system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from PIL import Image


class ActionType(Enum):
    """Types of robot actions."""
    POSITION = "position"           # Position control
    VELOCITY = "velocity"           # Velocity control  
    FORCE = "force"                # Force control
    JOINT = "joint"                # Joint control
    GRIPPER = "gripper"            # Gripper control
    COMPOSITE = "composite"        # Composite action


class ExecutionStatus(Enum):
    """Status of action execution."""
    SUCCESS = "success"
    FAILED = "failed"
    IN_PROGRESS = "in_progress"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class VisionFeatures:
    """Standardized vision features from any vision model."""
    features: torch.Tensor          # Main feature tensor
    attention_mask: Optional[torch.Tensor] = None
    spatial_features: Optional[torch.Tensor] = None  # For spatial reasoning
    metadata: Dict[str, Any] = None
    
    @property
    def shape(self) -> torch.Size:
        return self.features.shape
    
    @property 
    def device(self) -> torch.device:
        return self.features.device


@dataclass
class RobotAction:
    """Represents a robot action in standardized format."""
    action_type: ActionType
    values: np.ndarray              # Action values (e.g., 7DOF for position)
    confidence: float = 1.0         # Confidence score
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        """Validate action format."""
        if not isinstance(self.values, np.ndarray):
            self.values = np.array(self.values)
        
        # Ensure values are float32
        if self.values.dtype != np.float32:
            self.values = self.values.astype(np.float32)


@dataclass
class FunctionCall:
    """Represents a robot function call."""
    function_name: str              # Name of the function to call
    parameters: Dict[str, Any]      # Function parameters
    robot_id: Optional[str] = None  # Target robot identifier
    priority: int = 0               # Execution priority (higher = more urgent)
    timeout: float = 30.0           # Execution timeout in seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "function_name": self.function_name,
            "parameters": self.parameters,
            "robot_id": self.robot_id,
            "priority": self.priority,
            "timeout": self.timeout,
        }


@dataclass
class ExecutionResult:
    """Result of executing a robot function."""
    status: ExecutionStatus
    result: Any = None              # Function return value
    error: Optional[str] = None     # Error message if failed
    execution_time: float = 0.0     # Time taken to execute
    metadata: Dict[str, Any] = None
    
    @property
    def success(self) -> bool:
        return self.status == ExecutionStatus.SUCCESS
    
    @property
    def failed(self) -> bool:
        return self.status == ExecutionStatus.FAILED


@dataclass
class TaskContext:
    """Context information for a robot task."""
    task_id: str
    instruction: str                # Natural language instruction
    images: List[Image.Image]       # Input images
    history: List[Dict[str, Any]] = None  # Previous actions/results
    constraints: Dict[str, Any] = None    # Task constraints
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.history is None:
            self.history = []
        if self.constraints is None:
            self.constraints = {}
        if self.metadata is None:
            self.metadata = {}


# Abstract Interfaces

class VisionModelInterface(ABC):
    """Abstract interface for any vision model."""
    
    @abstractmethod
    def encode(self, images: List[Image.Image]) -> VisionFeatures:
        """Extract features from images."""
        pass
    
    @abstractmethod
    def get_feature_dim(self) -> int:
        """Get the dimensionality of extracted features."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name/identifier."""
        pass


class RobotInterface(ABC):
    """Abstract interface for robot control."""
    
    @abstractmethod
    def execute_function(self, function_call: FunctionCall) -> ExecutionResult:
        """Execute a robot function call."""
        pass
    
    @abstractmethod
    def get_available_functions(self) -> List[str]:
        """Get list of available function names."""
        pass
    
    @abstractmethod
    def get_function_schema(self, function_name: str) -> Dict[str, Any]:
        """Get the schema/signature of a function."""
        pass
    
    @property
    @abstractmethod
    def robot_id(self) -> str:
        """Get the robot identifier."""
        pass


class LanguageModelInterface(ABC):
    """Abstract interface for language models."""
    
    @abstractmethod
    def process_instruction(
        self, 
        instruction: str, 
        context: TaskContext,
        vision_features: VisionFeatures
    ) -> List[FunctionCall]:
        """Process instruction and generate function calls."""
        pass
    
    @abstractmethod
    def update_context(self, context: TaskContext, result: ExecutionResult) -> None:
        """Update context with execution results."""
        pass 