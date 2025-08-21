"""
Configuration management for RobotVLA system.

Uses Pydantic for type validation and configuration management.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, validator
import yaml


class VisionModelConfig(BaseModel):
    """Configuration for vision models."""
    
    model_type: str = Field(..., description="Type of vision model (e.g., 'clip', 'dinov2', 'custom')")
    model_name: str = Field(..., description="Specific model name or path")
    pretrained: bool = Field(True, description="Whether to use pretrained weights")
    device: str = Field("auto", description="Device to run model on")
    batch_size: int = Field(1, description="Batch size for processing")
    
    # Model-specific parameters
    image_size: int = Field(224, description="Input image size")
    normalize: bool = Field(True, description="Whether to normalize inputs")
    feature_dim: Optional[int] = Field(None, description="Output feature dimension")
    
    # Advanced options
    custom_config: Dict[str, Any] = Field(default_factory=dict, description="Model-specific configuration")
    
    @validator('device')
    def validate_device(cls, v):
        if v not in ['auto', 'cpu', 'cuda'] and not v.startswith('cuda:'):
            raise ValueError("Device must be 'auto', 'cpu', 'cuda', or 'cuda:N'")
        return v


class RobotConfig(BaseModel):
    """Configuration for robot interfaces."""
    
    robot_id: str = Field(..., description="Unique robot identifier")
    robot_type: str = Field(..., description="Type of robot (e.g., 'widowx', 'franka', 'universal')")
    connection_type: str = Field("local", description="Connection type: 'local', 'remote', 'simulation'")
    
    # Connection parameters
    host: Optional[str] = Field(None, description="Host address for remote robots")
    port: Optional[int] = Field(None, description="Port for remote connection")
    device_path: Optional[str] = Field(None, description="Device path for local connection")
    
    # Robot capabilities
    dof: int = Field(7, description="Degrees of freedom")
    has_gripper: bool = Field(True, description="Whether robot has a gripper")
    workspace_bounds: Optional[List[List[float]]] = Field(None, description="Workspace bounds [[x_min, x_max], [y_min, y_max], [z_min, z_max]]")
    
    # Control parameters
    max_velocity: float = Field(1.0, description="Maximum velocity limit")
    max_acceleration: float = Field(2.0, description="Maximum acceleration limit")
    safety_limits: Dict[str, float] = Field(default_factory=dict, description="Safety limits")
    
    # Function registration
    function_registry_path: Optional[str] = Field(None, description="Path to function registry file")
    custom_functions: Dict[str, Any] = Field(default_factory=dict, description="Custom function definitions")


class LanguageModelConfig(BaseModel):
    """Configuration for language models."""
    
    model_type: str = Field(..., description="Type of language model")
    model_name: str = Field(..., description="Model name or path")
    device: str = Field("auto", description="Device to run model on")
    
    # Generation parameters
    max_length: int = Field(512, description="Maximum sequence length")
    temperature: float = Field(0.1, description="Generation temperature")
    top_p: float = Field(0.9, description="Top-p sampling parameter")
    do_sample: bool = Field(False, description="Whether to use sampling")
    
    # Function calling
    enable_function_calling: bool = Field(True, description="Enable function calling capabilities")
    max_function_calls: int = Field(5, description="Maximum function calls per instruction")
    
    # Context management
    context_window: int = Field(4096, description="Context window size")
    memory_size: int = Field(10, description="Number of previous interactions to remember")


class TrainingConfig(BaseModel):
    """Configuration for training."""
    
    # Data
    data_path: str = Field(..., description="Path to training data")
    validation_split: float = Field(0.1, description="Validation split ratio")
    
    # Training parameters
    batch_size: int = Field(8, description="Training batch size")
    learning_rate: float = Field(1e-4, description="Learning rate")
    num_epochs: int = Field(10, description="Number of training epochs")
    weight_decay: float = Field(0.01, description="Weight decay")
    
    # Optimization
    optimizer: str = Field("adamw", description="Optimizer type")
    scheduler: str = Field("cosine", description="Learning rate scheduler")
    warmup_steps: int = Field(1000, description="Warmup steps")
    
    # Checkpointing
    save_interval: int = Field(1000, description="Save checkpoint every N steps")
    max_checkpoints: int = Field(5, description="Maximum checkpoints to keep")
    
    # Mixed precision
    use_amp: bool = Field(True, description="Use automatic mixed precision")
    gradient_clip: float = Field(1.0, description="Gradient clipping norm")


class RobotVLAConfig(BaseModel):
    """Main configuration for RobotVLA system."""
    
    # Component configurations
    vision: VisionModelConfig
    language: LanguageModelConfig
    robots: List[RobotConfig] = Field(default_factory=list, description="List of robot configurations")
    
    # System settings
    log_level: str = Field("INFO", description="Logging level")
    log_file: Optional[str] = Field(None, description="Log file path")
    
    # Performance settings
    num_workers: int = Field(4, description="Number of worker threads")
    enable_caching: bool = Field(True, description="Enable feature caching")
    cache_size: int = Field(1000, description="Cache size limit")
    
    # Training (optional)
    training: Optional[TrainingConfig] = Field(None, description="Training configuration")
    
    # Advanced settings
    debug_mode: bool = Field(False, description="Enable debug mode")
    profiling: bool = Field(False, description="Enable performance profiling")
    
    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> "RobotVLAConfig":
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
    
    def to_yaml(self, output_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        with open(output_path, 'w') as f:
            yaml.dump(self.dict(), f, default_flow_style=False, indent=2)
    
    def get_robot_config(self, robot_id: str) -> Optional[RobotConfig]:
        """Get configuration for a specific robot."""
        for robot_config in self.robots:
            if robot_config.robot_id == robot_id:
                return robot_config
        return None 