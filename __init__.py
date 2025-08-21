"""
RobotVLA: Universal Vision-Language-Action Framework

A next-generation robot control framework that enables any visual model 
to control any robot through a universal function-call interface.

Key Features:
- Universal Vision Model Interface: Plug in any vision model
- Function-Call Robot Control: Robot actions as composable function calls
- Vendor-Agnostic Design: Robot manufacturers can integrate their methods as plugins
- Dynamic Robot Registration: Support for unknown robot morphologies without retraining

Example Usage:
```python
from robotvla import RobotVLAPipeline, RobotVLAConfig
from robotvla.vision import CLIPAdapter
from robotvla.language import SimpleRuleBasedModel

# Create configuration
config = RobotVLAConfig.from_yaml("config.yaml")

# Initialize pipeline
pipeline = RobotVLAPipeline(config)
pipeline.initialize_vision_model()
pipeline.initialize_language_model(SimpleRuleBasedModel(config.language))

# Process instruction
result = pipeline.process_instruction(
    instruction="Pick up the red cube",
    images=[image],
)
```
"""

from .core import (
    RobotVLAConfig,
    VisionModelConfig, 
    RobotConfig,
    LanguageModelConfig,
    RobotVLAPipeline,
)

from .core.types import (
    VisionFeatures,
    RobotAction,
    FunctionCall,
    ExecutionResult,
    TaskContext,
    ActionType,
    ExecutionStatus,
)

__version__ = "0.1.0"
__author__ = "RobotVLA Team"
__email__ = "contact@robotvla.org"
__description__ = "Universal Vision-Language-Action Framework for Robot Control"

__all__ = [
    # Main Components
    "RobotVLAPipeline",
    
    # Configuration
    "RobotVLAConfig",
    "VisionModelConfig",
    "RobotConfig", 
    "LanguageModelConfig",
    
    # Core Types
    "VisionFeatures",
    "RobotAction",
    "FunctionCall", 
    "ExecutionResult",
    "TaskContext",
    "ActionType",
    "ExecutionStatus",
    
    # Metadata
    "__version__",
    "__author__",
    "__email__",
    "__description__",
] 