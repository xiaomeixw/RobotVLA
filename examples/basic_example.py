#!/usr/bin/env python3
"""
Basic RobotVLA Usage Example

This example demonstrates how to:
1. Initialize RobotVLA with any vision model
2. Register a robot interface
3. Process natural language instructions
4. Execute robot function calls

Run with: python examples/basic_example.py
"""

import asyncio
import sys
from pathlib import Path
from PIL import Image
import numpy as np

# Add robotvla to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from robotvla.core import RobotVLAConfig, VisionModelConfig, RobotConfig, LanguageModelConfig
from robotvla.core.pipeline import RobotVLAPipeline
from robotvla.core.types import (
    RobotInterface, 
    FunctionCall, 
    ExecutionResult, 
    ExecutionStatus
)
from robotvla.language.models import SimpleRuleBasedModel


class MockRobotInterface(RobotInterface):
    """Mock robot interface for demonstration purposes."""
    
    def __init__(self, robot_id: str = "mock_robot"):
        self._robot_id = robot_id
        self._position = [0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0]  # x, y, z, rx, ry, rz, gripper
    
    @property
    def robot_id(self) -> str:
        return self._robot_id
    
    def execute_function(self, function_call: FunctionCall) -> ExecutionResult:
        """Execute a robot function call."""
        import time
        start_time = time.time()
        
        try:
            func_name = function_call.function_name
            params = function_call.parameters
            
            if func_name == "move_to_position":
                # Simulate movement
                x = params.get("x", 0.0)
                y = params.get("y", 0.0) 
                z = params.get("z", 0.2)
                
                print(f"🤖 Moving to position: ({x:.3f}, {y:.3f}, {z:.3f})")
                
                # Update internal position
                self._position[0] = x
                self._position[1] = y
                self._position[2] = z
                
                # Simulate execution time
                time.sleep(0.5)
                
                return ExecutionResult(
                    status=ExecutionStatus.SUCCESS,
                    result={"new_position": self._position[:3]},
                    execution_time=time.time() - start_time,
                )
                
            elif func_name == "grasp_object":
                force = params.get("force", 0.5)
                print(f"🤖 Grasping object with force: {force}")
                
                self._position[6] = force  # Set gripper position
                time.sleep(0.3)
                
                return ExecutionResult(
                    status=ExecutionStatus.SUCCESS,
                    result={"gripper_force": force},
                    execution_time=time.time() - start_time,
                )
                
            elif func_name == "release_object":
                print("🤖 Releasing object")
                
                self._position[6] = 0.0  # Open gripper
                time.sleep(0.2)
                
                return ExecutionResult(
                    status=ExecutionStatus.SUCCESS,
                    result={"gripper_open": True},
                    execution_time=time.time() - start_time,
                )
                
            else:
                return ExecutionResult(
                    status=ExecutionStatus.FAILED,
                    error=f"Unknown function: {func_name}",
                    execution_time=time.time() - start_time,
                )
                
        except Exception as e:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                error=str(e),
                execution_time=time.time() - start_time,
            )
    
    def get_available_functions(self) -> list[str]:
        """Get list of available function names."""
        return ["move_to_position", "grasp_object", "release_object"]
    
    def get_function_schema(self, function_name: str) -> dict:
        """Get the schema/signature of a function."""
        schemas = {
            "move_to_position": {
                "description": "Move robot end-effector to specified position",
                "parameters": {
                    "x": {"type": "float", "description": "X coordinate"},
                    "y": {"type": "float", "description": "Y coordinate"},
                    "z": {"type": "float", "description": "Z coordinate"},
                },
                "required": ["x", "y", "z"],
            },
            "grasp_object": {
                "description": "Grasp an object with specified force",
                "parameters": {
                    "force": {"type": "float", "description": "Grasp force (0.0-1.0)", "min": 0.0, "max": 1.0},
                },
                "required": ["force"],
            },
            "release_object": {
                "description": "Release grasped object",
                "parameters": {},
                "required": [],
            },
        }
        return schemas.get(function_name, {})


def create_sample_image() -> Image.Image:
    """Create a sample image for testing."""
    # Create a simple colored image
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(img_array)


def create_config() -> RobotVLAConfig:
    """Create a sample RobotVLA configuration."""
    
    # Vision model configuration - using CLIP
    vision_config = VisionModelConfig(
        model_type="clip",
        model_name="ViT-B/32",
        pretrained=True,
        device="auto",
        image_size=224,
    )
    
    # Language model configuration - using simple rule-based model for demo
    language_config = LanguageModelConfig(
        model_type="rule_based",
        model_name="simple_rules",
        device="cpu",
    )
    
    # Robot configuration
    robot_config = RobotConfig(
        robot_id="mock_robot",
        robot_type="mock",
        connection_type="local",
        dof=7,
        has_gripper=True,
    )
    
    # Main configuration
    config = RobotVLAConfig(
        vision=vision_config,
        language=language_config,
        robots=[robot_config],
        log_level="INFO",
        debug_mode=True,
    )
    
    return config


async def main():
    """Main example function."""
    print("🚀 RobotVLA Basic Example")
    print("=" * 50)
    
    # 1. Create configuration
    print("📝 Creating configuration...")
    config = create_config()
    
    # 2. Initialize pipeline
    print("🔧 Initializing RobotVLA pipeline...")
    pipeline = RobotVLAPipeline(config)
    
    # 3. Initialize vision model
    print("👁️ Initializing vision model...")
    try:
        pipeline.initialize_vision_model()
        print("✅ Vision model initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize vision model: {e}")
        print("💡 Note: Install required dependencies with: pip install clip-by-openai")
        return
    
    # 4. Initialize language model
    print("🧠 Initializing language model...")
    language_model = SimpleRuleBasedModel(config.language)
    pipeline.initialize_language_model(language_model)
    print("✅ Language model initialized")
    
    # 5. Register robot
    print("🤖 Registering robot...")
    robot = MockRobotInterface("demo_robot")
    pipeline.register_robot(robot)
    print("✅ Robot registered")
    
    # 6. Create sample image
    print("🖼️ Creating sample image...")
    sample_image = create_sample_image()
    
    # 7. Process instructions
    print("\n🎯 Processing instructions...")
    print("-" * 30)
    
    instructions = [
        "Move to position (0.3, 0.1, 0.25)",
        "Grasp the object gently",
        "Move to a new location",
        "Release the object",
    ]
    
    for i, instruction in enumerate(instructions, 1):
        print(f"\n📝 Instruction {i}: {instruction}")
        
        try:
            result = pipeline.process_instruction(
                instruction=instruction,
                images=[sample_image],
                robot_id="demo_robot",
            )
            
            if result["success"]:
                print(f"✅ Success! Processing time: {result['processing_time']:.3f}s")
                print(f"📋 Function calls: {len(result['function_calls'])}")
                for fc in result['function_calls']:
                    print(f"   - {fc['function_name']}: {fc['parameters']}")
            else:
                print(f"❌ Failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Error processing instruction: {e}")
    
    # 8. Show system status
    print("\n📊 System Status:")
    print("-" * 20)
    status = pipeline.get_system_status()
    
    print(f"Vision Model: {status['vision_model']['model_name']}")
    print(f"Language Model: {status['language_model']['model_name']}")
    print(f"Robots: {status['robots']['count']}")
    print(f"Active Contexts: {status['contexts']['active_count']}")
    
    # 9. Cleanup
    print("\n🧹 Cleaning up...")
    pipeline.shutdown()
    print("✅ Shutdown complete")
    
    print("\n🎉 Example completed successfully!")


if __name__ == "__main__":
    asyncio.run(main()) 