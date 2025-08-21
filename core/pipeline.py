"""
Main pipeline for RobotVLA system.

This module orchestrates the interaction between vision models, language models, and robot interfaces.
"""

from typing import List, Optional, Dict, Any
import time
import uuid
from loguru import logger

from .config import RobotVLAConfig
from .types import (
    VisionFeatures, 
    TaskContext, 
    FunctionCall, 
    ExecutionResult,
    ExecutionStatus,
    VisionModelInterface,
    LanguageModelInterface,
    RobotInterface,
)
from ..vision.registry import vision_registry


class RobotVLAPipeline:
    """Main pipeline for RobotVLA system."""
    
    def __init__(self, config: RobotVLAConfig):
        self.config = config
        
        # Initialize components
        self.vision_adapter: Optional[VisionModelInterface] = None
        self.language_model: Optional[LanguageModelInterface] = None
        self.robots: Dict[str, RobotInterface] = {}
        
        # State management
        self.active_contexts: Dict[str, TaskContext] = {}
        
        # Initialize logging
        self._setup_logging()
        
        logger.info("Initialized RobotVLA Pipeline")
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        logger.remove()  # Remove default handler
        logger.add(
            lambda msg: print(msg, end=""),
            level=self.config.log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        )
        
        if self.config.log_file:
            logger.add(
                self.config.log_file,
                level=self.config.log_level,
                rotation="10 MB",
                retention="1 week"
            )
    
    def initialize_vision_model(self, custom_model: Optional[Any] = None) -> None:
        """Initialize the vision model adapter."""
        try:
            self.vision_adapter = vision_registry.create_adapter(
                self.config.vision,
                custom_model=custom_model
            )
            logger.info(f"Initialized vision model: {self.config.vision.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize vision model: {e}")
            raise
    
    def initialize_language_model(self, language_model: LanguageModelInterface) -> None:
        """Initialize the language model."""
        self.language_model = language_model
        logger.info("Initialized language model")
    
    def register_robot(self, robot: RobotInterface) -> None:
        """Register a robot interface."""
        self.robots[robot.robot_id] = robot
        logger.info(f"Registered robot: {robot.robot_id}")
    
    def process_instruction(
        self,
        instruction: str,
        images: List[Any],  # PIL Images or paths
        robot_id: Optional[str] = None,
        context_id: Optional[str] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a natural language instruction with images.
        
        Args:
            instruction: Natural language instruction
            images: List of PIL images or image paths
            robot_id: Target robot ID (if None, uses first available)
            context_id: Existing context ID (if None, creates new)
            constraints: Task constraints
            
        Returns:
            Processing results including actions and execution status
        """
        start_time = time.time()
        
        # Validate inputs
        if not self.vision_adapter:
            raise RuntimeError("Vision model not initialized. Call initialize_vision_model() first.")
        if not self.language_model:
            raise RuntimeError("Language model not initialized. Call initialize_language_model() first.")
        if not self.robots:
            raise RuntimeError("No robots registered. Call register_robot() first.")
        
        # Process images to PIL format
        from PIL import Image
        processed_images = []
        for img in images:
            if isinstance(img, str):
                processed_images.append(Image.open(img))
            elif isinstance(img, Image.Image):
                processed_images.append(img)
            else:
                raise ValueError(f"Unsupported image type: {type(img)}")
        
        # Create or get task context
        if context_id is None:
            context_id = str(uuid.uuid4())
        
        if context_id not in self.active_contexts:
            self.active_contexts[context_id] = TaskContext(
                task_id=context_id,
                instruction=instruction,
                images=processed_images,
                constraints=constraints or {},
            )
        else:
            # Update existing context
            context = self.active_contexts[context_id]
            context.instruction = instruction
            context.images = processed_images
            if constraints:
                context.constraints.update(constraints)
        
        context = self.active_contexts[context_id]
        
        try:
            # Step 1: Extract vision features
            logger.info(f"Extracting vision features from {len(processed_images)} images")
            vision_features = self.vision_adapter.encode(processed_images)
            
            # Step 2: Process instruction and generate function calls
            logger.info(f"Processing instruction: {instruction}")
            function_calls = self.language_model.process_instruction(
                instruction, context, vision_features
            )
            
            # Step 3: Execute function calls
            execution_results = []
            for func_call in function_calls:
                result = self._execute_function_call(func_call, robot_id)
                execution_results.append(result)
                
                # Update context with results
                self.language_model.update_context(context, result)
                context.history.append({
                    "function_call": func_call.to_dict(),
                    "result": {
                        "status": result.status.value,
                        "execution_time": result.execution_time,
                        "error": result.error,
                    },
                    "timestamp": time.time(),
                })
            
            processing_time = time.time() - start_time
            
            # Prepare response
            response = {
                "context_id": context_id,
                "instruction": instruction,
                "function_calls": [fc.to_dict() for fc in function_calls],
                "execution_results": [
                    {
                        "status": result.status.value,
                        "result": result.result,
                        "error": result.error,
                        "execution_time": result.execution_time,
                    }
                    for result in execution_results
                ],
                "processing_time": processing_time,
                "success": all(result.success for result in execution_results),
            }
            
            logger.info(f"Completed instruction processing in {processing_time:.3f}s")
            return response
            
        except Exception as e:
            logger.error(f"Error processing instruction: {e}")
            return {
                "context_id": context_id,
                "error": str(e),
                "success": False,
                "processing_time": time.time() - start_time,
            }
    
    def _execute_function_call(
        self, 
        function_call: FunctionCall, 
        preferred_robot_id: Optional[str] = None
    ) -> ExecutionResult:
        """Execute a single function call."""
        
        # Determine target robot
        target_robot_id = function_call.robot_id or preferred_robot_id
        if target_robot_id is None:
            # Use first available robot
            if not self.robots:
                return ExecutionResult(
                    status=ExecutionStatus.FAILED,
                    error="No robots available"
                )
            target_robot_id = next(iter(self.robots.keys()))
        
        if target_robot_id not in self.robots:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                error=f"Robot {target_robot_id} not found"
            )
        
        robot = self.robots[target_robot_id]
        
        # Check if function is available
        available_functions = robot.get_available_functions()
        if function_call.function_name not in available_functions:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                error=f"Function {function_call.function_name} not available on robot {target_robot_id}"
            )
        
        # Execute function call
        logger.info(f"Executing {function_call.function_name} on robot {target_robot_id}")
        try:
            result = robot.execute_function(function_call)
            logger.info(f"Function execution completed: {result.status.value}")
            return result
        except Exception as e:
            logger.error(f"Function execution failed: {e}")
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                error=str(e)
            )
    
    def get_context(self, context_id: str) -> Optional[TaskContext]:
        """Get task context by ID."""
        return self.active_contexts.get(context_id)
    
    def list_active_contexts(self) -> List[str]:
        """Get list of active context IDs."""
        return list(self.active_contexts.keys())
    
    def clear_context(self, context_id: str) -> bool:
        """Clear a specific context."""
        if context_id in self.active_contexts:
            del self.active_contexts[context_id]
            logger.info(f"Cleared context: {context_id}")
            return True
        return False
    
    def clear_all_contexts(self) -> None:
        """Clear all active contexts."""
        self.active_contexts.clear()
        logger.info("Cleared all contexts")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status information."""
        return {
            "vision_model": {
                "initialized": self.vision_adapter is not None,
                "model_name": self.config.vision.model_name if self.vision_adapter else None,
                "feature_dim": self.vision_adapter.get_feature_dim() if self.vision_adapter else None,
            },
            "language_model": {
                "initialized": self.language_model is not None,
                "model_name": self.config.language.model_name if self.language_model else None,
            },
            "robots": {
                "count": len(self.robots),
                "robot_ids": list(self.robots.keys()),
                "available_functions": {
                    robot_id: robot.get_available_functions()
                    for robot_id, robot in self.robots.items()
                },
            },
            "contexts": {
                "active_count": len(self.active_contexts),
                "context_ids": list(self.active_contexts.keys()),
            },
            "memory_usage": vision_registry.get_memory_usage(),
        }
    
    def shutdown(self) -> None:
        """Cleanup and shutdown pipeline."""
        logger.info("Shutting down RobotVLA Pipeline")
        
        # Clear all contexts
        self.clear_all_contexts()
        
        # Clear vision model cache
        vision_registry.clear_cache()
        
        # Reset components
        self.vision_adapter = None
        self.language_model = None
        self.robots.clear()
        
        logger.info("Pipeline shutdown complete") 