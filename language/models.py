"""
Language model implementations for RobotVLA.

This module provides different language model backends for instruction processing.
"""

import json
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import torch
from loguru import logger

from ..core.types import (
    LanguageModelInterface,
    TaskContext,
    VisionFeatures,
    FunctionCall,
    ExecutionResult,
)
from ..core.config import LanguageModelConfig


class BaseLanguageModel(LanguageModelInterface):
    """Base class for language models."""
    
    def __init__(self, config: LanguageModelConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.device = self._resolve_device(config.device)
        
    def _resolve_device(self, device: str) -> torch.device:
        """Resolve device string to torch.device."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    @abstractmethod
    def _load_model(self) -> None:
        """Load the language model."""
        pass
    
    def _build_system_prompt(self, available_functions: Dict[str, Dict[str, Any]]) -> str:
        """Build system prompt with available robot functions."""
        prompt = """You are a robot control assistant. Your task is to interpret natural language instructions and generate appropriate robot function calls.

Available robot functions:
"""
        for func_name, func_schema in available_functions.items():
            prompt += f"\n- {func_name}: {func_schema.get('description', 'No description')}"
            if 'parameters' in func_schema:
                prompt += f"\n  Parameters: {func_schema['parameters']}"
        
        prompt += """

Instructions:
1. Analyze the user's instruction and the provided image(s)
2. Generate appropriate function calls to accomplish the task
3. Return your response as a JSON list of function calls
4. Each function call should have: function_name, parameters, priority (optional)

Example response format:
[
    {
        "function_name": "move_to_position",
        "parameters": {"x": 0.5, "y": 0.2, "z": 0.3},
        "priority": 1
    }
]
"""
        return prompt
    
    def _extract_function_calls(self, response_text: str) -> List[FunctionCall]:
        """Extract function calls from model response."""
        try:
            # Try to find JSON in the response
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            
            if start_idx == -1 or end_idx == 0:
                logger.warning("No JSON array found in response")
                return []
            
            json_str = response_text[start_idx:end_idx]
            function_data = json.loads(json_str)
            
            function_calls = []
            for func_data in function_data:
                if 'function_name' in func_data and 'parameters' in func_data:
                    function_calls.append(FunctionCall(
                        function_name=func_data['function_name'],
                        parameters=func_data['parameters'],
                        priority=func_data.get('priority', 0),
                        timeout=func_data.get('timeout', 30.0),
                    ))
            
            return function_calls
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse function calls from response: {e}")
            logger.debug(f"Response text: {response_text}")
            return []
    
    def update_context(self, context: TaskContext, result: ExecutionResult) -> None:
        """Update context with execution results."""
        context.history.append({
            "type": "execution_result",
            "status": result.status.value,
            "result": result.result,
            "error": result.error,
            "execution_time": result.execution_time,
        })


class HuggingFaceLanguageModel(BaseLanguageModel):
    """HuggingFace language model implementation."""
    
    def _load_model(self) -> None:
        """Load HuggingFace model."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto" if self.config.device == "auto" else None,
            )
            
            if self.config.device != "auto":
                self.model = self.model.to(self.device)
            
            self.model.eval()
            
            # Set pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            logger.info(f"Loaded HuggingFace model: {self.config.model_name}")
            
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers")
    
    def process_instruction(
        self,
        instruction: str,
        context: TaskContext,
        vision_features: VisionFeatures,
    ) -> List[FunctionCall]:
        """Process instruction using HuggingFace model."""
        
        if self.model is None:
            self._load_model()
        
        # For now, we'll use a simple approach without vision features
        # In a full implementation, we'd integrate vision features into the prompt
        
        # Get available functions (this would come from robot registry)
        available_functions = {
            "move_to_position": {
                "description": "Move robot end-effector to specified position",
                "parameters": {"x": "float", "y": "float", "z": "float"}
            },
            "grasp_object": {
                "description": "Grasp an object",
                "parameters": {"force": "float (0.0-1.0)"}
            },
            "release_object": {
                "description": "Release grasped object",
                "parameters": {}
            },
        }
        
        # Build prompt
        system_prompt = self._build_system_prompt(available_functions)
        user_prompt = f"Instruction: {instruction}\n\nGenerate the appropriate function calls:"
        
        full_prompt = f"{system_prompt}\n\nUser: {user_prompt}\nAssistant:"
        
        # Tokenize and generate
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_length,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                do_sample=self.config.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Extract function calls
        function_calls = self._extract_function_calls(response)
        
        logger.info(f"Generated {len(function_calls)} function calls from instruction")
        return function_calls


class OpenAILanguageModel(BaseLanguageModel):
    """OpenAI API language model implementation."""
    
    def __init__(self, config: LanguageModelConfig, api_key: str):
        super().__init__(config)
        self.api_key = api_key
        self.client = None
        
    def _load_model(self) -> None:
        """Initialize OpenAI client."""
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
            logger.info(f"Initialized OpenAI client with model: {self.config.model_name}")
        except ImportError:
            raise ImportError("Please install openai: pip install openai")
    
    def process_instruction(
        self,
        instruction: str,
        context: TaskContext,
        vision_features: VisionFeatures,
    ) -> List[FunctionCall]:
        """Process instruction using OpenAI API."""
        
        if self.client is None:
            self._load_model()
        
        # Get available functions
        available_functions = {
            "move_to_position": {
                "description": "Move robot end-effector to specified position",
                "parameters": {"x": "float", "y": "float", "z": "float"}
            },
            "grasp_object": {
                "description": "Grasp an object",
                "parameters": {"force": "float (0.0-1.0)"}
            },
            "release_object": {
                "description": "Release grasped object", 
                "parameters": {}
            },
        }
        
        # Build messages
        system_prompt = self._build_system_prompt(available_functions)
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Instruction: {instruction}\n\nGenerate the appropriate function calls:"}
        ]
        
        # Add context history if available
        if context.history:
            context_str = "Previous actions:\n"
            for item in context.history[-3:]:  # Last 3 actions
                if item.get("type") == "execution_result":
                    context_str += f"- Status: {item.get('status')}\n"
            messages.insert(-1, {"role": "user", "content": context_str})
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                max_tokens=self.config.max_length,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )
            
            response_text = response.choices[0].message.content
            function_calls = self._extract_function_calls(response_text)
            
            logger.info(f"Generated {len(function_calls)} function calls from instruction")
            return function_calls
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return []


class SimpleRuleBasedModel(BaseLanguageModel):
    """Simple rule-based model for testing purposes."""
    
    def _load_model(self) -> None:
        """No model to load for rule-based approach."""
        logger.info("Initialized rule-based language model")
    
    def process_instruction(
        self,
        instruction: str,
        context: TaskContext,
        vision_features: VisionFeatures,
    ) -> List[FunctionCall]:
        """Process instruction using simple rules."""
        
        instruction_lower = instruction.lower()
        function_calls = []
        
        # Simple keyword-based rules
        if "move" in instruction_lower or "go to" in instruction_lower:
            # Default move command
            function_calls.append(FunctionCall(
                function_name="move_to_position",
                parameters={"x": 0.5, "y": 0.2, "z": 0.3},
                priority=1
            ))
        
        if "grasp" in instruction_lower or "grab" in instruction_lower or "pick" in instruction_lower:
            function_calls.append(FunctionCall(
                function_name="grasp_object",
                parameters={"force": 0.5},
                priority=2
            ))
        
        if "release" in instruction_lower or "drop" in instruction_lower or "let go" in instruction_lower:
            function_calls.append(FunctionCall(
                function_name="release_object",
                parameters={},
                priority=1
            ))
        
        if not function_calls:
            # Default fallback
            function_calls.append(FunctionCall(
                function_name="move_to_position",
                parameters={"x": 0.0, "y": 0.0, "z": 0.2},
                priority=1
            ))
        
        logger.info(f"Generated {len(function_calls)} function calls using rule-based model")
        return function_calls 