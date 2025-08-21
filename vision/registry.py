"""
Vision model registry for dynamic model management.

This module provides a registry system for registering and creating vision model adapters.
"""

from typing import Dict, Type, Optional, Any, List
import torch.nn as nn
from loguru import logger

from ..core.config import VisionModelConfig
from .adapters import VisionModelAdapter, create_vision_adapter


class VisionModelRegistry:
    """Registry for managing vision model adapters."""
    
    def __init__(self):
        self._adapters: Dict[str, Type[VisionModelAdapter]] = {}
        self._instances: Dict[str, VisionModelAdapter] = {}
        self._configs: Dict[str, VisionModelConfig] = {}
        
        # Register default adapters
        self._register_default_adapters()
    
    def _register_default_adapters(self) -> None:
        """Register default vision model adapters."""
        from .adapters import CLIPAdapter, DINOv2Adapter, HuggingFaceAdapter, CustomVisionAdapter
        
        self.register_adapter("clip", CLIPAdapter)
        self.register_adapter("dinov2", DINOv2Adapter)
        self.register_adapter("huggingface", HuggingFaceAdapter)
        self.register_adapter("custom", CustomVisionAdapter)
        
        logger.info("Registered default vision model adapters")
    
    def register_adapter(self, model_type: str, adapter_class: Type[VisionModelAdapter]) -> None:
        """
        Register a new vision model adapter.
        
        Args:
            model_type: Type identifier for the model
            adapter_class: Adapter class to register
        """
        self._adapters[model_type.lower()] = adapter_class
        logger.info(f"Registered vision adapter: {model_type}")
    
    def create_adapter(
        self, 
        config: VisionModelConfig,
        custom_model: Optional[nn.Module] = None,
        force_reload: bool = False
    ) -> VisionModelAdapter:
        """
        Create or retrieve a vision model adapter.
        
        Args:
            config: Vision model configuration
            custom_model: Optional custom model instance
            force_reload: Whether to force reload even if cached
            
        Returns:
            Vision model adapter instance
        """
        # Create unique key for caching
        cache_key = f"{config.model_type}:{config.model_name}"
        
        # Return cached instance if available
        if not force_reload and cache_key in self._instances:
            logger.debug(f"Using cached vision adapter: {cache_key}")
            return self._instances[cache_key]
        
        # Create new adapter
        model_type = config.model_type.lower()
        
        if model_type not in self._adapters:
            raise ValueError(f"Unknown model type: {model_type}. Available types: {list(self._adapters.keys())}")
        
        if custom_model is not None:
            adapter = CustomVisionAdapter(config, custom_model)
        else:
            adapter_class = self._adapters[model_type]
            adapter = adapter_class(config)
        
        # Cache the adapter and config
        self._instances[cache_key] = adapter
        self._configs[cache_key] = config
        
        logger.info(f"Created vision adapter: {cache_key}")
        return adapter
    
    def get_adapter(self, model_type: str, model_name: str) -> Optional[VisionModelAdapter]:
        """
        Get cached adapter by type and name.
        
        Args:
            model_type: Model type
            model_name: Model name
            
        Returns:
            Cached adapter or None
        """
        cache_key = f"{model_type}:{model_name}"
        return self._instances.get(cache_key)
    
    def list_available_types(self) -> List[str]:
        """Get list of available model types."""
        return list(self._adapters.keys())
    
    def list_loaded_models(self) -> List[str]:
        """Get list of currently loaded models."""
        return list(self._instances.keys())
    
    def clear_cache(self) -> None:
        """Clear all cached adapters."""
        self._instances.clear()
        self._configs.clear()
        logger.info("Cleared vision model cache")
    
    def remove_adapter(self, model_type: str, model_name: str) -> bool:
        """
        Remove specific adapter from cache.
        
        Args:
            model_type: Model type
            model_name: Model name
            
        Returns:
            True if removed, False if not found
        """
        cache_key = f"{model_type}:{model_name}"
        if cache_key in self._instances:
            del self._instances[cache_key]
            del self._configs[cache_key]
            logger.info(f"Removed vision adapter: {cache_key}")
            return True
        return False
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics for loaded models."""
        stats = {
            "num_loaded_models": len(self._instances),
            "model_details": {},
        }
        
        for cache_key, adapter in self._instances.items():
            if adapter.model is not None:
                # Try to get model memory usage
                try:
                    num_params = sum(p.numel() for p in adapter.model.parameters())
                    stats["model_details"][cache_key] = {
                        "num_parameters": num_params,
                        "device": str(adapter.device),
                        "feature_dim": adapter.get_feature_dim(),
                    }
                except Exception as e:
                    stats["model_details"][cache_key] = {"error": str(e)}
        
        return stats


# Global registry instance
vision_registry = VisionModelRegistry() 