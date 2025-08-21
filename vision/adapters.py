"""
Vision model adapters for integrating any vision model into RobotVLA.

This module provides standardized adapters for different types of vision models.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Union, Any, Dict
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

from ..core.types import VisionFeatures, VisionModelInterface
from ..core.config import VisionModelConfig
from .utils import preprocess_image, normalize_features


class VisionModelAdapter(VisionModelInterface):
    """Base adapter class for vision models."""
    
    def __init__(self, config: VisionModelConfig):
        self.config = config
        self.device = self._resolve_device(config.device)
        self.model = None
        self.preprocessor = None
        self._feature_dim = config.feature_dim
        
    def _resolve_device(self, device: str) -> torch.device:
        """Resolve device string to torch.device."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    @abstractmethod
    def _load_model(self) -> None:
        """Load the actual vision model."""
        pass
    
    @abstractmethod 
    def _extract_features(self, processed_images: torch.Tensor) -> torch.Tensor:
        """Extract features from preprocessed images."""
        pass
    
    def encode(self, images: List[Image.Image]) -> VisionFeatures:
        """Extract features from images using the vision model."""
        if self.model is None:
            self._load_model()
            
        # Preprocess images
        processed_images = []
        for img in images:
            processed_img = preprocess_image(
                img, 
                size=self.config.image_size,
                normalize=self.config.normalize
            )
            processed_images.append(processed_img)
        
        # Stack into batch
        batch_images = torch.stack(processed_images).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self._extract_features(batch_images)
            
        # Normalize if requested
        if self.config.normalize:
            features = normalize_features(features)
            
        return VisionFeatures(
            features=features,
            metadata={
                "model_type": self.config.model_type,
                "model_name": self.config.model_name,
                "batch_size": len(images),
                "image_size": self.config.image_size,
            }
        )
    
    def get_feature_dim(self) -> int:
        """Get feature dimension."""
        if self._feature_dim is not None:
            return self._feature_dim
            
        # Try to infer from model if not specified
        if self.model is None:
            self._load_model()
            
        # Create dummy input to get feature dimension
        dummy_input = torch.randn(1, 3, self.config.image_size, self.config.image_size).to(self.device)
        with torch.no_grad():
            dummy_features = self._extract_features(dummy_input)
            self._feature_dim = dummy_features.shape[-1]
            
        return self._feature_dim
    
    @property
    def model_name(self) -> str:
        return self.config.model_name


class CLIPAdapter(VisionModelAdapter):
    """Adapter for CLIP vision models."""
    
    def _load_model(self) -> None:
        """Load CLIP model."""
        try:
            import clip
            self.model, self.preprocessor = clip.load(self.config.model_name, device=self.device)
            self.model.eval()
        except ImportError:
            raise ImportError("Please install clip-by-openai: pip install clip-by-openai")
    
    def _extract_features(self, processed_images: torch.Tensor) -> torch.Tensor:
        """Extract CLIP vision features."""
        return self.model.encode_image(processed_images)


class DINOv2Adapter(VisionModelAdapter):
    """Adapter for DINOv2 vision models."""
    
    def _load_model(self) -> None:
        """Load DINOv2 model."""
        try:
            import timm
            self.model = timm.create_model(
                self.config.model_name,
                pretrained=self.config.pretrained,
                num_classes=0,  # Remove classification head
            ).to(self.device)
            self.model.eval()
        except ImportError:
            raise ImportError("Please install timm: pip install timm")
    
    def _extract_features(self, processed_images: torch.Tensor) -> torch.Tensor:
        """Extract DINOv2 features."""
        return self.model(processed_images)


class HuggingFaceAdapter(VisionModelAdapter):
    """Adapter for HuggingFace vision models."""
    
    def _load_model(self) -> None:
        """Load HuggingFace vision model."""
        try:
            from transformers import AutoModel, AutoImageProcessor
            
            self.preprocessor = AutoImageProcessor.from_pretrained(self.config.model_name)
            self.model = AutoModel.from_pretrained(self.config.model_name).to(self.device)
            self.model.eval()
        except ImportError:
            raise ImportError("Please install transformers: pip install transformers")
    
    def _extract_features(self, processed_images: torch.Tensor) -> torch.Tensor:
        """Extract features using HuggingFace model."""
        outputs = self.model(pixel_values=processed_images)
        # Try different output formats
        if hasattr(outputs, 'last_hidden_state'):
            return outputs.last_hidden_state.mean(dim=1)  # Pool sequence dimension
        elif hasattr(outputs, 'pooler_output'):
            return outputs.pooler_output
        else:
            # Fallback to first output
            return outputs[0].mean(dim=1) if len(outputs[0].shape) > 2 else outputs[0]


class CustomVisionAdapter(VisionModelAdapter):
    """Adapter for custom vision models."""
    
    def __init__(self, config: VisionModelConfig, model: nn.Module, preprocessor: Optional[Any] = None):
        super().__init__(config)
        self.custom_model = model
        self.custom_preprocessor = preprocessor
        
    def _load_model(self) -> None:
        """Load custom model."""
        self.model = self.custom_model.to(self.device)
        self.model.eval()
        if self.custom_preprocessor is not None:
            self.preprocessor = self.custom_preprocessor
    
    def _extract_features(self, processed_images: torch.Tensor) -> torch.Tensor:
        """Extract features using custom model."""
        outputs = self.model(processed_images)
        
        # Handle different output formats
        if isinstance(outputs, torch.Tensor):
            return outputs
        elif isinstance(outputs, (list, tuple)):
            return outputs[0]  # Take first output
        elif isinstance(outputs, dict):
            # Try common keys
            for key in ['features', 'last_hidden_state', 'pooler_output', 'logits']:
                if key in outputs:
                    features = outputs[key]
                    # Pool if needed
                    if len(features.shape) > 2:
                        features = features.mean(dim=1)
                    return features
            # Fallback to first value
            return list(outputs.values())[0]
        else:
            raise ValueError(f"Unsupported output format: {type(outputs)}")


def create_vision_adapter(config: VisionModelConfig, custom_model: Optional[nn.Module] = None) -> VisionModelAdapter:
    """Factory function to create appropriate vision adapter."""
    
    if custom_model is not None:
        return CustomVisionAdapter(config, custom_model)
    
    model_type = config.model_type.lower()
    
    if model_type == "clip":
        return CLIPAdapter(config)
    elif model_type == "dinov2":
        return DINOv2Adapter(config)
    elif model_type == "huggingface":
        return HuggingFaceAdapter(config)
    elif model_type == "custom":
        raise ValueError("Custom model type requires passing custom_model parameter")
    else:
        raise ValueError(f"Unsupported model type: {model_type}") 