"""
Utility functions for vision processing in RobotVLA.

This module provides common image preprocessing and feature manipulation utilities.
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision import transforms


def preprocess_image(
    image: Image.Image,
    size: Union[int, Tuple[int, int]] = 224,
    normalize: bool = True,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> torch.Tensor:
    """
    Preprocess a PIL image for vision model input.
    
    Args:
        image: Input PIL image
        size: Target size (int or (height, width))
        normalize: Whether to normalize with ImageNet stats
        mean: Normalization mean values
        std: Normalization std values
        
    Returns:
        Preprocessed image tensor [C, H, W]
    """
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Create transform pipeline
    transform_list = []
    
    # Resize
    if isinstance(size, int):
        transform_list.append(transforms.Resize((size, size)))
    else:
        transform_list.append(transforms.Resize(size))
    
    # Convert to tensor
    transform_list.append(transforms.ToTensor())
    
    # Normalize
    if normalize:
        transform_list.append(transforms.Normalize(mean=mean, std=std))
    
    transform = transforms.Compose(transform_list)
    return transform(image)


def resize_image(
    image: Image.Image,
    size: Union[int, Tuple[int, int]],
    mode: str = "bilinear"
) -> Image.Image:
    """
    Resize image while maintaining aspect ratio.
    
    Args:
        image: Input PIL image
        size: Target size
        mode: Resize mode
        
    Returns:
        Resized PIL image
    """
    if isinstance(size, int):
        size = (size, size)
    
    return image.resize(size, getattr(Image, mode.upper(), Image.BILINEAR))


def normalize_features(features: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Normalize feature vectors.
    
    Args:
        features: Input feature tensor
        dim: Dimension to normalize along
        
    Returns:
        Normalized feature tensor
    """
    return F.normalize(features, p=2, dim=dim)


def batch_images(images: list[Image.Image], size: Union[int, Tuple[int, int]] = 224) -> torch.Tensor:
    """
    Batch process multiple images.
    
    Args:
        images: List of PIL images
        size: Target size for all images
        
    Returns:
        Batched tensor [B, C, H, W]
    """
    processed_images = []
    for img in images:
        processed_img = preprocess_image(img, size=size)
        processed_images.append(processed_img)
    
    return torch.stack(processed_images)


def extract_spatial_features(
    features: torch.Tensor,
    spatial_size: Optional[Tuple[int, int]] = None
) -> torch.Tensor:
    """
    Extract spatial features from vision features.
    
    Args:
        features: Input features [B, D] or [B, H*W, D]
        spatial_size: Target spatial dimensions (H, W)
        
    Returns:
        Spatial features [B, D, H, W]
    """
    if len(features.shape) == 2:
        # Global features, can't extract spatial
        return None
    elif len(features.shape) == 3:
        # Sequence features [B, seq_len, D]
        B, seq_len, D = features.shape
        
        if spatial_size is None:
            # Try to infer square spatial size
            H = W = int(seq_len ** 0.5)
            if H * W != seq_len:
                return None  # Can't reshape to spatial
        else:
            H, W = spatial_size
            
        # Reshape to spatial format
        features = features.view(B, H, W, D).permute(0, 3, 1, 2)
        return features
    else:
        # Already spatial
        return features


def compute_feature_similarity(
    features1: torch.Tensor,
    features2: torch.Tensor,
    method: str = "cosine"
) -> torch.Tensor:
    """
    Compute similarity between feature vectors.
    
    Args:
        features1: First set of features
        features2: Second set of features  
        method: Similarity method ('cosine', 'l2', 'dot')
        
    Returns:
        Similarity scores
    """
    if method == "cosine":
        return F.cosine_similarity(features1, features2, dim=-1)
    elif method == "l2":
        return -torch.norm(features1 - features2, p=2, dim=-1)
    elif method == "dot":
        return torch.sum(features1 * features2, dim=-1)
    else:
        raise ValueError(f"Unsupported similarity method: {method}")


def create_attention_mask(
    batch_size: int,
    sequence_length: int,
    padding_lengths: Optional[list[int]] = None
) -> torch.Tensor:
    """
    Create attention mask for batched sequences.
    
    Args:
        batch_size: Batch size
        sequence_length: Maximum sequence length
        padding_lengths: List of padding lengths for each sequence
        
    Returns:
        Attention mask [B, seq_len]
    """
    if padding_lengths is None:
        # No padding, all ones
        return torch.ones(batch_size, sequence_length, dtype=torch.bool)
    
    mask = torch.zeros(batch_size, sequence_length, dtype=torch.bool)
    for i, pad_len in enumerate(padding_lengths):
        mask[i, :sequence_length - pad_len] = True
    
    return mask 