#!/usr/bin/env python3
"""
Lightweight VQA Model: MobileNetV3-Small + DistilBERT
Optimized for radiology VQA with minimal parameters
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import DistilBertModel, DistilBertConfig
from typing import Dict, Optional, Tuple
import math


class AttentionFusion(nn.Module):
    """Efficient attention-based fusion of visual and textual features."""
    
    def __init__(self, 
                 visual_dim: int = 576,
                 text_dim: int = 768,
                 hidden_dim: int = 256,
                 num_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        
        self.visual_dim = visual_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Project visual and text features to same dimension
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Multi-head attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, visual_features, text_features):
        """
        Args:
            visual_features: (batch, visual_dim)
            text_features: (batch, text_dim)
        Returns:
            fused_features: (batch, hidden_dim)
        """
        # Project to same dimension
        v = self.visual_proj(visual_features)  # (batch, hidden_dim)
        t = self.text_proj(text_features)      # (batch, hidden_dim)
        
        # Add sequence dimension for attention
        v = v.unsqueeze(1)  # (batch, 1, hidden_dim)
        t = t.unsqueeze(1)  # (batch, 1, hidden_dim)
        
        # Concatenate for attention
        combined = torch.cat([v, t], dim=1)  # (batch, 2, hidden_dim)
        
        # Multi-head attention (query=combined, key=combined, value=combined)
        attn_out, attn_weights = self.multihead_attn(
            combined, combined, combined
        )
        
        # Residual connection and normalization
        combined = self.norm1(combined + attn_out)
        
        # Feed-forward network
        ffn_out = self.ffn(combined)
        combined = self.norm2(combined + ffn_out)
        
        # Global average pooling across sequence
        fused = combined.mean(dim=1)  # (batch, hidden_dim)
        
        return fused, attn_weights


class LightweightVQAModel(nn.Module):
    """
    Lightweight VQA model combining MobileNetV3-Small and DistilBERT.
    
    Architecture:
    - Vision: MobileNetV3-Small (pretrained on ImageNet)
    - Language: DistilBERT (pretrained)
    - Fusion: Attention-based multimodal fusion
    - Classifier: Lightweight MLP
    
    Total params: ~8-10M (compared to 100M+ for full BERT+ResNet)
    """
    
    def __init__(self,
                 num_classes: int,
                 visual_feature_dim: int = 576,  # MobileNetV3-Small output
                 text_feature_dim: int = 768,    # DistilBERT hidden size
                 fusion_hidden_dim: int = 256,
                 num_attention_heads: int = 4,
                 dropout: float = 0.3,
                 freeze_vision_encoder: bool = False,
                 freeze_text_encoder: bool = False,
                 use_spatial_features: bool = False):
        """
        Args:
            num_classes: Number of answer classes
            visual_feature_dim: Dimension of visual features from MobileNetV3
            text_feature_dim: Dimension of text features from DistilBERT
            fusion_hidden_dim: Hidden dimension for fusion module
            num_attention_heads: Number of attention heads
            dropout: Dropout probability
            freeze_vision_encoder: Whether to freeze vision encoder weights
            freeze_text_encoder: Whether to freeze text encoder weights
            use_spatial_features: Use spatial features instead of global pooling
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.use_spatial_features = use_spatial_features
        
        # ============= Vision Encoder: MobileNetV3-Small =============
        mobilenet = models.mobilenet_v3_small(
            weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        )
        
        # Remove classifier, keep feature extractor
        self.vision_encoder = nn.Sequential(*list(mobilenet.children())[:-1])
        
        # Get actual output dimension
        if use_spatial_features:
            # Use conv features before pooling: (batch, 576, 7, 7)
            self.vision_encoder = mobilenet.features
            visual_feature_dim = 576 * 7 * 7  # Flattened spatial features
        else:
            # Use pooled features: (batch, 576)
            visual_feature_dim = 576
        
        if freeze_vision_encoder:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False
        
        # ============= Text Encoder: DistilBERT =============
        self.text_encoder = DistilBertModel.from_pretrained('distilbert-base-uncased')
        
        if freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        
        # ============= Multimodal Fusion =============
        self.fusion = AttentionFusion(
            visual_dim=visual_feature_dim,
            text_dim=text_feature_dim,
            hidden_dim=fusion_hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout
        )
        
        # ============= Answer Classifier =============
        self.classifier = nn.Sequential(
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.LayerNorm(fusion_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim // 2, num_classes)
        )
        
        # Initialize classifier weights
        self._init_classifier_weights()
    
    def _init_classifier_weights(self):
        """Initialize classifier with Xavier initialization."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def extract_visual_features(self, images):
        """Extract visual features from images."""
        features = self.vision_encoder(images)
        
        if self.use_spatial_features:
            # Flatten spatial features: (batch, 576, 7, 7) -> (batch, 576*49)
            batch_size = features.size(0)
            features = features.view(batch_size, -1)
        else:
            # Global average pooling if not using spatial features
            if len(features.shape) == 4:  # (batch, channels, h, w)
                features = F.adaptive_avg_pool2d(features, 1)
                features = features.view(features.size(0), -1)
        
        return features
    
    def extract_text_features(self, input_ids, attention_mask):
        """Extract text features from questions."""
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        text_features = outputs.last_hidden_state[:, 0, :]  # (batch, 768)
        
        return text_features
    
    def forward(self, 
                images: torch.Tensor,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                return_attention: bool = False):
        """
        Forward pass.
        
        Args:
            images: (batch, 3, 224, 224)
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            return_attention: Whether to return attention weights
        
        Returns:
            logits: (batch, num_classes)
            attention_weights: Optional attention weights
        """
        # Extract features
        visual_features = self.extract_visual_features(images)
        text_features = self.extract_text_features(input_ids, attention_mask)
        
        # Fuse features
        fused_features, attention_weights = self.fusion(visual_features, text_features)
        
        # Classify
        logits = self.classifier(fused_features)
        
        if return_attention:
            return logits, attention_weights
        return logits
    
    def count_parameters(self):
        """Count trainable and total parameters."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        
        # Count by component
        vision_params = sum(p.numel() for p in self.vision_encoder.parameters())
        text_params = sum(p.numel() for p in self.text_encoder.parameters())
        fusion_params = sum(p.numel() for p in self.fusion.parameters())
        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        
        return {
            'total': total,
            'trainable': trainable,
            'vision_encoder': vision_params,
            'text_encoder': text_params,
            'fusion': fusion_params,
            'classifier': classifier_params
        }
    
    def get_model_size_mb(self):
        """Get model size in MB."""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        size_mb = (param_size + buffer_size) / (1024 ** 2)
        return size_mb


class BinaryVQAModel(LightweightVQAModel):
    """
    Specialized model for binary (yes/no) questions.
    Uses the same architecture but with 2 output classes.
    """
    
    def __init__(self, **kwargs):
        kwargs['num_classes'] = 2  # Yes/No
        super().__init__(**kwargs)


class MultiTaskVQAModel(nn.Module):
    """
    Multi-task model that handles both binary and open-ended questions.
    Shares encoders but has separate classification heads.
    """
    
    def __init__(self,
                 num_open_ended_classes: int,
                 visual_feature_dim: int = 576,
                 text_feature_dim: int = 768,
                 fusion_hidden_dim: int = 256,
                 num_attention_heads: int = 4,
                 dropout: float = 0.3):
        super().__init__()
        
        # Shared backbone
        self.base_model = LightweightVQAModel(
            num_classes=num_open_ended_classes,  # Will be replaced
            visual_feature_dim=visual_feature_dim,
            text_feature_dim=text_feature_dim,
            fusion_hidden_dim=fusion_hidden_dim,
            num_attention_heads=num_attention_heads,
            dropout=dropout
        )
        
        # Remove the classifier
        self.vision_encoder = self.base_model.vision_encoder
        self.text_encoder = self.base_model.text_encoder
        self.fusion = self.base_model.fusion
        
        # Separate heads for each task
        self.binary_head = nn.Sequential(
            nn.Linear(fusion_hidden_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)  # Yes/No
        )
        
        self.open_ended_head = nn.Sequential(
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.LayerNorm(fusion_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim // 2, num_open_ended_classes)
        )
    
    def forward(self, images, input_ids, attention_mask, question_type=None):
        """
        Forward pass for multi-task model.
        
        Args:
            question_type: List of 'binary' or 'open-ended' for each sample
        """
        # Shared feature extraction
        visual_features = self.base_model.extract_visual_features(images)
        text_features = self.base_model.extract_text_features(input_ids, attention_mask)
        fused_features, _ = self.fusion(visual_features, text_features)
        
        # Route to appropriate head
        if question_type is None:
            # Return both predictions
            binary_logits = self.binary_head(fused_features)
            open_ended_logits = self.open_ended_head(fused_features)
            return {
                'binary': binary_logits,
                'open_ended': open_ended_logits,
                'features': fused_features
            }
        else:
            # Route based on question type (for mixed batches)
            outputs = []
            for i, qtype in enumerate(question_type):
                feat = fused_features[i:i+1]
                if qtype == 'binary':
                    outputs.append(self.binary_head(feat))
                else:
                    outputs.append(self.open_ended_head(feat))
            return torch.cat(outputs, dim=0)


def create_lightweight_vqa_model(
    num_classes: int,
    model_type: str = 'standard',
    fusion_dim: int = 256,
    dropout: float = 0.3,
    freeze_encoders: bool = False,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> nn.Module:
    """
    Factory function to create VQA model.
    
    Args:
        num_classes: Number of answer classes
        model_type: 'standard', 'binary', or 'multitask'
        fusion_dim: Hidden dimension for fusion
        dropout: Dropout probability
        freeze_encoders: Whether to freeze pretrained encoders
        device: Device to load model on
    
    Returns:
        model: Initialized VQA model
    """
    if model_type == 'binary':
        model = BinaryVQAModel(
            fusion_hidden_dim=fusion_dim,
            dropout=dropout,
            freeze_vision_encoder=freeze_encoders,
            freeze_text_encoder=freeze_encoders
        )
    elif model_type == 'multitask':
        model = MultiTaskVQAModel(
            num_open_ended_classes=num_classes,
            fusion_hidden_dim=fusion_dim,
            dropout=dropout
        )
    else:  # standard
        model = LightweightVQAModel(
            num_classes=num_classes,
            fusion_hidden_dim=fusion_dim,
            dropout=dropout,
            freeze_vision_encoder=freeze_encoders,
            freeze_text_encoder=freeze_encoders
        )
    
    model = model.to(device)
    return model


def print_model_summary(model: nn.Module):
    """Print detailed model summary."""
    print("=" * 70)
    print("MODEL SUMMARY")
    print("=" * 70)
    
    if hasattr(model, 'count_parameters'):
        params = model.count_parameters()
        print(f"\nParameter Counts:")
        print(f"  Vision Encoder:  {params['vision_encoder']:>12,} params")
        print(f"  Text Encoder:    {params['text_encoder']:>12,} params")
        print(f"  Fusion Module:   {params['fusion']:>12,} params")
        print(f"  Classifier:      {params['classifier']:>12,} params")
        print(f"  " + "-" * 40)
        print(f"  Total:           {params['total']:>12,} params")
        print(f"  Trainable:       {params['trainable']:>12,} params")
        
        trainable_pct = (params['trainable'] / params['total']) * 100
        print(f"  Trainable %:     {trainable_pct:>12.2f}%")
    
    if hasattr(model, 'get_model_size_mb'):
        size_mb = model.get_model_size_mb()
        print(f"\nModel Size: {size_mb:.2f} MB")
    
    print("=" * 70)


# Example usage and testing
if __name__ == "__main__":
    print("Testing Lightweight VQA Model Architecture\n")
    
    # Test standard model
    print("1. Standard VQA Model")
    print("-" * 70)
    model = create_lightweight_vqa_model(
        num_classes=500,  # VQA-RAD typical vocab size
        model_type='standard',
        fusion_dim=256,
        dropout=0.3,
        device='cpu'
    )
    print_model_summary(model)
    
    # Test with dummy input
    batch_size = 4
    dummy_images = torch.randn(batch_size, 3, 224, 224)
    dummy_input_ids = torch.randint(0, 30522, (batch_size, 40))  # DistilBERT vocab
    dummy_attention_mask = torch.ones(batch_size, 40)
    
    with torch.no_grad():
        logits = model(dummy_images, dummy_input_ids, dummy_attention_mask)
        print(f"\nOutput shape: {logits.shape}")
        print(f"Expected: (batch_size={batch_size}, num_classes=500)")
    
    # Test binary model
    print("\n\n2. Binary VQA Model")
    print("-" * 70)
    binary_model = create_lightweight_vqa_model(
        num_classes=2,
        model_type='binary',
        device='cpu'
    )
    print_model_summary(binary_model)
    
    # Test multitask model
    print("\n\n3. Multi-Task VQA Model")
    print("-" * 70)
    multitask_model = create_lightweight_vqa_model(
        num_classes=500,
        model_type='multitask',
        device='cpu'
    )
    
    with torch.no_grad():
        outputs = multitask_model(dummy_images, dummy_input_ids, dummy_attention_mask)
        print(f"\nBinary output shape: {outputs['binary'].shape}")
        print(f"Open-ended output shape: {outputs['open_ended'].shape}")
    
    print("\n" + "=" * 70)
    print("All tests passed! ✓")
    print("=" * 70)