#!/usr/bin/env python3
"""
Baseline VQA Model: ResNet-34 + BERT-base
Heavier baseline model to compare against LightweightVQAModel (MobileNetV3-Small + DistilBERT)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from transformers import BertModel
from typing import Dict


class AttentionFusion(nn.Module):
    """Attention-based fusion of visual and textual features (same design as lightweight)."""

    def __init__(
        self,
        visual_dim: int = 512,       # ResNet-34 pooled output
        text_dim: int = 768,         # BERT-base hidden size
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.visual_dim = visual_dim
        self.text_dim = text_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Project visual and text features to same dimension
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)

        # Multi-head self-attention over {visual, text}
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
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
            nn.Dropout(dropout),
        )

    def forward(self, visual_features, text_features):
        """
        Args:
            visual_features: (batch, visual_dim)
            text_features:   (batch, text_dim)
        Returns:
            fused_features:  (batch, hidden_dim)
            attn_weights:    (batch, num_heads, seq_len, seq_len) with seq_len=2
        """
        v = self.visual_proj(visual_features).unsqueeze(1)  # (B, 1, H)
        t = self.text_proj(text_features).unsqueeze(1)      # (B, 1, H)

        # Sequence of 2 tokens: [vision, text]
        combined = torch.cat([v, t], dim=1)  # (B, 2, H)

        # Self-attention
        attn_out, attn_weights = self.multihead_attn(
            combined, combined, combined
        )

        # Residual + norm
        combined = self.norm1(combined + attn_out)

        # FFN + residual + norm
        ffn_out = self.ffn(combined)
        combined = self.norm2(combined + ffn_out)

        # Global average pooling along sequence dimension
        fused = combined.mean(dim=1)  # (B, H)

        return fused, attn_weights


class BaselineVQAModel(nn.Module):
    """
    Baseline VQA model combining ResNet-34 and BERT-base.

    Architecture:
    - Vision: ResNet-34 (pretrained on ImageNet)
    - Language: BERT-base-uncased (pretrained)
    - Fusion: Attention-based multimodal fusion (same as lightweight)
    - Classifier: 2-layer MLP (same structure as lightweight)
    """

    def __init__(
        self,
        num_classes: int,
        visual_feature_dim: int = 512,   # ResNet-34 global pooled output
        text_feature_dim: int = 768,     # BERT-base hidden size
        fusion_hidden_dim: int = 512,
        num_attention_heads: int = 8,
        dropout: float = 0.3,
        freeze_vision_encoder: bool = False,
        freeze_text_encoder: bool = False,
    ):
        super().__init__()

        self.num_classes = num_classes

        # ============= Vision Encoder: ResNet-34 =============
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        # Remove the final FC layer, keep up to global pooling
        self.vision_encoder = nn.Sequential(*list(resnet.children())[:-1])  # (B, 512, 1, 1)

        if freeze_vision_encoder:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

        # ============= Text Encoder: BERT-base =============
        self.text_encoder = BertModel.from_pretrained("bert-base-uncased")

        if freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        # ============= Multimodal Fusion =============
        self.fusion = AttentionFusion(
            visual_dim=visual_feature_dim,
            text_dim=text_feature_dim,
            hidden_dim=fusion_hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
        )

        # ============= Answer Classifier (same style as lightweight) =============
        self.classifier = nn.Sequential(
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.LayerNorm(fusion_hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(fusion_hidden_dim // 2, num_classes),
        )

        self._init_classifier_weights()

    def _init_classifier_weights(self):
        """Initialize classifier with Xavier initialization."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    # ================= Feature extraction helpers =================
    def extract_visual_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (batch, 3, 224, 224)
        Returns:
            features: (batch, 512)
        """
        features = self.vision_encoder(images)     # (B, 512, 1, 1)
        features = features.view(features.size(0), -1)  # (B, 512)
        return features

    def extract_text_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
        Returns:
            text_features: (batch, 768) from [CLS]
        """
        outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # [CLS] representation
        text_features = outputs.last_hidden_state[:, 0, :]  # (B, 768)
        return text_features

    # ================= Forward =================
    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_attention: bool = False,
    ):
        """
        Forward pass.

        Args:
            images: (batch, 3, 224, 224)
            input_ids: (batch, seq_len)
            attention_mask: (batch, seq_len)
            return_attention: whether to return attention weights

        Returns:
            logits: (batch, num_classes)
            attention_weights (optional)
        """
        visual_features = self.extract_visual_features(images)
        text_features = self.extract_text_features(input_ids, attention_mask)

        fused_features, attention_weights = self.fusion(
            visual_features, text_features
        )

        logits = self.classifier(fused_features)

        if return_attention:
            return logits, attention_weights
        return logits

    # ================= Utils: parameter counts & size =================
    def count_parameters(self) -> Dict[str, int]:
        """Count trainable and total parameters for each component."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())

        vision_params = sum(p.numel() for p in self.vision_encoder.parameters())
        text_params = sum(p.numel() for p in self.text_encoder.parameters())
        fusion_params = sum(p.numel() for p in self.fusion.parameters())
        classifier_params = sum(p.numel() for p in self.classifier.parameters())

        return {
            "total": total,
            "trainable": trainable,
            "vision_encoder": vision_params,
            "text_encoder": text_params,
            "fusion": fusion_params,
            "classifier": classifier_params,
        }

    def get_model_size_mb(self) -> float:
        """Approximate model size in MB."""
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        size_mb = (param_size + buffer_size) / (1024 ** 2)
        return size_mb


class BinaryBaselineVQAModel(BaselineVQAModel):
    """
    Specialized baseline model for binary (yes/no) questions.
    Uses same architecture but with 2 output classes.
    """

    def __init__(self, **kwargs):
        kwargs["num_classes"] = 2
        super().__init__(**kwargs)


class MultiTaskBaselineVQAModel(nn.Module):
    """
    Multi-task baseline model that handles both binary and open-ended questions.
    Shares encoders but has separate classification heads (same idea as lightweight).
    """

    def __init__(
        self,
        num_open_ended_classes: int,
        visual_feature_dim: int = 512,
        text_feature_dim: int = 768,
        fusion_hidden_dim: int = 512,
        num_attention_heads: int = 8,
        dropout: float = 0.3,
    ):
        super().__init__()

        # Shared backbone
        self.base_model = BaselineVQAModel(
            num_classes=num_open_ended_classes,  # will not be used directly
            visual_feature_dim=visual_feature_dim,
            text_feature_dim=text_feature_dim,
            fusion_hidden_dim=fusion_hidden_dim,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
        )

        self.vision_encoder = self.base_model.vision_encoder
        self.text_encoder = self.base_model.text_encoder
        self.fusion = self.base_model.fusion

        # Separate heads
        self.binary_head = nn.Sequential(
            nn.Linear(fusion_hidden_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2),  # Yes/No
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

            nn.Linear(fusion_hidden_dim // 2, num_open_ended_classes),
        )

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        question_type=None,
    ):
        """
        Args:
            question_type: list of 'binary' or 'open-ended' or None.
        """
        visual_features = self.base_model.extract_visual_features(images)
        text_features = self.base_model.extract_text_features(
            input_ids, attention_mask
        )
        fused_features, _ = self.fusion(visual_features, text_features)

        if question_type is None:
            # Return both heads (for separate loss computation)
            binary_logits = self.binary_head(fused_features)
            open_ended_logits = self.open_ended_head(fused_features)
            return {
                "binary": binary_logits,
                "open_ended": open_ended_logits,
                "features": fused_features,
            }

        # Mixed batch routing
        outputs = []
        for i, qtype in enumerate(question_type):
            feat = fused_features[i : i + 1]
            if qtype == "binary":
                outputs.append(self.binary_head(feat))
            else:
                outputs.append(self.open_ended_head(feat))
        return torch.cat(outputs, dim=0)


def create_baseline_vqa_model(
    num_classes: int,
    model_type: str = "standard",  # 'standard', 'binary', 'multitask'
    fusion_dim: int = 256,
    dropout: float = 0.3,
    freeze_encoders: bool = False,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> nn.Module:
    """
    Factory function to create baseline VQA model.

    Args:
        num_classes: Number of answer classes (for open-ended)
        model_type: 'standard', 'binary', or 'multitask'
        fusion_dim: Hidden dimension for fusion
        dropout: Dropout probability
        freeze_encoders: Whether to freeze pretrained encoders
        device: Device to load model on

    Returns:
        model: Initialized baseline VQA model
    """
    if model_type == "binary":
        model = BinaryBaselineVQAModel(
            fusion_hidden_dim=fusion_dim,
            dropout=dropout,
            freeze_vision_encoder=freeze_encoders,
            freeze_text_encoder=freeze_encoders,
        )
    elif model_type == "multitask":
        model = MultiTaskBaselineVQAModel(
            num_open_ended_classes=num_classes,
            fusion_hidden_dim=fusion_dim,
            dropout=dropout,
        )
    else:  # standard
        model = BaselineVQAModel(
            num_classes=num_classes,
            fusion_hidden_dim=fusion_dim,
            dropout=dropout,
            freeze_vision_encoder=freeze_encoders,
            freeze_text_encoder=freeze_encoders,
        )

    model = model.to(device)
    return model


def print_model_summary(model: nn.Module):
    """Print detailed model summary (mirrors lightweight version)."""
    print("=" * 70)
    print("BASELINE MODEL SUMMARY (ResNet-34 + BERT-base)")
    print("=" * 70)

    if hasattr(model, "count_parameters"):
        params = model.count_parameters()
        print(f"\nParameter Counts:")
        print(f"  Vision Encoder:  {params['vision_encoder']:>12,} params")
        print(f"  Text Encoder:    {params['text_encoder']:>12,} params")
        print(f"  Fusion Module:   {params['fusion']:>12,} params")
        print(f"  Classifier:      {params['classifier']:>12,} params")
        print("  " + "-" * 40)
        print(f"  Total:           {params['total']:>12,} params")
        print(f"  Trainable:       {params['trainable']:>12,} params")
        trainable_pct = (params["trainable"] / params["total"]) * 100
        print(f"  Trainable %:     {trainable_pct:>12.2f}%")

    if hasattr(model, "get_model_size_mb"):
        size_mb = model.get_model_size_mb()
        print(f"\nModel Size: {size_mb:.2f} MB")

    print("=" * 70)


# ================= Example usage and testing =================
if __name__ == "__main__":
    print("Testing Baseline VQA Model Architecture (ResNet-34 + BERT-base)\n")

    # 1. Standard model
    print("1. Standard Baseline VQA Model")
    print("-" * 70)
    model = create_baseline_vqa_model(
        num_classes=500,   # same as your lightweight example
        model_type="standard",
        fusion_dim=256,
        dropout=0.3,
        device="cpu",
    )
    print_model_summary(model)

    # Dummy input
    batch_size = 4
    dummy_images = torch.randn(batch_size, 3, 224, 224)
    dummy_input_ids = torch.randint(0, 30522, (batch_size, 40))  # BERT vocab size
    dummy_attention_mask = torch.ones(batch_size, 40, dtype=torch.long)

    with torch.no_grad():
        logits = model(dummy_images, dummy_input_ids, dummy_attention_mask)
        print(f"\nOutput shape: {logits.shape}")
        print(f"Expected: (batch_size={batch_size}, num_classes=500)")

    # 2. Binary model
    print("\n\n2. Binary Baseline VQA Model")
    print("-" * 70)
    binary_model = create_baseline_vqa_model(
        num_classes=2,
        model_type="binary",
        device="cpu",
    )
    print_model_summary(binary_model)

    with torch.no_grad():
        binary_logits = binary_model(dummy_images, dummy_input_ids, dummy_attention_mask)
        print(f"\nBinary output shape: {binary_logits.shape}")
        print(f"Expected: (batch_size={batch_size}, num_classes=2)")

    # 3. Multi-task model
    print("\n\n3. Multi-Task Baseline VQA Model")
    print("-" * 70)
    multitask_model = create_baseline_vqa_model(
        num_classes=500,
        model_type="multitask",
        device="cpu",
    )

    with torch.no_grad():
        outputs = multitask_model(dummy_images, dummy_input_ids, dummy_attention_mask)
        print(f"\nBinary head output shape: {outputs['binary'].shape}")
        print(f"Open-ended head output shape: {outputs['open_ended'].shape}")

    print("\n" + "=" * 70)
    print("All baseline tests passed! ✓")
    print("=" * 70)