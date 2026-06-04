"""Model definition for the Stratocaster origin classifier.

EfficientNet-B0 pretrained on ImageNet, with the final classifier layer
replaced to output ``num_classes`` logits.
"""

import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """Build an EfficientNet-B0 with a fresh ``num_classes``-way head.

    Set ``pretrained=False`` when loading saved weights for inference, so the
    ImageNet checkpoint is not downloaded unnecessarily.
    """
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model
