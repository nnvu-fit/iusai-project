from typing import TypeVar
import torch
import torch.nn as nn
from torchvision import models
from typing import Generic
import torch

import torch.nn as nn

T = TypeVar('T', bound=nn.Module)

class FeatureExtractor(nn.Module, Generic[T]):
  """Base model that extracts features without classification layer"""
  
  def __init__(self, backbone: T = None, pretrained: bool = True):
    super().__init__()
    
    if backbone is None:
      # Default to ResNet50 if no backbone provided
      backbone = models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT if pretrained else None)
      if pretrained:
        print("Using pretrained ResNet50 backbone")
    
    # Find and store the feature dimension and remove classification layer
    if hasattr(backbone, 'fc'):
      # ResNet, DenseNet style
      self.feature_dim = backbone.fc.in_features
      backbone.fc = nn.Identity()
    elif hasattr(backbone, 'classifier'):
      # VGG, EfficientNet style
      if isinstance(backbone.classifier, nn.Sequential):
        self.feature_dim = backbone.classifier[0].in_features
        backbone.classifier = nn.Identity()
      else:
        self.feature_dim = backbone.classifier.in_features
        backbone.classifier = nn.Identity()
    else:
      raise ValueError("Unsupported backbone: cannot locate classification layer")
      
    self.backbone = backbone
    
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Extract features from input"""
    features = self.backbone(x)
    return features


class Classifier(FeatureExtractor[T]):
  """Model with classification layer on top of feature extractor"""
  
  def __init__(self, num_classes: int, backbone: T = None, pretrained: bool = True):
    super().__init__(backbone=backbone, pretrained=pretrained)
    
    # Add the classification layer
    self.classifier = nn.Linear(self.feature_dim, num_classes)
    
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    features = super().forward(x)
    output = self.classifier(features)
    return output
