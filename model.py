from typing import TypeVar
import torch
import torch.nn as nn
from torchvision import models
import torch

import torch.nn as nn


class FeatureExtractor(nn.Module):
  def __init__(self, backbone: nn.Module):
    super(FeatureExtractor, self).__init__()
    self.backbone = backbone

    # drop the last layer
    self.features = nn.Sequential(*list(backbone.children())[:-1])

    # define a new fully connected layer
    # assuming the input size is 2048 for ResNet50
    if isinstance(backbone, models.ResNet):
      output_size = 2048
    elif isinstance(backbone, models.VGG):
      output_size = 25088  # VGG16/19
    elif isinstance(backbone, models.MobileNetV2):
      output_size = 1280  # MobileNetV2
    elif isinstance(backbone, models.DenseNet):
      output_size = 1024  # DenseNet121
    else:
      raise ValueError("Unsupported backbone model")
    self.fc1 = nn.Linear(output_size, 768)  # Fully connected layer to reduce dimensions

    self.out_features = output_size  # Size of the output feature vector

  def _get_name(self):
    return f"FeatureExtractor({self.backbone.__class__.__name__})"

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.features(x)
    # Return the feature vector
    return x
  
class Classifier(FeatureExtractor):
  def __init__(self, backbone: FeatureExtractor, output_size: int = 1000):
    super(Classifier, self).__init__(backbone.backbone)
    # Ensure the backbone is an instance of FeatureExtractor
    if not isinstance(backbone, FeatureExtractor):
      raise ValueError("backbone must be an instance of FeatureExtractor")
    self.backbone = backbone

    # get the last layer from the backbone
    self.fc = nn.Linear(backbone.out_features, output_size)  # Assuming binary classification

  def _get_name(self):
    return f"Classifier({self.backbone._get_name()})"

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.fc(x)  # Classify the features
  