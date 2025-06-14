from typing import TypeVar
import torch
import torch.nn as nn
from torchvision import models
from typing import Generic
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
      input_size = 2048
    elif isinstance(backbone, models.VGG):
      input_size = 25088  # VGG16/19
    elif isinstance(backbone, models.MobileNetV2):
      input_size = 1280  # MobileNetV2
    else:
      raise ValueError("Unsupported backbone model")
    self.fc1 = nn.Linear(input_size, 768)  # Fully connected layer to reduce dimensions

  def _get_name(self):
    return f"FeatureExtractor({self.backbone.__class__.__name__})"

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.features(x)
    x = x.view(x.size(0), -1)  # Flatten the output
    x = nn.ReLU(True)(self.fc1(x))  # Apply ReLU activation
    x = nn.Dropout(0.5)(x)  # Apply dropout for regularization
    # Return the feature vector
    return x
  
class Classifier(FeatureExtractor):
  def __init__(self, backbone: FeatureExtractor, output_size: int = 2):
    super(Classifier, self).__init__(backbone.backbone)
    # Ensure the backbone is an instance of FeatureExtractor
    if not isinstance(backbone, FeatureExtractor):
      raise ValueError("backbone must be an instance of FeatureExtractor")
    self.backbone = backbone

    # get the last layer from the backbone
    self.fc = nn.Linear(backbone.fc1.out_features, output_size)  # Assuming binary classification

  def _get_name(self):
    return f"Classifier({self.backbone._get_name()})"

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.fc(x)