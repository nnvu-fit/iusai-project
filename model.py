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
    if isinstance(backbone, nn.Sequential):
      self.features = nn.Sequential(*list(backbone.children())[:-1])
    else:
      self.features = nn.Sequential(*list(backbone.children()))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.features(x)
    x = x.view(x.size(0), -1)  # Flatten the output
    return x
  
class Classifier(nn.Module):
  def __init__(self, backbone: nn.Module):
    super(Classifier, self).__init__()

    # get the last layer from the backbone
    if isinstance(backbone, nn.Sequential):
      self.fc = backbone[-1]
    else:
      self.fc = backbone.fc if hasattr(backbone, 'fc') else backbone.classifier

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.fc(x)