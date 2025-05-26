from typing import TypeVar
import torch
import torch.nn as nn
from torchvision import models

# define generic linear classifier
TClass = TypeVar('TClass', bound=nn.Module)


class EmbededClassifier(TClass):
  # Base class for classifiers that embed features.
  def __init__(self, **kwargs):
    """
    Initialize the classifier with the given parameters.
    Args:
        kwargs (dict): Dictionary containing the parameters for the classifier.
            Expected keys: 'in_features' (int), 'num_classes' (int).
    """
    super(EmbededClassifier, self).__init__(kwargs)
    self.fc_temp = super.fc
    # check if super().fc is defined
    if not hasattr(super(), 'fc'):
      raise NotImplementedError("The base class must implement the 'fc' method.")

    # assign fc to a temporary variable
    self.fc_temp = super.fc

  # define fc function to do nothing
  def fc(self, x: torch.Tensor) -> torch.Tensor:
    """    Dummy function to be overridden in subclasses.
    Args:
        x (torch.Tensor): Input tensor.
    Returns:
        torch.Tensor: Output tensor.
    """
    return x

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Forward pass through the classifier.
    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_features).
    Returns:
        torch.Tensor: Output tensor of shape (batch_size, num_classes).
    """
    x = self.forward(x)
    return self.fc(x)


class LinearClassifier(EmbededClassifier):
  def __init__(self, **kwargs):
    """
    Initialize the linear classifier with the given parameters.
    Args:
        kwargs (dict): Dictionary containing the parameters for the classifier.
            Expected keys: 'in_features' (int), 'num_classes' (int).
    """
    super(LinearClassifier, self).__init__(kwargs)

    self.fc = self.fc_temp

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Forward pass through the linear classifier.
    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, in_features).
    Returns:
        torch.Tensor: Output tensor of shape (batch_size, num_classes).
    """
    super().forward(x)
    if not hasattr(self, 'fc'):
      raise NotImplementedError("The 'fc' method must be implemented in the subclass.")
    return self.fc(x)
