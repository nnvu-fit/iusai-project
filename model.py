import random
from typing import TypeVar
import torch
import torch.nn as nn
from torchvision import models
import torch

import torch.nn as nn


class FeatureExtractor(nn.Module):
  def __init__(self, backbone: nn.Module, output_size: int = None):
    super(FeatureExtractor, self).__init__()
    self.backbone = backbone

    # drop the last layer
    self.features = nn.Sequential(*list(backbone.children())[:-1])
    # If output_size is not provided, we will infer it from the backbone
    if output_size is None:
      # define a new fully connected layer
      # assuming the input size is 2048 for ResNet50
      if isinstance(backbone, models.ResNet):
        output_size = 2048
        self.features.append(nn.Flatten())
      elif isinstance(backbone, models.VGG):
        output_size = 25088  # VGG16/19
        self.features.append(nn.Flatten())
      elif isinstance(backbone, models.MobileNetV2):
        output_size = 1280  # MobileNetV2
        self.features.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.features.append(nn.Flatten())
        self.features.append(nn.Dropout(p=0.2))
      elif isinstance(backbone, models.DenseNet):
        output_size = 1024  # DenseNet121
        self.features.append(nn.ReLU(inplace=True))
        self.features.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.features.append(nn.Flatten())
      else:
        raise ValueError("Unsupported backbone model")
      
    # Fully connected layer to reduce dimensions
    self.fc1 = nn.Linear(output_size, 768)
    # Size of the output feature vector
    self.out_features = output_size

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
    self.backbone_out_features = backbone.out_features

    # get the last layer from the backbone
    self.fc = nn.Linear(backbone.out_features, output_size)  # Assuming binary classification

  def _get_name(self):
    return f"Classifier({self.backbone._get_name()})"

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.fc(x)  # Classify the features


class DDQNAgent:
  def __init__(self, num_actions: int, model: Classifier, batch_size=64, device=None):
    self.num_actions = num_actions
    self.q_net = model

    # Initialize the target network with the same architecture and weights
    # but set it to evaluation mode
    self.target_net = model
    self.target_net.load_state_dict(self.q_net.state_dict())
    self.target_net.eval()

    # Define the optimizer and loss function
    self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=0.001)
    self.criterion = nn.MSELoss()
    self.gamma = 0.99  # Discount factor for future rewards
    self.tau = 0.005  # Soft update factor for target network

    # Set the device for the model
    self.device = device
    if self.device is None:
      self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.q_net.to(self.device)
    self.target_net.to(self.device)

    self.memory = []  # Experience replay memory
    self.batch_size = batch_size  # Batch size for training

  def _get_name(self):
    return f"DDQNAgent({self.q_net._get_name()})"

  def select_action(self, state: torch.Tensor, epsilon: float) -> int:
    """
    Select an action based on epsilon-greedy policy.
    :param state: Current state of the environment.
    :param epsilon: Probability of selecting a random action.
    :return: Selected action index.
    """
    if torch.rand(1).item() < epsilon:
      # Select a random action
      return torch.randint(0, self.num_actions, (1,)).item()
    else:
      # Select the action with the highest Q-value
      with torch.no_grad():
        state = state.to(self.device)
        q_values = self.q_net(state)
        return q_values.argmax().item()

  def store_experience(self, state: torch.Tensor, action: int, reward: float, next_state: torch.Tensor, done: bool):
    """
    Store the experience in memory.
    :param state: Current state of the environment.
    :param action: Action taken.
    :param reward: Reward received.
    :param next_state: Next state of the environment.
    :param done: Whether the episode is done.
    """
    self.memory.append((state, action, reward, next_state, done))

  def sample_experience(self):
    """
    Sample a batch of experiences from memory.
    :return: A tuple of (states, actions, rewards, next_states, dones).
    """
    return zip(*random.sample(self.memory, min(len(self.memory), self.batch_size)))

  def update(self):
    """
    Update the Q-network using a batch of experiences.
    """
    if len(self.memory) < self.batch_size:
      return

    # Sample a batch of experiences
    states, actions, rewards, next_states, dones = self.sample_experience()

    states = torch.stack(states).to(self.device)
    actions = torch.tensor(actions).to(self.device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
    next_states = torch.stack(next_states).to(self.device)
    dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

    # Compute Q-values for the current states
    q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Compute Q-values for the next states
    with torch.no_grad():
      next_q_values = self.target_net(next_states).max(1)[0]
      expected_q_values = rewards + (1 - dones) * self.gamma * next_q_values

    # Compute the loss
    loss = self.criterion(q_values, expected_q_values)

    # Optimize the Q-network
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    # Soft update the target network
    for target_param, local_param in zip(self.target_net.parameters(), self.q_net.parameters()):
      target_param.data.copy_(self.tau * local_param.data + (1 - self.tau) * target_param.data)

  def train_q_net(self, data_loader, semantic_embeddings, epochs=10):
    """
    Train the agent using the stored experiences.
    """

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(self.q_net.parameters(), lr=0.001)

    prev_accuracy = ''

    for epoch in range(epochs):
      correct = 0
      total = 0
      loss_list = []
      self.q_net.train()  # Set the model to training mode
      for (inputs, labels) in data_loader:
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        # get the semantic embeddings for the labels
        label_embeddings = torch.stack([semantic_embeddings[str(label.item())].detach() for label in labels])
        # Subtract the semantic embeddings from the inputs
        inputs = inputs - label_embeddings.to(self.device)

        outputs = self.q_net(inputs)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        loss_list.append(loss.item())
      train_accuracy = 100 * correct / total
      avg_loss = sum(loss_list) / len(loss_list)

      if prev_accuracy != '' and train_accuracy != prev_accuracy:
        print()
      print(f'\rEpoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%', end='', flush=True)
      prev_accuracy = train_accuracy

    print()  # New line after the last epoch output
    # return the trained Q-network, average loss, and training accuracy
    return self.q_net, avg_loss, train_accuracy
