import numpy as np
import torch
import torchvision
from torchvision.models import resnet50, resnet
from torch.utils.data import DataLoader, Subset
import copy
from typing import List, Dict, Tuple

import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms


class FederatedResNet50:
  def __init__(
      self,
      num_clients: int = 5,
      num_rounds: int = 10,
      local_epochs: int = 2,
      batch_size: int = 32,
      lr: float = 0.001,
      device: str = "cuda" if torch.cuda.is_available() else "cpu",
      **kwargs
  ):
    self.num_clients = num_clients
    self.num_rounds = num_rounds
    self.local_epochs = local_epochs
    self.batch_size = batch_size
    self.lr = lr
    self.device = device

    if 'weights' in kwargs:
      self.weights = kwargs['weights']
      print(f"Using weights: {self.weights}")
    else:
      self.weights = resnet.ResNet50_Weights.DEFAULT
      print("No weights provided, using default ResNet-50 weights.")

    # Initialize the global model (ResNet50)
    self.global_model = resnet50(weights=self.weights)
    num_ftrs = self.global_model.fc.in_features
    # Adjust number of classes as needed
    self.global_model.fc = nn.Linear(num_ftrs, 10)
    self.global_model = self.global_model.to(self.device)

    # Dataset preparation
    self.prepare_data()

  def prepare_data(self):
    """Prepare and split the dataset for federated learning"""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(224),  # ResNet50 requires 224x224 images
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    # Use CIFAR-10 for demonstration
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    self.testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)

    # Split the training set among clients
    self.client_data = self.split_data_for_clients(trainset)

  def split_data_for_clients(self, dataset):
    """Split the dataset into parts for each client"""
    num_items = len(dataset)
    items_per_client = num_items // self.num_clients
    client_data = []

    indices = np.random.permutation(num_items)
    for i in range(self.num_clients):
      start_idx = i * items_per_client
      end_idx = (i + 1) * items_per_client if i < self.num_clients - \
          1 else num_items
      client_indices = indices[start_idx:end_idx]
      client_data.append(DataLoader(
          Subset(dataset, client_indices),
          batch_size=self.batch_size,
          shuffle=True
      ))

    return client_data

  def train_client(self, client_id: int, model: nn.Module) -> nn.Module:
    """Train the model for a single client"""
    model = model.to(self.device)
    model.train()

    optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(self.local_epochs):
      for inputs, labels in self.client_data[client_id]:
        inputs, labels = inputs.to(self.device), labels.to(self.device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    return model

  def aggregate_models(self, client_models: List[nn.Module]) -> None:
    """Aggregate client models using FedAvg algorithm"""
    global_dict = self.global_model.state_dict()

    for k in global_dict.keys():
      global_dict[k] = torch.stack([client_models[i].state_dict()[k].float()
                                    for i in range(len(client_models))], 0).mean(0)

    self.global_model.load_state_dict(global_dict)

  def evaluate_model(self, model: nn.Module) -> Tuple[float, float]:
    """Evaluate the model on test data"""
    model.eval()
    test_loader = DataLoader(
        self.testset, batch_size=self.batch_size, shuffle=False)

    correct = 0
    total = 0
    loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
      for inputs, labels in test_loader:
        inputs, labels = inputs.to(self.device), labels.to(self.device)
        outputs = model(inputs)
        loss += criterion(outputs, labels).item()

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = loss / len(test_loader)
    return accuracy, avg_loss

  def train_federated(self):
    """Execute the federated learning process"""
    print(f"Starting Federated Learning with {self.num_clients} clients")

    for round_num in range(self.num_rounds):
      print(f"\nRound {round_num + 1}/{self.num_rounds}")

      # Distribute the global model to all clients
      client_models = []
      for client_id in range(self.num_clients):
        # Create a deep copy of the model for each client
        client_model = copy.deepcopy(self.global_model)

        # Train the client model
        print(f"Training client {client_id + 1}/{self.num_clients}")
        trained_client_model = self.train_client(client_id, client_model)
        client_models.append(trained_client_model)

      # Aggregate the client models to update the global model
      self.aggregate_models(client_models)

      # Evaluate the global model
      accuracy, avg_loss = self.evaluate_model(self.global_model)
      print(
          f"Round {round_num + 1} completed. Test Accuracy: {accuracy:.2f}%, Test Loss: {avg_loss:.4f}")

    print("\nFederated Learning completed!")

  def save_model(self, path: str):
    """Save the trained global model"""
    torch.save(self.global_model.state_dict(), path)
    print(f"Model saved to {path}")


if __name__ == "__main__":
  # Configure federated learning parameters
  fed_learning = FederatedResNet50(
      num_clients=5,
      num_rounds=10,
      local_epochs=2,
      batch_size=32,
      lr=0.001
  )

  # Start federated learning
  fed_learning.train_federated()

  # Save the final model
  fed_learning.save_model("federated_resnet50.pth")
