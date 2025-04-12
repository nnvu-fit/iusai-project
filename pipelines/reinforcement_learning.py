import torch
from torch.distributions import Categorical
import numpy as np
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

class ResNetRL:
  def __init__(self, num_classes=1000, learning_rate=0.001, gamma=0.99, **kwargs):
    """
    Initialize ResNet-50 with RL components
    
    Args:
      num_classes: Number of output classes
      learning_rate: Learning rate for policy optimization
      gamma: Discount factor for rewards
    """
    # check if weights are available in kwargs
    if 'weights' in kwargs:
      weights = kwargs['weights']
      print(f"Using weights: {weights}")
    else:
      weights = models.ResNet50_Weights.DEFAULT
      print("No weights provided, using default ResNet-50 weights.")
      
    # Load pre-trained ResNet-50
    self.resnet = models.resnet50(weights=weights)
    
    # Replace final layer for our task
    in_features = self.resnet.fc.in_features
    self.resnet.fc = nn.Sequential(
      nn.Linear(in_features, 512),
      nn.ReLU(),
      nn.Linear(512, num_classes)
    )
    
    # Policy network to guide ResNet updates
    self.policy_net = nn.Sequential(
      nn.Linear(num_classes, 128),
      nn.ReLU(),
      nn.Linear(128, 2)  # Binary action: strengthen or weaken current features
    )
    
    self.optimizer = optim.Adam(list(self.resnet.parameters()) + 
                  list(self.policy_net.parameters()), 
                  lr=learning_rate)
    
    self.gamma = gamma
    self.saved_log_probs = []
    self.rewards = []
    
  def forward(self, x):
    """Forward pass through the ResNet model and policy network"""
    features = self.resnet(x)
    policy_logits = self.policy_net(features)
    return features, policy_logits
  
  def select_action(self, state):
    """Select an action based on current state using the policy network"""
    features, policy_logits = self.forward(state)
    
    # Create a categorical distribution over the actions
    m = Categorical(logits=policy_logits)
    action = m.sample()
    
    # Save log probability for backprop
    self.saved_log_probs.append(m.log_prob(action))
    
    return features, action.item()
  
  def apply_action(self, layer_idx, action):
    """Apply selected action to modify ResNet weights"""
    # Get the target layer
    target_layers = list(self.resnet.parameters())
    if layer_idx < len(target_layers):
      layer = target_layers[layer_idx]
      
      # Action 0: Strengthen important features (increase weights)
      # Action 1: Weaken less important features (decrease weights)
      modifier = 0.01 if action == 0 else -0.01
      with torch.no_grad():
        layer.data *= (1 + modifier)
  
  def finish_episode(self):
    """Update policy parameters based on collected rewards"""
    R = 0
    policy_loss = []
    returns = []
    
    # Calculate discounted returns
    for r in self.rewards[::-1]:
      R = r + self.gamma * R
      returns.insert(0, R)
      
    returns = torch.tensor(returns)
    # Normalize returns for stability
    returns = (returns - returns.mean()) / (returns.std() + 1e-9)
    
    # Calculate policy loss
    for log_prob, R in zip(self.saved_log_probs, returns):
      policy_loss.append(-log_prob * R)
      
    # Perform backpropagation
    self.optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    self.optimizer.step()
    
    # Reset episode data
    self.saved_log_probs = []
    self.rewards = []
  
  def train(self, train_loader, val_loader, num_episodes=10):
    """Train the model using RL to improve ResNet performance"""
    criterion = nn.CrossEntropyLoss()
    best_accuracy = 0.0
    
    for episode in range(num_episodes):
      print(f"Episode {episode+1}/{num_episodes}")
      
      # Training phase
      self.resnet.train()
      running_loss = 0.0
      correct = 0
      total = 0
      
      for inputs, targets in tqdm(train_loader):
        if torch.cuda.is_available():
          inputs, targets = inputs.cuda(), targets.cuda()
        
        # Get predictions and select action
        outputs, action = self.select_action(inputs)
        
        # Apply the action to a random layer
        layer_idx = np.random.randint(0, len(list(self.resnet.parameters())))
        self.apply_action(layer_idx, action)
        
        # Standard supervised loss
        loss = criterion(outputs, targets)
        
        # Forward and backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Reward based on prediction accuracy
        batch_accuracy = predicted.eq(targets).sum().item() / targets.size(0)
        self.rewards.append(batch_accuracy)
      
      # End of episode - update policy
      self.finish_episode()
      
      # Validation phase
      self.resnet.eval()
      val_correct = 0
      val_total = 0
      
      with torch.no_grad():
        for inputs, targets in val_loader:
          if torch.cuda.is_available():
            inputs, targets = inputs.cuda(), targets.cuda()
          
          outputs = self.resnet(inputs)
          _, predicted = outputs.max(1)
          val_total += targets.size(0)
          val_correct += predicted.eq(targets).sum().item()
      
      train_accuracy = 100.0 * correct / total
      val_accuracy = 100.0 * val_correct / val_total
      
      print(f"Training Accuracy: {train_accuracy:.2f}%")
      print(f"Validation Accuracy: {val_accuracy:.2f}%")
      
      # Save best model
      if val_accuracy > best_accuracy:
        print("Saving best model...")
        best_accuracy = val_accuracy
        torch.save(self.resnet.state_dict(), 'resnet50_rl_improved.pth')
    
    print(f"Best validation accuracy: {best_accuracy:.2f}%")
    return self.resnet


# Example usage:
if __name__ == "__main__":
  import torchvision.transforms as transforms
  
  # Set up data transformations
  transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
  ])
  
  transform_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
  ])
  
  # Example dataset paths (replace with your dataset path)
  train_dataset = ImageFolder(root='./data/train', transform=transform_train)
  val_dataset = ImageFolder(root='./data/val', transform=transform_val)
  
  train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
  val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
  
  # Create and train the RL-enhanced ResNet-50
  resnet_rl = ResNetRL(num_classes=len(train_dataset.classes))
  improved_model = resnet_rl.train(train_loader, val_loader, num_episodes=5)