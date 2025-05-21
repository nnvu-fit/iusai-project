import os
import numpy as np
import torch
import time
from sklearn.metrics import classification_report, confusion_matrix

def collate_fn(batch):
    """
    Custom collate function for DataLoader that handles variable-sized inputs.
    Args:
        batch: A list of tuples (image, target)
    Returns:
        A tuple of (images, targets) where images is a tensor and targets is a list of dicts
    """
    images = []
    targets = []
    
    for image, target in batch:
        images.append(image)
        targets.append(target)
    
    # Stack images along a new batch dimension
    images = torch.stack(images, 0)
    
    return images, targets

class ClassifierTrainer:
  def __init__(self, model, optimizer, loss_fn, random_seed_value=None, device=None):
    self.model = model
    self.optimizer = optimizer
    self.loss_fn = loss_fn
    if device is None:
      device = get_device()
    self.device = device
    self.current_fold = 0
    
    # set runing timestamp for the model to save the result
    # in the folder with the timestamp name
    # timestamp must be taken by current time
    self.timestamp = time.strftime("%Y%m%d-%H%M%S")

    if random_seed_value is not None:
      seed_everything(random_seed_value)

  def cross_validate(self, train_dataset, k=5, epochs=1, batch_size=32):
    """
    Performs k-fold cross-validation on a PyTorch model using the specified optimizer and loss function.

    Args:
      model (torch.nn.Module): The PyTorch model to cross-validate.
      optimizer (torch.optim.Optimizer): The optimizer to use for cross-validation.
      loss_fn (callable): The loss function to use for cross-validation.
      dataset (torch.utils.data.Dataset): The dataset to use for cross-validation.
      k (int, optional): The number of folds to use for cross-validation. Defaults to 5.
      num_epochs (int, optional): The number of epochs to train for. Defaults to 1.
      device (str, optional): The device to use for cross-validation. If None, defaults to 'cuda' if available, else 'cpu'.

    Returns:
      float: The average loss on the given dataset across all folds.
    """
    self.model.to(self.device)
    fold_size = len(train_dataset) // k
    total_loss = 0.0
    report_metric = []
    for fold in range(k):
      self.current_fold = fold
      print(f'Fold {fold+1}/{k}:')
      train_data = torch.utils.data.Subset(train_dataset, list(range(fold_size * fold)) + list(range(fold_size * (fold + 1), len(train_dataset))))
      test_data = torch.utils.data.Subset(train_dataset, range(fold_size * fold, fold_size * (fold + 1)))
      train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
      test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
      
      lost_metric = self.train(train_loader, test_loader, epochs)
      report_metric.append(lost_metric)
      fold_loss = self.score(test_loader)
      total_loss += fold_loss

      y_true = []
      y_pred = []
      with torch.no_grad():
        for inputs, targets in test_loader:
          inputs, targets = inputs.to(self.device), targets.to(self.device)
          outputs = self.model(inputs)
          _, predicted = torch.max(outputs, 1)
          y_true += targets.tolist()
          y_pred += predicted.tolist()
      report = classification_report(y_true, y_pred)
      confusion = confusion_matrix(y_true, y_pred)
      self.save_report(report, confusion)

      print(f'Fold {fold+1}/{k}, Total Test Loss: {total_loss:.4f}, Fold accuracy: {100*(1 - fold_loss):.4f}')

      # save model by model
      torch.save(self.model.state_dict(), f"{self.model.__class__.__name__}\\{self.timestamp}\\fold_{fold}.pth")

    return [total_loss / k, report_metric]


  def train(self, train_loader, test_loader, epochs=1):
    """
    Trains a PyTorch model on a given dataset using the specified optimizer and loss function.

    Args:
      model (torch.nn.Module): The PyTorch model to train.
      optimizer (torch.optim.Optimizer): The optimizer to use for training.
      loss_fn (callable): The loss function to use for training.
      train_dataset (torch.utils.data.Dataset): The dataset to use for training.
      test_dataset (torch.utils.data.Dataset): The dataset to use for testing.
      num_epochs (int, optional): The number of epochs to train for. Defaults to 1.
      device (str, optional): The device to use for training. If None, defaults to 'cuda' if available, else 'cpu'.

    Returns:
      None
    """
    self.model.to(self.device)
    ## set start training in time
    start_time = time.time()
    report_metric = []
    for epoch in range(epochs):
      self.model.train()
      train_loss = 0.0
      for inputs, targets in train_loader:
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets)
        loss.backward()
        self. optimizer.step()
        train_loss += loss.item() * inputs.size(0)
      train_loss /= len(train_loader.dataset)
      self. model.eval()
      test_loss = 0.0
      with torch.no_grad():
        for inputs, targets in test_loader:
          inputs, targets = inputs.to(self.device), targets.to(self.device)
          outputs = self.model(inputs)
          loss = self.loss_fn(outputs, targets)
          test_loss += loss.item() * inputs.size(0)
        test_loss /= len(test_loader.dataset)
      ## set end training in time
      end_time = time.time()
      ## set total time
      total_time = end_time - start_time
      ## convert total time to hours, minutes, seconds
      total_time = time.strftime("%H hours, %M minutes, %S seconds", time.gmtime(total_time))
      ## print the result of training
      report_metric.append([epoch+1, train_loss, test_loss, total_time])
      print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Total Time: {total_time}')
      ## set start training in time
      start_time = time.time()
    return report_metric

  def score(self, test_dataset):
    """
    Scores a PyTorch model on a given dataset using the specified loss function.

    Args:
      model (torch.nn.Module): The PyTorch model to score.
      loss_fn (callable): The loss function to use for scoring.
      test_dataset (torch.utils.data.Dataset): The dataset to use for scoring.
      device (str, optional): The device to use for scoring. If None, defaults to 'cuda' if available, else 'cpu'.

    Returns:
      float: The average loss on the given dataset.
    """
    if self.device is None:
      self.device = get_device()
    self.model.to(self.device)
    self.model.eval()
    test_loss = 0.0
    with torch.no_grad():
      for inputs, targets in test_dataset:
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        outputs = self.model(inputs)
        loss = self.loss_fn(outputs, targets)
        test_loss += loss.item() * inputs.size(0)
      test_loss /= len(test_dataset.dataset)
    return test_loss

  def save_report(self, report, confusion):
    model_name = self.model.__class__.__name__ # get model name
    path_valid = "./results/" + model_name + "/"+ str(self.timestamp) +"/fold_" + str(self.current_fold) + ".txt"
    
    # create folder if not exist
    if not os.path.exists(os.path.dirname(path_valid)):
      os.makedirs(os.path.dirname(path_valid))
    
    # save report and confusion matrix
    with open(path_valid, 'w') as f:
        f.write(report)
        f.write('\n')
        f.write(str(confusion))
        

def get_device():
  """
  Returns the device to use for training and scoring. Defaults to 'cuda' if available, else 'cpu'.

  Returns:
    str: The device to use for training and scoring.
  """
  return 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

def seed_everything(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
