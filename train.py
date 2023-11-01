import torch
import time

def train(model, optimizer, loss_fn, train_dataset, test_dataset, epochs=1, device=None):
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
  if device is None:
    device = get_device()
  model.to(device)
  ## set start training in time
  start_time = time.time()
  for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for inputs, targets in train_dataset:
      inputs, targets = inputs.to(device), targets.to(device)
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = loss_fn(outputs, targets)
      loss.backward()
      optimizer.step()
      train_loss += loss.item() * inputs.size(0)
    train_loss /= len(train_dataset.dataset)
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
      for inputs, targets in test_dataset:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        test_loss += loss.item() * inputs.size(0)
      test_loss /= len(test_dataset.dataset)
    ## set end training in time
    end_time = time.time()
    ## set total time
    total_time = end_time - start_time
    ## convert total time to hours, minutes, seconds
    total_time = time.strftime("%H hours, %M minutes, %S seconds", time.gmtime(total_time))
    ## print the result of training
    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}, Total Time: {total_time}')
    ## set start training in time
    start_time = time.time()

## generate score function to scoring the model performance on test dataset with lost function
def score_model(model, loss_fn, test_dataset, device=None):
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
  if device is None:
    device = get_device()
  model.to(device)
  model.eval()
  test_loss = 0.0
  with torch.no_grad():
    for inputs, targets in test_dataset:
      inputs, targets = inputs.to(device), targets.to(device)
      outputs = model(inputs)
      loss = loss_fn(outputs, targets)
      test_loss += loss.item() * inputs.size(0)
    test_loss /= len(test_dataset.dataset)
  return test_loss

def get_device():
  """
  Returns the device to use for training and scoring. Defaults to 'cuda' if available, else 'cpu'.

  Returns:
    str: The device to use for training and scoring.
  """
  return 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'