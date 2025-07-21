
import os
import sys

# add the parent directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def triplet_train_process(dataset, model, k_fold=5, batch_size=64):
  """
  Train the model on the given dataset and return the scored model and average loss.
  """
  from trainer import get_device
  from torch.utils.data import DataLoader
  import time
  import torch

  device = get_device()

  for fold in range(k_fold):
    print(f'Running fold {fold + 1}/{k_fold}...')

    # Initialize the trainer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    loss_fn = torch.nn.TripletMarginLoss(margin=0.2, p=2)

    # Split dataset into train and test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])

    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Train the model
    epochs = 10
    loss_list = []
    test_loss_list = []
    # loop through epochs
    for epoch in range(epochs):
      print(f'Epoch {epoch + 1}/{epochs}...')
      start_time = time.time()
      model = model.to(device)
      model.train()
      total_loss = 0.0

      # loop through batches and train
      for batch in train_loader:
        anchor, positive, negative = batch
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

        optimizer.zero_grad()
        output_anchor = model(anchor)
        output_positive = model(positive)
        output_negative = model(negative)

        loss = loss_fn(output_anchor, output_positive, output_negative)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

      # loop through test set to evaluate
      model.eval()
      total_test_loss = 0.0
      with torch.no_grad():
        for batch in test_loader:
          anchor, positive, negative = batch
          anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

          output_anchor = model(anchor)
          output_positive = model(positive)
          output_negative = model(negative)

          loss = loss_fn(output_anchor, output_positive, output_negative)
          total_test_loss += loss.item()

      # Calculate average loss for the epoch
      avg_loss = total_loss / len(train_loader)
      avg_test_loss = total_test_loss / len(test_loader)
      print(f'Fold {fold + 1} average loss: {avg_loss}')
      print(f'Fold {fold + 1} average test loss: {avg_test_loss}')

      loss_list.append(avg_loss)
      test_loss_list.append(avg_test_loss)
    print(f'Fold {fold + 1} completed.')
  print(f'Average loss over all folds: {sum(loss_list) / len(loss_list)}')
  print(f'Average test loss over all folds: {sum(test_loss_list) / len(test_loss_list)}')

  # Return the trained model and the average loss
  return model, sum(loss_list) / len(loss_list), sum(test_loss_list) / len(test_loss_list)

def classification_train_process(dataset, model, k_fold=5, batch_size=64):
  """
  Train the model on the given dataset and return the scored model and average loss.
  """
  from trainer import get_device
  from torch.utils.data import DataLoader
  import time
  import torch

  device = get_device()

  for fold in range(k_fold):
    print(f'Running fold {fold + 1}/{k_fold}...')

    # Initialize the trainer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Split dataset into train and test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])

    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Train the model
    epochs = 10
    loss_list = []
    test_loss_list = []
    # loop through epochs
    for epoch in range(epochs):
      print(f'Epoch {epoch + 1}/{epochs}...')
      start_time = time.time()
      model = model.to(device)
      model.train()
      total_loss = 0.0

      # loop through batches and train
      for batch in train_loader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

      # loop through test set to evaluate
      model.eval()
      total_test_loss = 0.0
      with torch.no_grad():
        for batch in test_loader:
          inputs, labels = batch
          inputs, labels = inputs.to(device), labels.to(device)

          outputs = model(inputs)
          loss = loss_fn(outputs, labels)
          total_test_loss += loss.item()

      # Calculate average loss for the epoch
      avg_loss = total_loss / len(train_loader)
      avg_test_loss = total_test_loss / len(test_loader)
      print(f'Fold {fold + 1} average loss: {avg_loss}')
      print(f'Fold {fold + 1} average test loss: {avg_test_loss}')

      loss_list.append(avg_loss)
      test_loss_list.append(avg_test_loss)
    print(f'Fold {fold + 1} completed.')
  print(f'Average loss over all folds: {sum(loss_list) / len(loss_list)}')
  print(f'Average test loss over all folds: {sum(test_loss_list) / len(test_loss_list)}')

def train(dataset, model, train_process='triplet', k_fold=5, batch_size=64):
  """
  Train the model on the given dataset and return the scored model and average loss.

  Args:
    dataset: The dataset to train on.
    model: The model to train.
    train_process: The training process to use, either 'triplet' or 'classification'.
    k_fold: The number of folds for k-fold cross-validation.
    batch_size: The batch size to use for training.
  """
  print('Starting training process...')
  if train_process == 'triplet':
    trained_model, avg_loss, avg_test_loss = triplet_train_process(dataset, model, k_fold=k_fold, batch_size=batch_size)
  elif train_process == 'classification':
    trained_model, avg_loss, avg_test_loss = classification_train_process(dataset, model, k_fold=k_fold, batch_size=batch_size)
  else:
    raise ValueError(f'Unknown training process: {train_process}')
  print('Training completed.')
  return trained_model, avg_loss, avg_test_loss


if __name__ == '__main__':
  import pandas as pd
  import torchvision
  from dataset import ImageDataset, EmbeddedDataset, TripletImageDataset
  from model import FeatureExtractor, Classifier

  triplet_df = pd.DataFrame(columns=['dataset_path', 'model'])
  classifier_df = pd.DataFrame(columns=['dataset', 'model'])
  # DataFrame to store results of the training process
  result_df = pd.DataFrame(columns=['dataset', 'model', 'avg_loss', 'avg_accuracy', 'total_time'])

  # Add datasets and models to the training process DataFrame
  transform = torchvision.transforms.Compose([
      torchvision.transforms.Resize((224, 224)),
      torchvision.transforms.ToTensor(),
  ])
  dataset_path = './datasets/gi4e_raw_eyes'  # Replace with your image directory path
  model = FeatureExtractor(torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1))
  triplet_df = pd.concat([triplet_df, pd.DataFrame({
      'dataset_path': [dataset_path],
      'model': [model]
  })], ignore_index=True)

  # The process of training models following the triplet loss approach the same way as classification
  # loop through datasets and train triplet models
  for index, row in triplet_df.iterrows():
    triplet_dataset = TripletImageDataset(row['dataset_path'], file_extension='png', transform=transform)
    model = row['model']

    # First, train the triplet model
    print(f'Training triplet model {model._get_name()} on dataset {triplet_dataset.__class__.__name__}...')
    trained_model, avg_loss, avg_test_loss = train(triplet_dataset, model, train_process='triplet', k_fold=5, batch_size=64)
    print(f'Triplet model {model._get_name()} trained on dataset {triplet_dataset.__class__.__name__}.')
    print(f'Average loss: {avg_loss}, Average test loss: {avg_test_loss}')

    # After training the triplet model, we can also train a classification model based on the same dataset
    image_dataset = ImageDataset(row['dataset_path'], file_extension='png', transform=transform)
    classifier_dataset = EmbeddedDataset(image_dataset, trained_model)
    classifier_model = Classifier(trained_model)
    print(f'Training classification model {classifier_model._get_name()} on dataset {classifier_dataset.__class__.__name__}...')
    trained_classifier_model, avg_loss, avg_test_loss = train(classifier_dataset, classifier_model, train_process='classification', k_fold=5, batch_size=64)
    print(f'Classification model {classifier_model._get_name()} trained on dataset {classifier_dataset.__class__.__name__}.')
    print(f'Average loss: {avg_loss}, Average test loss: {avg_test_loss}')

    result_df = pd.concat([result_df, pd.DataFrame({
        'model': [model._get_name()],
        'dataset': [classifier_dataset._get_name()],
        'avg_loss': [avg_loss],
        'avg_accuracy': [100 * (1 - avg_test_loss)],
        'total_time': [0]  # Placeholder for total time, can be calculated if needed
    })], ignore_index=True)



  # Save the results to a CSV file
  result_df.to_csv('triplet_training_results.csv', index=False)
