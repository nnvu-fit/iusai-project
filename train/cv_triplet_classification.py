
import os
import sys

# add the parent directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def train_process(dataset, model):
  """
  Train the model on the given dataset and return the scored model and average loss.
  """
  from trainer import get_device
  from torch.utils.data import DataLoader
  import time
  import torch

  k_fold = 5
  batch_size = 64
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


def train(dataset, model):
  """
  Train the model on the given dataset and return the scored model and average loss.
  """
  print('Starting training process...')
  trained_model, avg_loss, avg_test_loss = train_process(dataset, model)
  print('Training completed.')
  return trained_model, avg_loss, avg_test_loss


if __name__ == '__main__':
  import pandas as pd
  import torchvision
  from dataset import ImageDataset, TripletDataset
  from model import FeatureExtractor, Classifier

  train_process_df = pd.DataFrame(columns=['dataset', 'model'])
  result_df = pd.DataFrame(columns=['dataset', 'model', 'avg_loss', 'avg_accuracy', 'total_time'])

  # Add datasets and models to the training process DataFrame
  transform = torchvision.transforms.Compose([
      torchvision.transforms.Resize((224, 224)),
      torchvision.transforms.ToTensor(),
  ])
  image_path = 'path/to/your/image/directory'  # Replace with your image directory path
  image_dataset = ImageDataset(image_path, transform=transform)
  model = FeatureExtractor(torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1))
  train_process_df = train_process_df.append({
      'dataset': TripletDataset(dataset=image_dataset, model=model),
      'model': model
  }, ignore_index=True)

  # loop through datasets and train models
  for index, row in train_process_df.iterrows():
    dataset = row['dataset']
    model = row['model']

    print(f'Training model {model._get_name()} on dataset {dataset.__class__.__name__}...')
    trained_model, avg_loss, avg_test_loss = train(dataset, model)
    print(f'Model {model._get_name()} trained on dataset {dataset.__class__.__name__}.')

    result_df = pd.concat([result_df, pd.DataFrame({
        'model': [model._get_name()],
        'dataset': [dataset.__class__.__name__],
        'avg_loss': [avg_loss],
        'avg_accuracy': [100 * (1 - avg_test_loss)],
        'total_time': [0]  # Placeholder for total time, can be calculated if needed
    })], ignore_index=True)

    print('Training process completed for all models.')
  result_df.to_csv('triplet_training_results.csv', index=False)
