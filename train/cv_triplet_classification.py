
import os
import sys
import torchvision
import pandas as pd

# add the parent directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

current_backbone_model = None
current_dataset = None

def save_model(model_path, model):
  """
  Save the model to the given path.

  Args:
    model_path: The path to save the model to.
    model: The model to save.
  """
  import torch

  # Ensure the model path is a string
  if not isinstance(model_path, str):
    raise ValueError("model_path must be a string")

  # Ensure the directory exists
  if not os.path.exists(os.path.dirname(model_path)):
    os.makedirs(os.path.dirname(model_path))
  torch.save(model.state_dict(), model_path)
  print(f'Model saved to {model_path}')


def validate_model(model, dataset, batch_size=64):
  """
  Validate the model on the given dataset and return the average loss and accuracy.

  Args:
    model: The model to validate.
    dataset: The dataset to validate on.
    batch_size: The batch size to use for validation.
  """
  from trainer import get_device
  from torch.utils.data import DataLoader
  import torch

  device = get_device()
  model = model.to(device)
  model.eval()

  loss_fn = torch.nn.CrossEntropyLoss()
  data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

  total_loss = 0.0
  correct_predictions = 0
  total_samples = 0

  with torch.no_grad():
    for inputs, labels in data_loader:
      inputs, labels = inputs.to(device), labels.to(device)
      outputs = model(inputs)
      loss = loss_fn(outputs, labels)
      total_loss += loss.item()

      # Calculate accuracy
      _, predicted = torch.max(outputs.data, 1)
      total_samples += labels.size(0)
      correct_predictions += (predicted == labels).sum().item()

  # Calculate average loss and accuracy
  avg_loss = total_loss / len(data_loader)
  accuracy = (correct_predictions / total_samples) * 100
  print(f'Validation average loss: {avg_loss}, Accuracy: {accuracy:.2f}%')
  return avg_loss, accuracy


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
    train_ds, test_ds = torch.utils.data.random_split(
        dataset, [train_size, test_size])

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
        anchor, positive, negative = anchor.to(
            device), positive.to(device), negative.to(device)

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
          anchor, positive, negative = anchor.to(
              device), positive.to(device), negative.to(device)

          output_anchor = model(anchor)
          output_positive = model(positive)
          output_negative = model(negative)

          loss = loss_fn(output_anchor, output_positive, output_negative)
          total_test_loss += loss.item()

      # Calculate average loss for the epoch
      avg_loss = total_loss / len(train_loader)
      avg_test_loss = total_test_loss / len(test_loader)
      print(
          f'Fold {fold + 1}: Average loss: {avg_loss}, Average test loss: {avg_test_loss}')
      print(
          f'Time taken for fold {fold + 1}, epoch {epoch + 1}: {time.time() - start_time:.2f} seconds')
      loss_list.append(avg_loss)
      test_loss_list.append(avg_test_loss)
    print(f'Fold {fold + 1} completed.')

    # save the model after each fold
    model_dir = f'models/triplet/{current_dataset}_{model._get_name()}'
    save_model(f'{model_dir}/model_fold_{fold + 1}.pth', model)

  # Print the average loss over all folds
  average_loss = sum(loss_list) / len(loss_list)
  average_test_loss = sum(test_loss_list) / len(test_loss_list)
  print(
      f'Over all folds: Average loss : {average_loss}, Average test loss: {average_test_loss}')

  # Return the trained model and the average loss
  return model, average_loss, average_test_loss


def classification_train_process(dataset, model, k_fold=5, batch_size=64, test_dataset=None):
  """
  Train the model on the given dataset and return the scored model and average loss.
  """
  from trainer import get_device
  from torch.utils.data import DataLoader
  import time
  import torch
  import pandas as pd

  device = get_device()

  # create result df
  result_df = pd.DataFrame(columns=['dataset', 'model', 'fold', 'avg_loss', 'avg_test_loss',
                           'avg_val_loss', 'accuracy', 'total_time'])

  for fold in range(k_fold):
    print(f'Running fold {fold + 1}/{k_fold}...')
    fold_start_time = time.time()
    # Initialize the trainer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Split dataset into train and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Train the model
    epochs = 10
    loss_list = []
    val_loss_list = []
    # loop through epochs
    for epoch in range(epochs):
      print(f'Epoch {epoch + 1}/{epochs}...')
      epoch_start_time = time.time()
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

      epoch_end_time = time.time()
      # loop through test set to evaluate
      model.eval()
      total_val_loss = 0.0
      with torch.no_grad():
        for batch in val_loader:
          inputs, labels = batch
          inputs, labels = inputs.to(device), labels.to(device)

          outputs = model(inputs)
          loss = loss_fn(outputs, labels)
          total_val_loss += loss.item()

      # Calculate average loss for the epoch
      avg_loss = total_loss / len(train_loader)
      avg_val_loss = total_val_loss / len(val_loader)
      print(
          f'Fold {fold + 1}: Average loss: {avg_loss}, Average validation loss: {avg_val_loss}')
      print(
          f'Time taken for fold {fold + 1}, epoch {epoch + 1}: {epoch_end_time - epoch_start_time:.2f} seconds')
      loss_list.append(avg_loss)
      val_loss_list.append(avg_val_loss)
    print(f'Fold {fold + 1} completed.')

    # save the model after each fold
    model_dir = f'models/classification/{current_dataset}_{model._get_name()}'
    save_model(f'{model_dir}/model_fold_{fold + 1}.pth', model)

    # Validate the model on the test set each fold if provided
    if test_dataset is not None:
      avg_test_loss, accuracy = validate_model(
          model, test_dataset, batch_size=batch_size)
      print(
          f'Fold {fold + 1}: Average test loss: {avg_test_loss}, Accuracy: {accuracy:.2f}%')
      result_df = pd.concat([result_df, pd.DataFrame({
          'dataset': [dataset.__class__.__name__],
          'model': [model._get_name()],
          'fold': [fold + 1],
          'avg_loss': [avg_loss],
          'avg_test_loss': [avg_test_loss],
          'avg_val_loss': [avg_val_loss],
          'accuracy': [accuracy],
          'total_time': [time.time() - fold_start_time]
      })], ignore_index=True)

  # Print the average loss over all folds
  average_loss = sum(loss_list) / len(loss_list)
  average_val_loss = sum(val_loss_list) / len(val_loss_list)
  print(
      f'Over all folds: Average loss : {average_loss}, Average validation loss: {average_val_loss}')

  # save the results to a CSV file to keep track of the training process by current_backbone_model and current_dataset
  import pandas as pd
  if current_backbone_model is not None and current_dataset is not None:
    result_df['dataset'] = current_dataset
  if not result_df.empty:
    result_dir = 'results/cv-triplet'
    if not os.path.exists(result_dir):
      os.makedirs(result_dir)
    result_path = f'{result_dir}/{current_dataset}_{current_backbone_model._get_name()}_{time.strftime("%Y%m%d-%H%M%S")}.csv'
    # If the directory does not exist, create it
    # save the results to a CSV file with the current dataset and model in name and date
    result_df.to_csv(result_path, index=False)

  # Return the trained model and the average loss
  return model, average_loss, average_val_loss


def train(dataset, model, train_process='triplet', k_fold=5, batch_size=32, test_dataset=None):
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
    trained_model, avg_loss, avg_test_loss = triplet_train_process(
        dataset, model, k_fold=k_fold, batch_size=batch_size)
  elif train_process == 'classification':
    trained_model, avg_loss, avg_test_loss = classification_train_process(
        dataset, model, k_fold=k_fold, batch_size=batch_size, test_dataset=test_dataset)
  else:
    raise ValueError(f'Unknown training process: {train_process}')
  print('Training completed.')
  return trained_model, avg_loss, avg_test_loss


def create_training_process_df(
        dataset_type,
        create_triplet_dataset_fn,
        create_classification_dataset_fn,
        create_classification_test_dataset_fn=None,
        models: list[str] = ['resnet', 'vgg', 'mobilenet', 'densenet'],
        batch_size=32):
  """
  Create a DataFrame to store the training process information.
  """
  import pandas as pd

  triplet_df = pd.DataFrame(columns=[
      'backbone_model',
      'feature_extractor_model',
      'dataset_type',
      'create_triplet_dataset_fn',
      'create_classification_dataset_fn',
      'create_classification_test_dataset_fn',
      'batch_size'
  ])

  # from modes, create a list of functions to create backbone models
  create_backbone_model_funcs = []
  for model in models:
    if model == 'resnet':
      create_backbone_model_funcs.append(lambda: torchvision.models.resnet50(weights=None))
    elif model == 'vgg':
      create_backbone_model_funcs.append(lambda: torchvision.models.vgg16(weights=None))
    elif model == 'mobilenet':
      create_backbone_model_funcs.append(lambda: torchvision.models.mobilenet_v2(weights=None))
    elif model == 'densenet':
      create_backbone_model_funcs.append(lambda: torchvision.models.densenet121(weights=None))
    else:
      raise ValueError(f'Unknown model type: {model}')
    

  # Add all backbone models to the DataFrame
  for create_fn in create_backbone_model_funcs:
    triplet_df = pd.concat([triplet_df, pd.DataFrame({
        'backbone_model': [create_fn()],
        'feature_extractor_model': [FeatureExtractor(create_fn())],
        'dataset_type': [dataset_type],
        'create_triplet_dataset_fn': [create_triplet_dataset_fn],
        'create_classification_dataset_fn': [create_classification_dataset_fn],
        'create_classification_test_dataset_fn': [create_classification_test_dataset_fn],
        'batch_size': [batch_size]
    })], ignore_index=True)

  return triplet_df


def create_train_test_dataset(create_train_dataset_fn, create_test_dataset_fn=None):
  """
  Create a train and test dataset from the given functions.
  """
  import torch

  # Create the train dataset
  train_dataset = create_train_dataset_fn()

  # If a test dataset function is provided, use it to create the test dataset
  if create_test_dataset_fn is not None:
    test_dataset = create_test_dataset_fn()
  else:
    # Otherwise, split the train dataset into train and test sets
    train_size = int(0.8 * len(train_dataset))
    test_size = len(train_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, test_size])

  return train_dataset, test_dataset


if __name__ == '__main__':
  import dataset as ds
  from model import FeatureExtractor, Classifier

  triplet_df = pd.DataFrame(columns=[
      'backbone_model',
      'feature_extractor_model',
      'dataset_type',
      'create_triplet_dataset_fn',
      'create_classification_dataset_fn',
      'create_classification_test_dataset_fn',
      'batch_size'
  ])
  classifier_df = pd.DataFrame(columns=['dataset', 'model', 'transform'])
  # DataFrame to store results of the training process
  result_df = pd.DataFrame(columns=[
      'dataset', 'model', 'avg_loss', 'avg_test_loss', 'avg_val_loss', 'accuracy', 'total_time'
  ])

  # # gi4e_full dataset
  # def create_gi4e_triplet_dataset_fn(): return ds.TripletGi4eDataset(
  #     './datasets/gi4e',
  #     transform=torchvision.transforms.Compose([
  #         torchvision.transforms.ToPILImage(),
  #         torchvision.transforms.Resize((224, 224)),
  #         torchvision.transforms.ToTensor()
  #     ]))
  # def create_gi4e_classification_dataset_fn(): return ds.Gi4eDataset(
  #     './datasets/gi4e',
  #     transform=torchvision.transforms.Compose([
  #         torchvision.transforms.ToPILImage(),
  #         torchvision.transforms.Resize((224, 224)),
  #         torchvision.transforms.ToTensor()
  #     ]),
  #     is_classification=True)
  # # Add training process for gi4e_full dataset
  # triplet_df = pd.concat([triplet_df, create_training_process_df(
  #     'gi4e_full',
  #     create_gi4e_triplet_dataset_fn,
  #     create_gi4e_classification_dataset_fn
  # )], ignore_index=True)

  # gi4e_raw_eyes dataset
  def create_gi4e_raw_eyes_triplet_dataset_fn(): return ds.TripletImageDataset(
      './datasets/gi4e_raw_eyes',
      file_extension='png',
      transform=torchvision.transforms.Compose([
          torchvision.transforms.Resize((224, 224)),
          torchvision.transforms.ToTensor()
      ]))

  def create_gi4e_raw_eyes_classification_dataset_fn(): return ds.ImageDataset(
      './datasets/gi4e_raw_eyes',
      file_extension='png',
      transform=torchvision.transforms.Compose([
          torchvision.transforms.Resize((224, 224)),
          torchvision.transforms.ToTensor()
      ]))
  # Add training process for gi4e_raw_eyes dataset
  triplet_df = pd.concat([triplet_df, create_training_process_df(
      'gi4e_raw_eyes',
      create_gi4e_raw_eyes_triplet_dataset_fn,
      create_gi4e_raw_eyes_classification_dataset_fn,
      models=['resnet'],
  )], ignore_index=True)

  # # Youtube Faces dataset
  # def create_youtube_faces_triplet_dataset_fn(): return ds.TripletYoutubeFacesDataset(
  #     data_path='./datasets/YouTubeFacesWithFacialKeypoints',
  #     transform=torchvision.transforms.Compose([
  #         torchvision.transforms.ToPILImage(),
  #         torchvision.transforms.Resize((224, 224)),
  #         torchvision.transforms.ToTensor()
  #     ]),
  #     number_of_samples=10,  # Limit the number of samples for faster training
  # )
  # def create_youtube_faces_classification_dataset_fn(): return ds.YoutubeFacesWithFacialKeypoints(
  #     data_path='./datasets/YouTubeFacesWithFacialKeypoints',
  #     transform=torchvision.transforms.Compose([
  #         torchvision.transforms.ToPILImage(),
  #         torchvision.transforms.Resize((224, 224)),
  #         torchvision.transforms.ToTensor()
  #     ]),
  #     number_of_samples=10,  # Limit the number of samples for faster training
  #     is_classification=True
  # )
  # # Add training process for YouTube Faces dataset
  # triplet_df = pd.concat([triplet_df, create_training_process_df(
  #     'youtube_faces',
  #     create_youtube_faces_triplet_dataset_fn,
  #     create_youtube_faces_classification_dataset_fn,
  #     batch_size=16  # Adjust batch size as needed
  # )], ignore_index=True)

  # # CelebA dataset
  # def create_celeb_a_triplet_dataset_fn(): return ds.TripletImageDataset(
  #     './datasets/CelebA_HQ_facial_identity_dataset/train',
  #     file_extension='jpg',
  #     transform=torchvision.transforms.Compose([
  #         torchvision.transforms.Resize((224, 224)),
  #         torchvision.transforms.ToTensor()
  #     ]),
  # )
  # def create_celeb_a_classification_dataset_fn(): return ds.ImageDataset(
  #     './datasets/CelebA_HQ_facial_identity_dataset/train',
  #     file_extension='jpg',
  #     transform=torchvision.transforms.Compose([
  #         torchvision.transforms.Resize((224, 224)),
  #         torchvision.transforms.ToTensor()
  #     ]),
  # )
  # def create_celeb_a_classification_test_dataset_fn(): return ds.ImageDataset(
  #     './datasets/CelebA_HQ_facial_identity_dataset/test',
  #     file_extension='jpg',
  #     transform=torchvision.transforms.Compose([
  #         torchvision.transforms.Resize((224, 224)),
  #         torchvision.transforms.ToTensor()
  #     ]),
  # )
  # # Add training process for CelebA dataset
  # triplet_df = pd.concat([triplet_df, create_training_process_df(
  #     'celeb_a',
  #     create_celeb_a_triplet_dataset_fn,
  #     create_celeb_a_classification_dataset_fn,
  #     create_celeb_a_classification_test_dataset_fn
  # )], ignore_index=True)

  # # Nus2Hands dataset
  # def create_nus2hands_triplet_dataset_fn(): return ds.TripletImageDataset(
  #     './datasets/nus2hands',
  #     file_extension='jpg',
  #     transform=torchvision.transforms.Compose([
  #         torchvision.transforms.Resize((224, 224)),
  #         torchvision.transforms.ToTensor()
  #     ]),
  # )
  # def create_nus2hands_classification_dataset_fn(): return ds.ImageDataset(
  #     './datasets/nus2hands',
  #     file_extension='jpg',
  #     transform=torchvision.transforms.Compose([
  #         torchvision.transforms.Resize((224, 224)),
  #         torchvision.transforms.ToTensor()
  #     ]),
  # )
  # # Add training process for Nus2Hands dataset
  # triplet_df = pd.concat([triplet_df, create_training_process_df(
  #     'nus2hands',
  #     create_nus2hands_triplet_dataset_fn,
  #     create_nus2hands_classification_dataset_fn,
  #     batch_size=32  # Adjust batch size as needed
  # )], ignore_index=True)

  # # FER2013 dataset
  # def create_fer2013_triplet_dataset_fn(): return ds.TripletImageDataset(
  #     './datasets/fer2013/train',
  #     file_extension='jpg',
  #     transform=torchvision.transforms.Compose([
  #         torchvision.transforms.Resize((224, 224)),
  #         torchvision.transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel
  #         torchvision.transforms.ToTensor()
  #     ]),
  # )
  # def create_fer2013_classification_dataset_fn(): return ds.ImageDataset(
  #     './datasets/fer2013/train',
  #     file_extension='jpg',
  #     transform=torchvision.transforms.Compose([
  #         torchvision.transforms.Resize((224, 224)),
  #         torchvision.transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel
  #         torchvision.transforms.ToTensor()
  #     ]),
  # )
  # def create_fer2013_classification_test_dataset_fn(): return ds.ImageDataset(
  #     './datasets/fer2013/test',
  #     file_extension='jpg',
  #     transform=torchvision.transforms.Compose([
  #         torchvision.transforms.Resize((224, 224)),
  #         torchvision.transforms.Grayscale(num_output_channels=3),  # Convert to 3-channel
  #         torchvision.transforms.ToTensor()
  #     ])
  # )
  # # Add training process for FER2013 dataset
  # triplet_df = pd.concat([triplet_df, create_training_process_df(
  #     'fer2013',
  #     create_fer2013_triplet_dataset_fn,
  #     create_fer2013_classification_dataset_fn,
  #     create_fer2013_classification_test_dataset_fn
  # )], ignore_index=True)

  # The process of training models following the triplet loss approach the same way as classification
  # loop through datasets and train triplet models
  for index, row in triplet_df.iterrows():
    batch_size = row['batch_size']
    backbone_model = row['backbone_model']
    triplet_model = row['feature_extractor_model']
    dataset_type = row['dataset_type']
    create_gi4e_triplet_dataset_fn = row['create_triplet_dataset_fn']
    create_classification_dataset_fn = row['create_classification_dataset_fn']
    create_classification_test_dataset_fn = row.get(
        'create_classification_test_dataset_fn', None)
    current_backbone_model = backbone_model

    # Run Cross-Validation on dataset to verify raw performance of backbone model
    print(
        f'Running Cross-Validation on dataset {row["dataset_type"]} with model {backbone_model._get_name()}...')
    current_dataset = dataset_type + '_Cross-Validation_None'
    classifier_dataset = create_classification_dataset_fn()
    # split the dataset into train and validation sets
    train_ds, test_ds = create_train_test_dataset(
        create_classification_dataset_fn,
        create_classification_test_dataset_fn
    )
    # Create the classifier model
    trained_model, avg_loss, avg_val_loss = train(
        train_ds, backbone_model, train_process='classification', k_fold=5, batch_size=batch_size, test_dataset=test_ds)
    print(
        f'Cross-Validation completed for dataset {row["dataset_type"]} with model {backbone_model._get_name()}.')
    # Validate the model on the test set
    avg_test_loss, accuracy = validate_model(
        trained_model, test_ds, batch_size=batch_size)
    print(
        f'Average loss: {avg_loss}, Average val loss: {avg_val_loss}, Average test loss: {avg_test_loss}, Accuracy: {accuracy:.2f}%')
    result_df = pd.concat([result_df, pd.DataFrame({
        'model': [backbone_model._get_name()],
        'dataset': [dataset_type + ': Cross-Validation'],
        'avg_loss': [avg_loss],
        'avg_val_loss': [avg_val_loss],
        'avg_test_loss': [avg_test_loss],
        'accuracy': [accuracy],
        # Placeholder for total time, can be calculated if needed
        'total_time': [0]
    })], ignore_index=True)

    # Create the triplet dataset
    triplet_dataset = create_gi4e_triplet_dataset_fn()
    # First, train the triplet model
    print(
        f'Training triplet model {triplet_model._get_name()} on dataset {triplet_dataset.__class__.__name__}...')
    trained_model, avg_loss, avg_val_loss = train(
        triplet_dataset, triplet_model, train_process='triplet', k_fold=5, batch_size=batch_size)
    print(
        f'Triplet model {triplet_model._get_name()} trained on dataset {triplet_dataset.__class__.__name__}.')
    print(f'Average loss: {avg_loss}, Average test loss: {avg_val_loss}')

    # After training the triplet model, we can also train a classification model based on the same dataset without moving labels to function
    current_dataset = dataset_type + '_Cross-Validation_Triplet'
    classifier_model = Classifier(trained_model)
    train_ds, test_ds = create_train_test_dataset(
        create_classification_dataset_fn,
        create_classification_test_dataset_fn
    )
    train_ds = ds.EmbeddedDataset(
        train_ds, trained_model, is_moving_labels_to_function=False)
    test_ds = ds.EmbeddedDataset(
        test_ds, trained_model, is_moving_labels_to_function=False)
    print(
        f'Training classification model {classifier_model._get_name()} on dataset {classifier_dataset.__class__.__name__}...')
    trained_classifier_model, avg_loss, avg_val_loss = train(
        train_ds, classifier_model, train_process='classification', k_fold=5, batch_size=batch_size, test_dataset=test_ds)
    print(
        f'Classification model {classifier_model._get_name()} trained on dataset {classifier_dataset.__class__.__name__}.')
    # Validate the model on the test set
    avg_test_loss, accuracy = validate_model(
        trained_classifier_model, test_ds, batch_size=batch_size)
    print(
        f'Average loss: {avg_loss}, Average val loss: {avg_val_loss}, Average test loss: {avg_test_loss}, Accuracy: {accuracy:.2f}%')

    result_df = pd.concat([result_df, pd.DataFrame({
        'model': [classifier_model._get_name()],
        'dataset': [dataset_type + ': Cross-Validation + Triplet'],
        'avg_loss': [avg_loss],
        'avg_val_loss': [avg_val_loss],
        'avg_test_loss': [avg_test_loss],
        'accuracy': [accuracy],
        # Placeholder for total time, can be calculated if needed
        'total_time': [0]
    })], ignore_index=True)

    # Training the classifier model with moving labels to function: move to labels embeddings
    current_dataset = dataset_type + '_Cross-Validation_Triplet_Moving_Labels_x-to-x'
    classifier_model = Classifier(trained_model)
    train_ds, test_ds = create_train_test_dataset(
        create_classification_dataset_fn,
        create_classification_test_dataset_fn
    )
    train_ds = ds.EmbeddedDataset(
        train_ds, trained_model, is_moving_labels_to_function=True)
    test_ds = ds.EmbeddedDataset(
        test_ds, trained_model, is_moving_labels_to_function=True)
    # Example transformation, can be customized
    train_ds.apply_function_to_labels_embeddings(lambda x: x)
    # Example transformation, can be customized
    test_ds.apply_function_to_labels_embeddings(lambda x: x)
    print(
        f'Training classification model {classifier_model._get_name()} on dataset {classifier_dataset.__class__.__name__}...')
    trained_classifier_model, avg_loss, avg_val_loss = train(
        train_ds, classifier_model, train_process='classification', k_fold=5, batch_size=batch_size, test_dataset=test_ds)
    print(
        f'Classification model {classifier_model._get_name()} trained on dataset {classifier_dataset.__class__.__name__}.')
    # Validate the model on the test set
    avg_test_loss, accuracy = validate_model(
        trained_classifier_model, test_ds, batch_size=batch_size)
    print(
        f'Average loss: {avg_loss}, Average val loss: {avg_val_loss}, Average test loss: {avg_test_loss}, Accuracy: {accuracy:.2f}%')

    result_df = pd.concat([result_df, pd.DataFrame({
        'model': [classifier_model._get_name()],
        'dataset': [dataset_type + ': Cross-Validation + Triplet + Moving Labels - x => x'],
        'avg_loss': [avg_loss],
        'avg_val_loss': [avg_val_loss],
        'avg_test_loss': [avg_test_loss],
        'accuracy': [accuracy],
        # Placeholder for total time, can be calculated if needed
        'total_time': [0]
    })], ignore_index=True)

    # Training the classifier model with moving labels to function: move to labels embeddings x4
    current_dataset = dataset_type + '_Cross-Validation_Triplet_Moving_Labels_x-to-4x'
    classifier_model = Classifier(trained_model)
    train_ds, test_ds = create_train_test_dataset(
        create_classification_dataset_fn,
        create_classification_test_dataset_fn
    )
    train_ds = ds.EmbeddedDataset(
        train_ds, trained_model, is_moving_labels_to_function=True)
    test_ds = ds.EmbeddedDataset(
        test_ds, trained_model, is_moving_labels_to_function=True)
    # Example transformation, can be customized
    train_ds.apply_function_to_labels_embeddings(lambda x: 4 * x)
    # Example transformation, can be customized
    test_ds.apply_function_to_labels_embeddings(lambda x: 4 * x)

    print(
        f'Training classification model {classifier_model._get_name()} on dataset {classifier_dataset.__class__.__name__}...')
    trained_classifier_model, avg_loss, avg_val_loss = train(
        train_ds, classifier_model, train_process='classification', k_fold=5, batch_size=batch_size, test_dataset=test_ds)
    print(
        f'Classification model {classifier_model._get_name()} trained on dataset {classifier_dataset.__class__.__name__}.')
    # Validate the model on the test set
    avg_test_loss, accuracy = validate_model(
        trained_classifier_model, test_ds, batch_size=batch_size)
    print(
        f'Average loss: {avg_loss}, Average val loss: {avg_val_loss}, Average test loss: {avg_test_loss}, Accuracy: {accuracy:.2f}%')

    result_df = pd.concat([result_df, pd.DataFrame({
        'model': [classifier_model._get_name()],
        'dataset': [dataset_type + ': Cross-Validation + Triplet + Moving Labels - x => 4*x'],
        'avg_loss': [avg_loss],
        'avg_val_loss': [avg_val_loss],
        'avg_test_loss': [avg_test_loss],
        'accuracy': [accuracy],
        # Placeholder for total time, can be calculated if needed
        'total_time': [0]
    })], ignore_index=True)

  print('Training process completed.')
  # Save the results to a CSV file
  result_df.to_csv('triplet_training_results.csv', index=False)
