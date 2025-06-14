# import os, sys # to add the parent directory to the path
import os
import sys
import time

# Using torchvision to create a dataset
import cv2
from torchvision import transforms
import torch
from torch.utils.data import random_split, DataLoader
import torchvision
from torchvision.models import ResNet50_Weights

import pandas as pd

# add the parent directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

#  import self library
from trainer import ClassifierTrainer as Trainer
import dataset as ds # type: ignore
import model as md # type: ignore


def load_model_state(model, state_dict_path):
  """
  Load the state dictionary into the model.
  
  Args:
    model (torch.nn.Module): The model to load the state dictionary into.
    state_dict_path (str): The path to the state dictionary file.
  
  Returns:
    torch.nn.Module: The model with the loaded state dictionary.
  """
  if os.path.exists(state_dict_path):
    model.load_state_dict(torch.load(state_dict_path, weights_only=False))
    print(f"Model state loaded from {state_dict_path}")
  else:
    print(f"State dictionary file {state_dict_path} does not exist.")
  return model

def train_process(dataset, model):
  # define batch_size
  batch_size = 64

  # init train val test ds
  train_val_size = int(0.8 * len(dataset))
  test_size = len(dataset) - train_val_size
  train_ds, test_ds = random_split(dataset, [train_val_size, test_size])

  # define optimizer using Adam and loss function
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
  loss_fn = torch.nn.CrossEntropyLoss()

  trainer = Trainer(model, optimizer, loss_fn, random_seed_value=86)
  print('device: ', trainer.device)
  avg_loss, metric = trainer.cross_validate(train_ds, k=5, epochs=10, batch_size=batch_size)
  print('avg_loss: ', avg_loss)

  # score model
  test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)
  model_scored = trainer.score(test_dataloader)
  print(f'model_scored: {model_scored:.4f}, avg_accuracy: {100*(1 - model_scored):.4f}')

  # return model scored, train_avg_lost
  return model_scored, avg_loss

if __name__ == "__main__":

  datasets = {
    'gi4e_full': ds.Gi4eDataset(
        './datasets/gi4e',
        transform=transforms.Compose([transforms.ToPILImage(), transforms.Resize((224, 224)), transforms.ToTensor()]),
        is_classification=True),
    # 'gi4e_raw_eyes': ds.ImageDataset(
    #   './datasets/gi4e_raw_eyes',
    #   transform=transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor()]),
    #   file_extension='png'),
    # 'gi4e_detected_eyes': ds.ImageDataset(
    #   './datasets/gi4e_eyes/20250521_200316',
    #   transform=transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor()]),
    #   file_extension='png'),
  }

  models = [
    load_model_state(torchvision.models.resnet50(weights=None), './models/ResNet/20250611_103405/fold_4.pth'),
    # torchvision.models.densenet121(weights=torchvision.models.DenseNet121_Weights.DEFAULT),
    # load_model_state(torchvision.models.vgg16(weights=None), './models/vgg16.pth'),
  ]

  embedded_models = [md.FeatureExtractor(model) for model in models]
  classifier_models = [md.Classifier(model, 103) for model in embedded_models]
  print('Embedded models:', [model._get_name() for model in embedded_models])
  print('Classifier models:', [model._get_name() for model in classifier_models])

  classifier_df = pd.DataFrame(columns=['key', 'dataset', 'model'])
  result_df = pd.DataFrame(columns=['dataset', 'model', 'avg_loss', 'avg_accuracy', 'total_time'])

  print('--' * 30)
  # Loop through datasets and models to create the classifier_df
  print('Starting to get features for datasets...')
  for name, dataset in datasets.items():
    for model in embedded_models:
      print(f'Getting features for {name} dataset with {model._get_name()}')

      # get the classifier model from the model
      classifier_model = next((m for m in classifier_models if m.backbone._get_name() == model._get_name()), None)
      if classifier_model is None:
        print(f'Classifier model not found for {model._get_name()}')
        continue

      model_dataset = ds.EmbeddedDataset(dataset, model)

      classifier_df = pd.concat([classifier_df, pd.DataFrame({
          'key': [f'{name}_{model._get_name()}'],
          'model': [classifier_model],
          'dataset': [model_dataset]
      })], ignore_index=True)

      print(f'Finished getting features for {name} dataset with {model._get_name()}')
  
  print('--' * 30)
  # Loop through the classifier_df to train each model on each dataset
  print('Starting to train models on datasets...')
  for index, row in classifier_df.iterrows():
    dataset = row['dataset']
    model = row['model']
    key = row['key']

    print(f'Running {key} dataset with {model._get_name()}')
    # do the train
    start_time = time.time()
    scored, loss = train_process(dataset, model)
    end_time = time.time()
    total_time = end_time - start_time
    print(f'Finished {key} dataset with {model._get_name()}')
    print('----------------------')

    # save the result
    result_df = pd.concat([result_df, pd.DataFrame({
        'model': [model._get_name()],
        'dataset': [dataset.__class__.__name__],
        'avg_loss': [loss],
        'avg_accuracy': [100*(1 - scored)],
        'total_time': [total_time]
    })], ignore_index=True)
  print('Finished training all models')
  print(result_df)