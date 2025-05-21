# import os, sys # to add the parent directory to the path
import os
import sys

# Using torchvision to create a dataset
import cv2
from torchvision import transforms
import torch
from torch.utils.data import random_split, DataLoader
import torchvision
from torchvision.models import ResNet50_Weights

# add the parent directory to the path
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

#  import self library
from trainer import ClassifierTrainer as Trainer
import dataset as ds # type: ignore


def doTheTrain(dataset, model):
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
    avg_lost = trainer.cross_validate(train_ds, k=10, epochs=10, batch_size=batch_size)
    print('avg_lost: ', avg_lost)

    # score model
    test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)
    model_scored = trainer.score(test_dataloader)
    print(f'model_scored: {model_scored:.4f}, avg_accuracy: {100*(1 - model_scored):.4f}')


if __name__ == "__main__":
  model = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)

  datasets = {
    # ds.Gi4eDataset(
    #   './datasets/gi4e',
    #   transform=transforms.Compose([transforms.ToPILImage(),transforms.Resize((224, 224)),transforms.ToTensor()]),
    #   is_classification=True),
    # 'gi4e_raw_eyes': ds.ImageDataset(
    #   './datasets/gi4e_raw_eyes',
    #   transform=transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor()]),
    #   file_extension='png'),
    'gi4e_detected_eyes': ds.ImageDataset(
      './datasets/gi4e_eyes/20250521_180915',
      transform=transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor()]),
      file_extension='png'),
  }
  
  for name, dataset in datasets.items():
      print(f'Running {name} dataset')
      # do the train
      doTheTrain(dataset, model)
      print(f'Finished {name} dataset')
      print('----------------------------------')
