# Using torchvision to create a dataset
from torchvision import transforms
import torch
from torch.utils.data import random_split, DataLoader
import torchvision
from torchvision.models import ResNet50_Weights
from model import CNN, VGGFace

import cv2

#  import self library
from train import ClassifierTrainer as Trainer
import data_set as ds


def doTheTrain(dataset, model):
    # define batch_size
    batch_size = 32

    # init train val test ds
    train_val_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_val_size
    train_val_ds, test_ds = random_split(dataset, [train_val_size, test_size])

    # define optimizer using Adam and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()

    trainer = Trainer(model, optimizer, loss_fn, random_seed_value=86)
    print('device: ', trainer.device)
    avg_lost = trainer.cross_validate(train_val_ds, epochs=5)
    print('avg_lost: ', avg_lost)

    # score model
    test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)
    model_scored = trainer.score(test_dataloader)
    print('model_scored: ', model_scored)


if __name__ == "__main__":

    # init models
    models = [torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)]

    # init transform for dataset
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()])

    # # create dataset for Gi4eEyes
    # images_path = './datasets/faster-rcnn/gi4e_eyes/20250307_224145'
    # dataset = ds.Gi4eEyesDataset(images_path, transform=transform)

    # create dataset for YouTubeFacesWithFacialKeypoints
    images_path = './datasets/YouTubeFacesWithFacialKeypoints'
    dataset = ds.YoutubeFacesWithFacialKeypoints(
        images_path, transform=transform)

    # # train model
    # for model in models:
    #     doTheTrain(dataset, model)
