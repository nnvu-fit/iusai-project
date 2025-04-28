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
    batch_size = 128

    # init train val test ds
    train_val_size = int(0.9 * len(dataset))
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
    
    # create dataset for Gi4e
    images_path = './datasets/gi4e'
    transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize((224, 224)),transforms.ToTensor()])
    dataset = ds.Gi4eDataset(images_path, transform=transform, is_classification=True)
    first_image, first_label = dataset[0]
    print('first_image: ', first_image.shape)
    print('first_label: ', first_label)
    print('dataset len: ', len(dataset))
    # train the model
    doTheTrain(dataset, model)

    # create dataset for Gi4eEyes
    transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor()])
    images_path = './datasets/gi4e_eyes/20250412_185051'
    dataset = ds.ImageDataset(images_path, transform=transform, file_extension='png')
    print('dataset labels: ', dataset.labels())
    first_image, first_label = dataset[0]
    print('first_image: ', first_image.shape)
    print('first_label: ', first_label)
    print('dataset len: ', len(dataset))
    doTheTrain(dataset, model)

    # # create dataset for YouTubeFacesWithFacialKeypoints
    # images_path = './datasets/YouTubeFacesWithFacialKeypoints'
    # dataset = ds.YoutubeFacesWithFacialKeypoints(
    #     images_path, transform=transform)

    # first_image, first_label = dataset[0]
    # print('first_image: ', first_image.shape)
    # print('dataset len: ', len(dataset))

    # doTheTrain(dataset, model)
