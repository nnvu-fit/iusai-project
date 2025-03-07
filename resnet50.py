## Using torchvision to create a dataset
from torchvision import transforms
import torch
from torch.utils.data import random_split, DataLoader
from model import CNN, VGGFace

#  import self library
from train import ClassifierTrainer as Trainer
import data_set as ds

def doTheTrain(images_path, model):

    # get devices
    transform = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    imageDataset = ds.Gi4eEyesDataset(images_path, transform=transform)

    ## define batch_size
    batch_size = 32

    # init train val test ds
    train_val_size = int(0.9 * len(imageDataset))
    test_size = len(imageDataset) - train_val_size
    train_val_ds, test_ds = random_split(imageDataset, [train_val_size, test_size])

    ## define optimizer using Adam and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()

    trainer = Trainer(model, optimizer, loss_fn, random_seed_value=86)
    print('device: ', trainer.device)
    avg_lost = trainer.cross_validate(train_val_ds, epochs=5)
    print('avg_lost: ', avg_lost)

    ## score model
    test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=True)
    model_scored = trainer.score(test_dataloader)
    print('model_scored: ', model_scored)

if __name__ == "__main__":
    # init images path for dataset
    images_path = 'subjects-small/'

    models = [
        # torchvision.models.resnet18(pretrained=True),
        CNN(24)
    ]
    for model in models:
        doTheTrain(images_path, model)
