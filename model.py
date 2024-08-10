## Create model that combine yolo and resnet
## use yolo to detect object and use resnet to classify object

import torch
import torch.nn as nn
from torchvision import models
    
class VGGFace(nn.Module):
    def __init__(self, num_classes):
        super(VGGFace, self).__init__()
        self.num_classes = num_classes
        self.features = models.vgg16(pretrained=True).features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            # Layer 1
            nn.Conv2d(3, 64, kernel_size=(3,3), stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,3), stride=2),
            nn.SyncBatchNorm(),
            # Layer 2
            nn.Conv2d(64, kernel_size=(3,3)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,3), stride=2),
            nn.SyncBatchNorm(),
            # Layer 3
            nn.Conv2d(64, 96, kernel_size=(3,3)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,3), stride=2),
            nn.SyncBatchNorm(),
            # Layer 4
            nn.Conv2d(96, 32, kernel_size=(3,3)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            nn.SyncBatchNorm(),
            nn.Dropout(0.2),
            nn.Flatten(),
            # Layer 5
            nn.Linear(32*6*6, 128),
            nn.ReLU(inplace=True),
            # Layer 6
            nn.Linear(128, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        
        return x