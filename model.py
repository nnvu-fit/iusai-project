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
        self.conv1 = nn.Conv2d(3, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.conv4 = nn.Conv2d(64, 96, 3, 1)
        self.conv5 = nn.Conv2d(96, 32, 3, 1)
        self.pool = nn.MaxPool2d(2, 2)
        # Fully connected layer
        
        self.fc1 = nn.Linear(800, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # apply transpose to change the shape of the input
        # x = torch.transpose(x, 1, 3)

        # Layer 1
        x = self.conv1(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.pool(x)
        # x = nn.BatchNorm2d(64)(x)
        # Layer 2
        x = self.conv2(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.pool(x)
        # x = nn.BatchNorm2d(64)(x)
        # Layer 3
        x = self.conv3(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.pool(x)
        # x = nn.BatchNorm2d(96)(x)
        # Layer 4
        x = self.conv4(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.pool(x)
        # x = nn.BatchNorm2d(32)(x)
        # Layer 5
        x = self.conv5(x)
        x = nn.ReLU(inplace=True)(x)
        x = nn.MaxPool2d(kernel_size=(2,2), stride=2)(x)
        # x = nn.BatchNorm2d(32)(x)
        x = nn.Dropout(0.2)(x)
        x = nn.Flatten()(x)
        # Layer 6
        x = self.fc1(x)
        x = nn.ReLU(inplace=True)(x)
        # Layer 7
        x = self.fc2(x)
        x = nn.Softmax(dim=1)(x)
        
        return x