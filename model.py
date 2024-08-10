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
    
