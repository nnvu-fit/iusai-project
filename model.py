## Create model that combine yolo and resnet
## use yolo to detect object and use resnet to classify object

import torch
import torch.nn as nn
from torchvision import models

from yolov5 import Yolo
from utils import *

class YoloResnet(nn.Module):
    def __init__(self, num_classes, anchors, num_anchors, pretrained=True):
        super(YoloResnet, self).__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.num_anchors = num_anchors
        self.yolo = Yolo(num_classes, anchors, num_anchors)
        self.resnet = models.resnet18(pretrained=pretrained)
        self.resnet.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.yolo(x)
        x = self.resnet(x)
        return x

    def load_weights(self, path):
        self.yolo.load_weights(path)
        self.resnet.load_state_dict(torch.load(path))

    def save_weights(self, path):
        torch.save(self.resnet.state_dict(), path)

    def predict(self, image, device, inp_dim=416, anchors=None):
        """
        Args:
            image: image to predict
            device: device to run model
            inp_dim: input dimension
            anchors: anchors to use
        Returns:
            output: output of model
        """
        if anchors is None:
            anchors = self.anchors
        image = prep_image(image, inp_dim)
        image = image.to(device)
        output = self.forward(image)
        output = write_results(output, confidence=0.5, num_classes=self.num_classes, nms_conf=0.4)
        return output

    def predict_batch(self, images, device, inp_dim=416, anchors=None):
        """
        Args:
            images: images to predict
            device: device to run model
            inp_dim: input dimension
            anchors: anchors to use
        Returns:
            output: output of model
        """
        if anchors is None:
            anchors = self.anchors
        images = prep_image(images, inp_dim)
        images = images.to(device)
        output = self.forward(images)
        output = write_results(output, confidence=0.5, num_classes=self.num_classes, nms_conf=0.4)
        return output
    