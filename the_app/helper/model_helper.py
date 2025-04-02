import io
import requests
import helper.image_helper as ih
import torch
import torch.nn as nn
import torchvision.models as models


# model_types
model_types = ['ResNet18', 'ResNet34', 'DenseNet121']

# model_type_url_dict
model_type_url_dict = {
    'ResNet18': 'https://github.com/nnvu-fit/iusai-project/releases/download/v0.1/model_resnes18.pth',
    'ResNet34': 'https://github.com/nnvu-fit/iusai-project/releases/download/v0.1/model_resnes34.small.pth',
    'DenseNet121': 'https://github.com/nnvu-fit/iusai-project/releases/download/v0.1/model_densenet121.small.pth'
}

# Download a file from a GitHub release and return the content


def download_file_from_github_release(url: str) -> bytes:
    """Download a file from a GitHub release and return the content"""
    # return null if url is not valid
    if url == "" or url is None or url == None:
        return None
    # return null if url is not an url
    if not url.startswith("http"):
        return None
    # download file
    response = requests.get(url)
    return response.content

# Classify image and return the result
def classify_image(image: bytes, model: nn.Module) -> str:
    """Classify image and return the result"""
    # return null if image is not valid
    if image == None:
        return None
    # return null if model is not valid
    if model == None:
        return None
    # return null if model is not an instance of nn.Module
    if not isinstance(model, nn.Module):
        return None
    
    # load image
    image_tensor = ih.load_image(image)

    # classify image
    model.eval()
    with torch.no_grad():
        outputs = model(image_tensor)
        _, preds = torch.max(outputs, 1)
        return preds.item()    

# get model from model type
def get_model(model_type: ['ResNet18', 'ResNet34', 'DenseNet121']): # type: ignore
    model_weight_url = ''
    if model_type in model_type_url_dict:
        model_weight_url = model_type_url_dict[model_type]
    else:
        return None
    
    model_weight_data = download_file_from_github_release(model_weight_url)
    if model_weight_data == None:
        return None
    
    parameters = torch.load(io.BytesIO(model_weight_data), map_location=torch.device('cpu'))
    if "optimizer" in parameters:
        parameters = parameters["model"]

    model = None
    if model_type == 'ResNet18':
        model = models.resnet18(weights=None)
    elif model_type == 'ResNet34':
        model = models.resnet34(weights=None)
    elif model_type == 'DenseNet121':
        model = models.densenet121(weights=None)

    model.load_state_dict(parameters)
    return model
