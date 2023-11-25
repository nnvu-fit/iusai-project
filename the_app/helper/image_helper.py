import requests
import io
import torch
from PIL import Image
from torchvision import transforms

# Download image from url and return as bytes
def download_image(url: str) -> bytes:
    """Download image from url and return as bytes"""
    # return null if url is not valid
    if url == "" or url is None or url == None:
        return None
    # return null if url is not an url
    if not url.startswith("http"):
        return None
    response = requests.get(url)
    return response.content

# Load image from bytes and return as tensor
def load_image(image: bytes) -> torch.Tensor:
    """Load image from bytes and return as tensor"""
    # return null if image is not valid
    if image == None:
        return None
    # return null if image is not an instance of bytes
    if not isinstance(image, bytes):
        return None
    # load image
    image_tensor = Image.open(io.BytesIO(image))
    image_tensor = image_transforms(image_tensor)
    image_tensor = image_tensor.unsqueeze(0)
    return image_tensor

# transform image
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])