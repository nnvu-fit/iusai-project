import glob
import json
import os
import re
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision


class ImageDataset(Dataset):
    # Initialize your data from data_path using glob
    def __init__(self, data_path, transform=None):
        self.data = glob.glob(data_path + '/*/*.jpg')
        # suffle data
        random.shuffle(self.data)
        self.transform = transform

    def __getitem__(self, index):
        path_x = self.data[index]
        x = Image.open(path_x)
        if self.transform:
            x = self.transform(x)
        # [0, 1, 2]: index == -2 => 1;
        label = self.label(index)
        return x, torch.tensor(int(label), dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def get_image(self, index):
        path_x = self.data[index]
        x = Image.open(path_x)
        y = self.label(index)
        return x, int(y)

    def label(self, index):
        path_x = self.data[index]
        label_str = path_x.split(os.sep)[-2]
        label_index = int(re.search(r'\d+', label_str).group())
        return label_index

    def labels(self):
        return sorted(set([path_x.split(os.sep)[-2] for path_x in self.data]))


class EyesDataset(Dataset):
    class Category:
        def __init__(self, id, name, supercategory):
            self.id = id
            self.name = name
            self.supercategory = supercategory

    class Annotation:
        def __init__(self, id, image_id, category_id, bbox, area, segmentation):
            self.id = id
            self.image_id = image_id
            self.category_id = category_id
            self.bbox = bbox
            self.area = area
            self.segmentation = segmentation

    class Image:
        def __init__(self, id, file_name, width, height):
            self.id = id
            self.file_name = file_name
            self.width = width
            self.height = height

    class JsonData:
        def __init__(self, categories, images, annotations):
            self.categories = categories
            self.images = images
            self.annotations = annotations

    def __init__(self, data_json_path, transform=None):
        self.json_data = self.read_image_from_path(data_json_path)
        self.data = [self.read_image(x) for x in self.json_data.images]

        # suffle data
        random.shuffle(self.data)
        self.transform = transform

    def __getitem__(self, index):
        x, image = self.data[index]
        if self.transform:
            x = self.transform(x)

        # get annotation for image
        annotations = [
            annotation for annotation in self.json_data.annotations if annotation.image_id == image.id]
        # get category for annotation
        categories = [
            category for category in self.json_data.categories if category.id == annotations[0].category_id]
        # get label for category
        label = categories[0].name
        return x, image

    def __len__(self):
        return len(self.data)

    def read_image_from_path(self, path):
        # Read data from path and store it in json_data
        with open(path, 'r') as f:
            json_data = json.load(f)
        # Load categories from json_data
        categories = []
        for category in json_data['categories']:
            categories.append(self.Category(
                category['id'], category['name'], category['supercategory']))
        # Load images from json_data
        images = []
        for image in json_data['images']:
            images.append(self.Image(
                image['id'], image['file_name'], image['width'], image['height']))
        # Load annotations from json_data
        annotations = []
        for annotation in json_data['annotations']:
            annotations.append(self.Annotation(
                annotation['id'], annotation['image_id'], annotation['category_id'], annotation['bbox'], annotation['area'], annotation['segmentation']))

        return self.JsonData(categories, images, annotations)

    def read_image(self, image_path, image: Image):
        # read image from image.file_name
        image_path = os.path.join(os.path.dirname(image_path), image.file_name)
        # read image from image_path
        x = Image.open(image_path)
        return x, image


class Gi4eDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.offset = 28
        self.labelsDir = os.path.join(data_path, 'labels')
        self.imagesDir = os.path.join(data_path, 'images')
        # load label files from labelsDir
        self.labelFiles = [label for label in os.listdir(
            self.labelsDir) if label.endswith('.txt')]
        # filter the files that match the regex pattern '^d+_w+.txt$'
        self.labelFiles = [label for label in self.labelFiles if re.match(
            r'^\d+_image_labels.txt$', label)]
        # read the annotation from the label files
        self.data = [self.read_annotation(label) for label in self.labelFiles]
        # suffle data
        random.shuffle(self.data)
        self.transform = transform

    def __getitem__(self, index):
        x, target = self.data[index]
        if self.transform:
            x = self.transform(x)
            # transform the target['boxes'] to match the transformed image
        else:
            x = torchvision.transforms.ToTensor()(x)
            
        return x, target

    def __len__(self):
        return len(self.data)

    def get_image(self, index):
        return self.data[index][0]

    def read_annotation(self, label):
        # read the label file
        label_path = os.path.join(self.labelsDir, label)
        with open(label_path, 'r') as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            values = line.split()
            file_name = values[0]
            isX = True
            x = []
            y = []
            for value in values[1:]:
                # parse the values to float
                value = float(value)
                # append the values to x or y
                if isX:
                    x.append(value)
                else:
                    y.append(value)
                # switch between x and y
                isX = not isX

        x_left_min, y_left_min = (min(x[0], x[2]), min(y[0], y[2] - self.offset))
        x_left_max, y_left_max = (max(x[0], x[2]), max(y[0], y[2] + self.offset))

        x_right_min, y_right_min = (min(x[3], x[5]), min(y[3], y[5]) - self.offset)
        x_right_max, y_right_max = (max(x[3], x[5]), max(y[3], y[5]) + self.offset)

        boxes = torch.stack([torch.tensor([x_left_min, y_left_min, x_left_max, y_left_max], dtype=torch.float32),
                             torch.tensor([x_right_min, y_right_min, x_right_max, y_right_max], dtype=torch.float32)])
        
        target = {'boxes': boxes, 'labels': torch.tensor([1, 1], dtype=torch.int64)}
        return Image.open(os.path.join(self.imagesDir, file_name)), target

