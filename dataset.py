import glob
import json
import os
import re
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

import numpy as np
import pandas as pd
import cv2
from PIL import Image
import os


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
    self.transform = transform
    self.labelsDir = os.path.join(data_path, 'labels')
    self.imagesDir = os.path.join(data_path, 'images')
    # load label files from labelsDir
    self.labelFiles = [label for label in os.listdir(
        self.labelsDir) if label.endswith('.txt')]
    # filter the files that match the regex pattern '^d+_w+.txt$'
    self.labelFiles = [label for label in self.labelFiles if re.match(
        r'^\d+_image_labels.txt$', label)]
    # read the annotation from the label files
    self.data = []
    for label in self.labelFiles:
      self.read_annotations(label)
    # shuffle data
    random.shuffle(self.data)

  def __getitem__(self, index):
    image, target = self.get_image(index)

    if self.transform:
      image = self.transform(image)

    return image, target

  def __len__(self):
    return len(self.data)

  def get_image(self, index):
    image_name, target = self.data[index]
    # get the image path from the imagesDir
    image_path = os.path.join(self.imagesDir, image_name)
    # read the image from the image path
    image = cv2.imread(image_path)
    # convert the image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image, target

  def read_annotations(self, label):
    # get the label path from the labelsDir
    label_path = os.path.join(self.labelsDir, label)
    # read the annotations from the label file
    with open(label_path, 'r') as f:
      lines = f.readlines()
    # each line contains the image name and 6 points of the eyes in the image
    for line in lines:
      image_id = len(self.data)
      line = line.strip().split()
      # get the image name and the 6 points of the eyes
      image_name = line[0]
      eyes = map(float, line[1:])
      # convert the eyes to the bounding boxes
      is_x, x, y = True, [], []
      for eye in eyes:
        if is_x:
          x.append(eye)
        else:
          y.append(eye)
        is_x = not is_x
      # create the bounding boxes for the eyes
      left_eye_center = (x[1], y[1])
      right_eye_center = (x[4], y[4])
      eye_width = eye_height = self.offset

      left_eye_box = [left_eye_center[0] - eye_width, left_eye_center[1] - eye_height,
                      left_eye_center[0] + eye_width, left_eye_center[1] + eye_height]
      right_eye_box = [right_eye_center[0] - eye_width, right_eye_center[1] - eye_height,
                       right_eye_center[0] + eye_width, right_eye_center[1] + eye_height]
      boxes = [left_eye_box, right_eye_box]

      # identify the labels for the eyes
      labels = [1, 2]  # 1: left eye, 2: right eye

      # identify the target for the eyes
      target = {}
      target['boxes'] = torch.tensor(boxes, dtype=torch.float32)
      target['labels'] = torch.tensor(labels, dtype=torch.int64)
      target['image_id'] = torch.tensor([image_id])
      target['area'] = torch.tensor([])
      target['iscrowd'] = torch.tensor([labels], dtype=torch.int64)

      # get user number from the image name
      user_number = int(re.search(r'\d+', image_name).group())
      # add the user number to the target
      target['user_number'] = torch.tensor([user_number])

      # push the image and the target to the data
      self.data.append((image_name, target))


class Gi4eEyesDataset(Dataset):
  def __init__(self, data_path, transform=None):
    self.data = glob.glob(data_path + '/*/*.png')

    # get all the left eye images
    self.left_eye_images = [image for image in self.data if 'left' in image]
    # get all the right eye images
    self.right_eye_images = [image for image in self.data if 'right' in image]

    # combine the left and right eye images by concatenating them on the same prefix
    self.data = []
    for left_eye in self.left_eye_images:
      right_eye = left_eye.replace('left', 'right')
      if right_eye in self.right_eye_images:
        self.data.append((left_eye, right_eye))
    # do the same for the right eye images that do not have a corresponding left eye image
    for right_eye in self.right_eye_images:
      left_eye = right_eye.replace('right', 'left')
      if (left_eye not in self.left_eye_images) and ((left_eye, right_eye) not in self.data):
        self.data.append((left_eye, right_eye))

    # suffle data
    random.shuffle(self.data)

    if transform:
      self.transform = transform
    else:
      self.transform = T.Compose([T.ToTensor()])

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    # get the left and right eye images
    left_eye, right_eye = self.data[index]

    # get the label from the left eye image
    label = left_eye.split(os.sep)[-2]
    # read the left and right eye images
    left_eye = Image.open(left_eye)
    right_eye = Image.open(right_eye)

    # combine the left and right eye images
    composed_eye = Image.new(
        'RGB', (left_eye.width + right_eye.width, left_eye.height))
    composed_eye.paste(left_eye, (0, 0))
    composed_eye.paste(right_eye, (left_eye.width, 0))

    # apply the transform to the composed eye image
    if self.transform:
      composed_eye = self.transform(composed_eye)

    return composed_eye, torch.tensor(int(label), dtype=torch.long)

  def get_image(self, index):
    left_eye, right_eye = self.data[index]
    left_eye = cv2.imread(left_eye)
    right_eye = cv2.imread(right_eye)

    composed_eye = self.compose_eyes(left_eye, right_eye)
    return composed_eye

  def label(self, index):
    left_eye, right_eye = self.data[index]
    label_str = left_eye.split(os.sep)[-2]
    label_index = int(re.search(r'\d+', label_str).group())
    return label_index

  def labels(self):
    return sorted(set([left_eye.split(os.sep)[-2] for left_eye, right_eye in self.data]))

  def compose_eyes(self, left_eye, right_eye):
    # resize the left and right eye images to the same size
    left_eye = cv2.resize(left_eye, (64, 64))
    right_eye = cv2.resize(right_eye, (64, 64))
    # combine the left and right eye images
    composed_eye = cv2.hconcat([right_eye, left_eye])
    return composed_eye


class YoutubeFacesWithFacialKeypoints(Dataset):
  # define which points need to be connected with a line
  jawPoints = [0, 17]
  rigthEyebrowPoints = [17, 22]
  leftEyebrowPoints = [22, 27]
  noseRidgePoints = [27, 31]
  noseBasePoints = [31, 36]
  rightEyePoints = [36, 42]
  leftEyePoints = [42, 48]
  outerMouthPoints = [48, 60]
  innerMouthPoints = [60, 68]

  def __init__(self, data_path, is_classification=True, transform=None):
    super().__init__()
    self.data_path = data_path
    self.transform = transform
    self.is_classification = is_classification

    # get all files in the data_path
    file_paths = glob.glob(data_path + '/**', recursive=True)

    label_files = [file for file in file_paths if file.endswith('.json')]
    data_paths = [file for file in file_paths if file.endswith('.jpg')]

    # number of label files, number of data paths
    print('number of label files: ', len(label_files))
    print('number of data paths: ', len(data_paths))
    # check if the label files and data paths are the same
    self.data = []
    
    annotations = []
    label_file_count = len(label_files)
    for index, label_file in enumerate(label_files[:10]):
      print(index+1, '/', label_file_count, '- label_file: ', label_file)
      # read the labels from the label file
      with open(label_file, 'r') as f:
        annotations.extend(json.load(f))
    # number of labels
    print('number of annotations: ', len(annotations))
    # ------------------------------------------------------------------------------------- #
    print('-' * 50)
    # read the annotations from the label files
    # check if the labels and data paths are the same
    for index, annotation in enumerate(annotations):
      if index % 1000 == 0:
        print('annotation: ', index+1, '/', len(annotations), ' - ', annotation['image_id'], ' - ', annotation['label'])
      self.read_annotation(annotation, data_paths)
    # suffle data
    random.shuffle(self.data)
    # check if the labels and data paths are the same
    self.labels = [annotation['label'] for annotation in annotations]
    self.labels = list(set(self.labels))
    self.labels.sort()
    print('number of labels: ', len(self.labels))
    
  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    # get the image path and target from the data
    target = self.data[index][1]

    # read the image from the image path
    image = self.get_image(index)
    image = Image.fromarray(image)

    if self.transform:
      image = self.transform(image)

    # convert target to tensor
    label = target['label']
    target = {
      'image_id': torch.tensor(self.labels.index(label), dtype=torch.long),
      'bounding_box': torch.tensor(target['bounding_box'], dtype=torch.float32),
      'landmarks_2d': torch.tensor(target['landmarks_2d'], dtype=torch.float32),
      'landmarks_3d': torch.tensor(target['landmarks_3d'], dtype=torch.float32),
      'label': torch.tensor(self.labels.index(label), dtype=torch.long)
    }

    return image, target['label'] if self.is_classification else target
  
  def get_image(self, index):
    # get the image path and target from the data
    image_path = self.data[index][0]
    # read the image from the image path
    image = cv2.imread(image_path)
    
    return image
  
  def read_annotation(self, annotation, data_paths):
    # get image_id, bounding_box, landmarks_2d, landmarks_3d from label
    image_id = annotation['image_id']
    bounding_box = annotation['bounding_box']
    landmarks_2d = annotation['landmarks_2d']
    landmarks_3d = annotation['landmarks_3d']
    # get the image path from the data_paths
    image_path = [path for path in data_paths if image_id in path]
    if len(image_path) == 0:
      print('image_path not found: ', image_id)
      return
    image_path = image_path[0]

    # create the target for the image
    target = {}
    target['image_id'] = image_id
    target['bounding_box'] = bounding_box
    target['landmarks_2d'] = landmarks_2d
    target['landmarks_3d'] = landmarks_3d
    target['label'] = annotation['label']

    # add the image path and target to the data
    self.data.append((image_path, target))

  def labels(self):
    return self.labels

