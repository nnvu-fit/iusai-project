import glob
import os
import re
import random
import torch
from torch.utils.data import Dataset
from PIL import Image

class ImageDataset(Dataset):
    # Initialize your data from data_path using glob
    def __init__(self, data_path, transform=None):
        self.data = glob.glob(data_path + '/*/*.jpg')
        ## suffle data
        random.shuffle(self.data)
        self.transform = transform

    def __getitem__(self, index):
        path_x = self.data[index]
        x = Image.open(path_x)
        if self.transform:
            x = self.transform(x)
        ## [0, 1, 2]: index == -2 => 1;
        label_str = path_x.split(os.sep)[-2]
        label_index = int(re.search(r'\d+', label_str).group())
        return x, torch.tensor(int(label_index), dtype=torch.long)

    def __len__(self):
        return len(self.data)
    
    def get_image(self, index):
        path_x = self.data[index]
        x = Image.open(path_x)
        y = path_x.split(os.sep)[-2]
        return x, int(y)
    
    def labels(self):
        return sorted(set([path_x.split(os.sep)[-2] for path_x in self.data]))
