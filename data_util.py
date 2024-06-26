import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
from torchvision.io.image import read_file
from pathlib import Path
from torch.utils.data import Dataset
import torch


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, mask = None, balance = False):
        with open(annotations_file, 'r') as file:
            content = file.read()
            lines = content.strip("\n").split('\n')        
        self.img_labels_init = []

        idex_list = self.extract_index_from_filenames(img_dir)

        for index in idex_list:
            self.img_labels_init.append(lines[index])
        self.img_dir = img_dir
        self.transform = transform
        self.idex_list = idex_list
        self.image = []
        self.mask = mask
        self.img_labels = []
        for idx in tqdm(range(len(self.idex_list))):
            label = self.img_labels_init[idx]
            img_path = os.path.join(self.img_dir, f"{self.idex_list[idx]}.pt")
            image = torch.load(img_path)[ :, :, 1:129]
            if self.transform:
                image = self.transform(image)
            if self.mask:
                image = image[:, :, self.mask]
            if balance: 
                if label == '0':
                    for i in range(3):
                        self.image.append(image)
                        self.img_labels.append(label)
            self.img_labels.append(label)
            self.image.append(image)

    def extract_index_from_filenames(self, directory):
        index_list = []
        for filename in os.listdir(directory):
            if os.path.isfile(os.path.join(directory, filename)):
                name, extension = os.path.splitext(filename)
                if name.isdigit():
                    index_list.append(int(name))
        return index_list


    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        label = int(self.img_labels[idx])
        image = self.image[idx]
        return image, label



class test_CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return sum(1 for file in Path(self.img_dir).iterdir() if file.suffix == '.pt')

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, f"{idx}.pt")
        # image = Image.open(img_path).convert('RGB')
        image = torch.load(img_path)[ :, :, 1:129]
        if self.transform:
            image = self.transform(image)
        return image


class metaDataset(Dataset):
    def __init__(self, x, y):
        super(metaDataset, self).__init__()
        self._x = x
        self._y = y

    def __len__(self):
        return self._x.shape[0]

    def __getitem__(self, index):
        x = self._x[index]
        y = self._y[index]
        return x, y




class test_metaDataset(Dataset):
    def __init__(self, x):
        super(test_metaDataset, self).__init__()
        self._x = x

    def __len__(self):
        return self._x.shape[0]

    def __getitem__(self, index):
        x = self._x[index]
        return x