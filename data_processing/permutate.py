import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from multiprocess import Pool
from functools import partial
import os
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from PIL import Image
from tqdm import tqdm
from sklearn.utils import shuffle

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, idex_list, transform=None, mask = None, balance = False):
        with open(annotations_file, 'r') as file:
            content = file.read()
            lines = content.strip("\n").split('\n')        
        self.img_labels_init = []
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
            img_path = os.path.join(self.img_dir, f"{self.idex_list[idx]}.png")
            image = Image.open(img_path).convert("RGB")
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

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        label = int(self.img_labels[idx])
        image = self.image[idx]
        return image, label


def main(permute):
    with open("/scratch/hh3043/ML_contest/dataset/train_label.txt", 'r') as file:
        content = file.read()
        num = len(content.strip("\n").split('\n'))

    train_data_idx, test_data_idx = train_test_split(list(range(num)), test_size=0.2, random_state=42)


    transform = transforms.Compose(
        [
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
        ])


    total_train = CustomImageDataset("/scratch/hh3043/ML_contest/dataset/train_label.txt", "/scratch/hh3043/ML_contest/dataset/train_img", train_data_idx, transform, None, True)
    #total_test = CustomImageDataset("/scratch/hh3043/ML_contest/dataset/train_label.txt", "/scratch/hh3043/ML_contest/dataset/train_img", test_data_idx, transform, None, False)
    number = list(range(32))
    for i in range(10, 20):
        mask = shuffle(number, random_state=i)
        train_data = CustomImageDataset("/scratch/hh3043/ML_contest/dataset/train_label.txt", "/scratch/hh3043/ML_contest/dataset/train_img", train_data_idx, transform, mask, True)
        total_train = ConcatDataset([total_train, train_data])
        
    width = 32
    torch.save(total_train, f'/scratch/hh3043/ML_contest/dataset/total_train_{width}_1.pt') 
    #torch.save(total_test, f'/scratch/hh3043/ML_contest/dataset/total_test_{width}.pt') 
    print("finished processing")


if __name__ == "__main__":
    permute = 10
    main(permute)
