from data_util import test_CustomImageDataset
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image
import torch
import os
import pandas as pd

class test_CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return sum(1 for file in Path(self.img_dir).iterdir() if file.suffix == '.png')

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, f"{idx}.png")
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

transform = transforms.Compose(
    [
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    ])

def prediction(net, transform):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    net.eval()
    test_data = test_CustomImageDataset("/scratch/hh3043/ML_contest/dataset/test_img", transform=transform)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False, num_workers=3)
    predicted_labels = []

    with torch.no_grad():
        for data in test_loader:
            images = data
            images = images.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            predicted_labels.extend(predicted.cpu().numpy())
            
    output = pd.DataFrame({
    "id": [i for i in range(len(test_data))],
    "category": predicted_labels
    })

    output.to_csv('/scratch/hh3043/ML_contest/my_submission.csv', index=False)

