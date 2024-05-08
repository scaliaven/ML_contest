from data_util import test_CustomImageDataset
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from pathlib import Path
from PIL import Image
import torch
import os
import pandas as pd
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset
from model_zoo import resnet18, Transformer, residual_attention_network
from predict import prediction
import torch.nn as nn
import random
from tqdm import tqdm

transform = transforms.Compose([transforms.Normalize((0.5, ), (0.5, )) ])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.version.cuda)
print(device)

resnet = resnet18.resnet18()
resnet.load_state_dict(torch.load("/scratch/hh3043/ML_contest/checkpoint_res_aug_split.pth")['state_dict'])


atten_92 = residual_attention_network.ResidualAttentionModel_92_32input_update()
atten_92.load_state_dict(torch.load("/scratch/hh3043/ML_contest/checkpoint_atten_aug_split.pth")['state_dict'])

resNext50 = torchvision.models.resnext50_32x4d(weights = None, num_classes = 4)
resNext50.conv1 = nn.Conv2d(1, resNext50.conv1.weight.shape[0], 3, 1, 1, bias = False)
resNext50.maxpool = nn.MaxPool2d(kernel_size = 1, stride = 1, padding = 0)
resNext50.load_state_dict(torch.load("/scratch/hh3043/ML_contest/checkpoint_next_aug_split.pth")['state_dict'])


Densenet_201 = torchvision.models.densenet201(weights = None, num_classes = 4)
Densenet_201.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
Densenet_201.features[3] = nn.MaxPool2d(kernel_size = 1, stride = 1, padding = 0)
Densenet_201.load_state_dict(torch.load("/scratch/hh3043/ML_contest/checkpoint_dense_aug_split.pth")['state_dict'])


model_list = [resnet, atten_92, resNext50, Densenet_201]

test_data = test_CustomImageDataset("/scratch/hh3043/ML_contest/separate/test_img", transform=transform)
test_loader = DataLoader(test_data, batch_size=16, shuffle=False, num_workers=3)
predicted_labels = []

for data in tqdm(test_loader):
    outputs = torch.zeros((data.shape[0], 4)).to(device)
    for model in model_list:
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            images = data
            images = images.to(device)
            output = model(images)
            outputs += output
    _, predicted = torch.max(outputs.data, 1)
    predicted_labels.extend(predicted.cpu().numpy())
            
output = pd.DataFrame({
"id": [i for i in range(len(test_data))],
"category": predicted_labels
})

output.to_csv('/scratch/hh3043/ML_contest/my_submission.csv', index=False)
print("finished saving csv file")