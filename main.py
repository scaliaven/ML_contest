import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
from torchvision.io.image import read_file
from torchvision.transforms.functional import to_pil_image
import torch
import torchvision
import torchvision.transforms as transforms
print(torch.version.cuda)
print(device)



transform = transforms.Compose(
    [
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    ])