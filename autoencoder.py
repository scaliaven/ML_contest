import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset
from model_zoo import resnet18, Transformer, shake_resnet, shake_resnext, residual_attention_network
from data_util import CustomImageDataset
import torch.optim as optim
from predict import prediction
import torch.nn as nn
from tqdm import tqdm
import torchaudio
import random
import numpy as np


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def collate_fn(batch):
    image = torch.stack([x[0] for x in batch])
    label = torch.tensor([x[1] for x in batch])
    return image, label



######## set up ##########
save = False
patience = 400
predict = False
continue_train = False
epochs = 30
name = "resnet"
PATH = "/scratch/hh3043/ML_contest/checkpoint_next_aug_test_1.pth"
best_accu_val = 0
#########################




if name == "resnet":
    model = resnet18.resnet18()
    # model.conv1 = nn.Conv2d(1, model.conv1.weight.shape[0], 3, 1, 1, bias = False)
    # model.maxpool = nn.MaxPool2d(kernel_size = 1, stride = 1, padding = 0)

elif name == "resnext50":
    model = torchvision.models.resnext50_32x4d(weights = None, num_classes = 4)
    model.conv1 = nn.Conv2d(1, model.conv1.weight.shape[0], 3, 1, 1, bias = False)
    model.maxpool = nn.MaxPool2d(kernel_size = 1, stride = 1, padding = 0)

elif name == "lstm":
    model = Transformer.LSTM()

elif name == "shake_resnet":
    model = shake_resnet.ShakeResNet(26, 64, 4)

elif name == "shake_resnext":
    model = shake_resnext.ShakeResNeXt(50, 64, 4, 4)

elif name == "attention_resnet":
    model = Transformer.attention_resnet18()

elif name == "densenet":
    model = torchvision.models.densenet201()
    model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.features[3] = nn.MaxPool2d(kernel_size = 1, stride = 1, padding = 0)

elif name == "attention_next_56":
    model = residual_attention_network.ResidualAttentionModel_56()
elif name == "attention_next_92":
    model = residual_attention_network.ResidualAttentionModel_92_32input_update()





# Define your autoencoder architecture
import torch.nn.functional as F

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid() # Sigmoid to ensure output values between 0 and 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded




# Hyperparameters
input_size = 128  # Size of input images (e.g., 28x28 for MNIST)
hidden_size = 256  # Size of hidden layers
latent_size = 64  # Size of latent space in the autoencoder
num_classes = 4  # Number of classes in the classification task
batch_size = 16
learning_rate = 0.0001
num_epochs = 40

train_data = torch.load('/scratch/hh3043/ML_contest/separate/train_64_gray.pt')
val_data = torch.load('/scratch/hh3043/ML_contest/separate/val_64_gray.pt')
train_loader = DataLoader(train_data, batch_size=16, collate_fn=collate_fn, shuffle=True, num_workers=3)
val_loader = DataLoader(val_data, batch_size=16, collate_fn=collate_fn, shuffle=False, num_workers=3)


# Initialize your models
autoencoder = Autoencoder().to(device)
classifier = model.to(device)

# Define loss function and optimizer for both models
criterion = nn.CrossEntropyLoss()
autoencoder_optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate)
classifier_optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for images, labels in train_loader:
        # Flatten images for both autoencoder and classifier
        images = images.to(device)
        labels.to(device)
        
        # Train the autoencoder
        autoencoder_optimizer.zero_grad()
        reconstructed_images = autoencoder(images)
        autoencoder_loss = nn.functional.mse_loss(reconstructed_images, images)
        autoencoder_loss.backward()
        autoencoder_optimizer.step()
        
        # Train the classifier using both real and synthetic data
        classifier_optimizer.zero_grad()
        encoded_images = autoencoder.encoder(images)
        classifier_output = classifier(encoded_images)
        classifier_loss = criterion(classifier_output, labels)
        classifier_loss.backward()
        classifier_optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Autoencoder Loss: {autoencoder_loss.item()}, Classifier Loss: {classifier_loss.item()}')

# Evaluate your models on a separate validation set
# Implement validation loop here...

# Generate synthetic data using the trained autoencoder
num_synthetic_samples = 1000
latent_samples = torch.randn(num_synthetic_samples, latent_size)
synthetic_images = autoencoder.decoder(latent_samples)

# Optionally, fine-tune the classifier using the combined dataset
combined_images = torch.cat([train_images, synthetic_images], dim=0)
combined_labels = torch.cat([train_labels, torch.zeros(num_synthetic_samples, dtype=torch.long)], dim=0)
# Fine-tune the classifier using combined_images and combined_labels...

# Save your models if needed
torch.save(autoencoder.state_dict(), '/scratch/hh3043/ML_contest/separate/autoencoder.pth')
torch.save(classifier.state_dict(), '/scratch/hh3043/ML_contest/separate/classifier.pth')
