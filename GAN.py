import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torchvision.utils as vutils
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
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
import torch.nn.functional as F
from torchvision import datasets
from torch.autograd import Variable



######## set up ##########
save = False
patience = 400
predict = False
continue_train = False
epochs = 30
name = "densenet"
PATH = "/scratch/hh3043/ML_contest/separate/checkpoint_GAN.pth"
best_accu_val = 0
#########################


transform = transforms.Compose(
    [
    transforms.Normalize((0.5, ), (0.5, )) 
    ])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.version.cuda)
print(device)
cudnn.benchmark = True


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


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def save_checkpoint(model, optimizer, PATH):

    checkpoint = {'model':  type(model).__name__,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict()}
    torch.save(checkpoint, PATH)


def testing(net, testloader, criterion):
    net.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images, labels)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()

    accuracy = (correct / total) * 100
    val_loss =running_loss/len(testloader)

    print(f'validation Loss:{val_loss:.3f}, accuracy: {accuracy:.3f}%')
    return val_loss, accuracy


def collate_fn(batch):
    image = torch.stack([x[0] for x in batch])
    label = torch.tensor([x[1] for x in batch])
    return image, label
 

if predict:

    model.load_state_dict(torch.load(PATH)['state_dict'])
    model = model.to(device)
    prediction(model, transform)
    print("Finish saving")

else:
    
    if save:
        train_data = CustomImageDataset("/scratch/hh3043/ML_contest/dataset/train_label.txt", "/scratch/hh3043/ML_contest/separate/train_img", transform = transform, mask = None, balance = True)
        val_data = CustomImageDataset("/scratch/hh3043/ML_contest/dataset/train_label.txt", "/scratch/hh3043/ML_contest/separate/val_img", transform = transform, mask = None, balance = False)
        torch.save(train_data,'/scratch/hh3043/ML_contest/separate/train_64_gray.pt')
        torch.save(val_data,'/scratch/hh3043/ML_contest/separate/val_64_gray.pt')
        trainloader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=3)
        valloader = DataLoader(val_data, batch_size=16, shuffle=False, num_workers=3)

    else:
        train_data = torch.load('/scratch/hh3043/ML_contest/separate/train_64_gray.pt')
        val_data = torch.load('/scratch/hh3043/ML_contest/separate/val_64_gray.pt')
        trainloader = DataLoader(train_data, batch_size=16, collate_fn=collate_fn, shuffle=True, num_workers=3)
        valloader = DataLoader(val_data, batch_size=16, collate_fn=collate_fn, shuffle=False, num_workers=3)

    if continue_train:
        model.load_state_dict(torch.load(PATH)['state_dict'])

    # model.to(device)



img_shape = (1, 128, 128)

n_classes = 4
latent_dim = 100
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(n_classes, n_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim + n_classes, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(n_classes, n_classes)

        self.model = nn.Sequential(
            nn.Linear(n_classes + int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 4),
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity



adversarial_loss = torch.nn.MSELoss()

generator = Generator()
discriminator = Discriminator()


generator.to(device)
discriminator.to(device)
adversarial_loss.to(device)

# Optimizers
max_lr = 2e-3
grad_clip = 0.1
optimizer_D = optim.SGD(discriminator.parameters(), lr = max_lr, weight_decay = 1.0e-3, momentum = 0.9) 
optimizer_G = optim.SGD(generator.parameters(), lr = max_lr, weight_decay = 1.0e-3, momentum = 0.9)
scheduler_D = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_D, T_0 = epochs*len(trainloader))
scheduler_G = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer_G, T_0 = epochs*len(trainloader))

criterion = nn.CrossEntropyLoss().to(device)

FloatTensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor


# ----------
#  Training
# ----------

for epoch in range(epochs):
    for i, (imgs, labels) in enumerate(tqdm(trainloader), 0):

        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, latent_dim))))
        gen_labels = Variable(LongTensor(np.random.randint(0, n_classes, batch_size)))

        # Generate a batch of images
        gen_imgs = generator(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_imgs, gen_labels)
        _, predicted = validity.max(1)
        g_loss = adversarial_loss(predicted, valid)

        g_loss.backward()
        optimizer_G.step()
        scheduler_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        validity_real = discriminator(real_imgs, labels)
        d_real_loss = criterion(validity_real, valid)

        # Loss for fake images
        validity_fake = discriminator(gen_imgs.detach(), gen_labels)
        d_fake_loss = criterion(validity_fake, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()
        scheduler_D.step()
    val_loss, val_accu = testing(discriminator, valloader, criterion)

