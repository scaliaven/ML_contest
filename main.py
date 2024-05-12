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

def set_random_seeds(seed_value=42):
    random.seed(seed_value)         
    np.random.seed(seed_value)     
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.backends.cudnn.deterministic = True

set_random_seeds(seed_value=42)



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(torch.version.cuda)
print(device)

######## set up ##########
save = False
patience = 400
predict = False
continue_train = False
epochs = 20
name = "resnext101"
PATH = "/scratch/hh3043/ML_contest/checkpoint_resnext101_aug_test_1.pth"
best_accu_val = 0

##########################

transform = transforms.Compose(
    [
    transforms.Normalize((0.5, ), (0.5, )) 
    ])


import numpy as np 
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
            outputs = net(images)
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
    model = torchvision.models.densenet201(weights = None, num_classes = 4)
    model.features[0] = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.features[3] = nn.MaxPool2d(kernel_size = 1, stride = 1, padding = 0)

elif name == "attention_next_56":
    model = residual_attention_network.ResidualAttentionModel_56()
elif name == "attention_next_92":
    model = residual_attention_network.ResidualAttentionModel_92_32input_update()
elif name == "resnext101":
    model = torchvision.models.resnext101_32x8d(weights = None, num_classes = 4)
    model.conv1 = nn.Conv2d(1, model.conv1.weight.shape[0], 3, 1, 1, bias = False)
    model.maxpool = nn.MaxPool2d(kernel_size = 1, stride = 1, padding = 0)

if predict:

    model.load_state_dict(torch.load(PATH)['state_dict'])
    model = model.to(device)
    prediction(model, transform)
    print("Finish saving")

else:
    
    if save:
        train_data = CustomImageDataset("/scratch/hh3043/ML_contest/dataset/train_label.txt", "/scratch/hh3043/ML_contest/separate/train_img", transform = transform, mask = None, balance = True)
        # train_data_1 = CustomImageDataset("/scratch/hh3043/ML_contest/dataset/train_label.txt", "/scratch/hh3043/ML_contest/dataset/train_img", transform = transform_1, mask = None, balance = True)
        val_data = CustomImageDataset("/scratch/hh3043/ML_contest/dataset/train_label.txt", "/scratch/hh3043/ML_contest/separate/val_img", transform = transform, mask = None, balance = False)
        torch.save(train_data,'/scratch/hh3043/ML_contest/separate/train_64_gray.pt')
        torch.save(val_data,'/scratch/hh3043/ML_contest/separate/val_64_gray.pt')
        # torch.save(train_data_1,'/scratch/hh3043/ML_contest/dataset/train_64_hflip.pt')
        trainloader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=3)
        valloader = DataLoader(val_data, batch_size=16, shuffle=False, num_workers=3)

    else:
        train_data = torch.load('/scratch/hh3043/ML_contest/separate/train_64_gray.pt')
        # train_data_1 = torch.load('/scratch/hh3043/ML_contest/dataset/train_64_hflip.pt')
        val_data = torch.load('/scratch/hh3043/ML_contest/separate/val_64_gray.pt')
        trainloader = DataLoader(train_data, batch_size=16, collate_fn=collate_fn, shuffle=True, num_workers=3)
        valloader = DataLoader(val_data, batch_size=16, collate_fn=collate_fn, shuffle=False, num_workers=3)

    if continue_train:
        model.load_state_dict(torch.load(PATH)['state_dict'])
    model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    max_lr = 5e-3
    optimizer = optim.SGD(model.parameters(), lr = max_lr, weight_decay = 1.0e-3, momentum = 0.9) 
    grad_clip = 0.1
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = epochs*len(trainloader))
    # sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(trainloader))
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [epochs // 3, epochs * 2 // 3], gamma = 0.1, last_epoch = -1)

    # early stopping accuracy


    best_val = 100
    cnt = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct=0
        total=0
        for i, data in enumerate(tqdm(list(trainloader))):

            inputs, labels = data
            t_mask_1 = torchaudio.transforms.TimeMasking(time_mask_param=6, iid_masks = True)
            freq_mask_1 = torchaudio.transforms.FrequencyMasking(freq_mask_param=6, iid_masks = True)
            t_mask_2 = torchaudio.transforms.TimeMasking(time_mask_param=6, iid_masks = True)
            freq_mask_2 = torchaudio.transforms.FrequencyMasking(freq_mask_param=6, iid_masks = True)
            inputs = t_mask_1(inputs)
            inputs = freq_mask_1(inputs)
            inputs = t_mask_2(inputs)
            inputs = freq_mask_2(inputs)
            inputs, labels = inputs.to(device), labels.to(device)
            if epoch <= epochs//2:
                inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, 0.2)

            optimizer.zero_grad()
            outputs = model(inputs)
            # loss = criterion(outputs, labels) 
            if epoch <= epochs//2:
                loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            else:
                loss = criterion(outputs, labels) 
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            optimizer.step()
            #sched.step()
            scheduler.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # scheduler.step()

        train_loss =running_loss/len(trainloader)
        accu=100.*correct/total

        my_lr = scheduler.get_last_lr()[0]
        print('Train Loss: %.3f | Accuracy: %.3f | lr: %f'%(train_loss,accu, my_lr), end = " | ")
        val_loss, val_accu = testing(model, valloader, criterion)
        if best_accu_val < val_accu:
            best_accu_val = val_accu
            save_checkpoint(model, optimizer, PATH)
        else:
            cnt += 1
            if cnt >= patience:
                break
    save_checkpoint(model, optimizer, "/scratch/hh3043/ML_contest/checkpoint_resnext101_test_max_1.pth")
    my_lr = scheduler.get_last_lr()[0]
    print('Finished Training', "last_learning_rate", my_lr)


    model.load_state_dict(torch.load(PATH)['state_dict'])
    model = model.to(device)
    prediction(model, transform)
    print("Finish saving")