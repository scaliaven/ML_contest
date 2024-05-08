import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset
from model_zoo import resnet18, Transformer, residual_attention_network
from data_util import CustomImageDataset, metaDataset, test_metaDataset, test_CustomImageDataset
import torch.optim as optim
from predict import prediction
import torch.nn as nn
from tqdm import tqdm
import torchaudio
import random
import numpy as np

# def set_random_seeds(seed_value=42):
#     random.seed(seed_value)         
#     np.random.seed(seed_value)     
#     torch.manual_seed(seed_value)
#     torch.cuda.manual_seed(seed_value)
#     torch.backends.cudnn.deterministic = True

# set_random_seeds(seed_value=42)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.version.cuda)
print(device)

######## set up ##########
save = False
meta_save = False
patience = 400
predict = False
continue_train = False
epochs = 10
PATH = "/scratch/hh3043/ML_contest/checkpoint_mata.pth"
best_accu_val = 0
#########################

transform = transforms.Compose([transforms.Normalize((0.5, ), (0.5, )) ])


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

def predicting(net, test_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    net.eval()
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



class meta(nn.Module):
    def __init__(self):
        super(meta, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(4, 1), stride = 1, padding = 0)
        self.bn = nn.BatchNorm2d(1) 
    def forward(self, x):
        conv_out = self.conv(x)
        conv_out = self.bn(conv_out)
        output = conv_out.view(conv_out.size(0), -1)
        return output


### load 4 models: atten_92, resNext50, resnet15, Densenet 201

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
if save:
    train_data = CustomImageDataset("/scratch/hh3043/ML_contest/dataset/train_label.txt", "/scratch/hh3043/ML_contest/separate/train_img", transform = transform, mask = None, balance = True)
    val_data = CustomImageDataset("/scratch/hh3043/ML_contest/dataset/train_label.txt", "/scratch/hh3043/ML_contest/separate/val_img", transform = transform, mask = None, balance = False)
    torch.save(train_data,'/scratch/hh3043/ML_contest/separate/train_64_gray.pt')
    torch.save(val_data,'/scratch/hh3043/ML_contest/separate/val_64_gray.pt')
    trainloader = DataLoader(train_data, batch_size=16, shuffle=False, num_workers=3)
    valloader = DataLoader(val_data, batch_size=16, shuffle=False, num_workers=3)

else:
    train_data = torch.load('/scratch/hh3043/ML_contest/separate/train_64_gray.pt')
    val_data = torch.load('/scratch/hh3043/ML_contest/separate/val_64_gray.pt')
    trainloader = DataLoader(train_data, batch_size=16, collate_fn=collate_fn, shuffle=False, num_workers=3)
    valloader = DataLoader(val_data, batch_size=16, collate_fn=collate_fn, shuffle=False, num_workers=3)
    pre_valloader = valloader

### create corresponding output (train, val, test): store in 3 pt files (dataloaders)
def converting_test(model_list, path):
    test_data = test_CustomImageDataset("/scratch/hh3043/ML_contest/separate/test_img", transform=transform)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False, num_workers=3)
    totaldata = torch.zeros((1, 1, 4, 4))
    for data in tqdm(test_loader):
        outputs = []
        images = data
        images = images.to(device)
        for model in model_list:
            model = model.to(device)
            model.eval()
            with torch.no_grad():
                output = model(images).unsqueeze(1).unsqueeze(1)
                outputs.append(output)
        outputs = torch.cat(outputs, dim = 2).cpu()
        totaldata = torch.cat((totaldata, outputs), dim=0)
    print(totaldata.shape)
    dataset = test_metaDataset(totaldata[1:])
    torch.save(dataset, path)

def converting(loader, model_list, path):
    totallabel = []
    totaldata = torch.torch.zeros((1, 1, 4, 4))
    for data in tqdm(loader):
        outputs = []
        images, labels = data
        images= images.to(device)
        for model in model_list:
            model = model.to(device)
            model.eval()
            with torch.no_grad():
                output = model(images).unsqueeze(1).unsqueeze(1)
                outputs.append(output)
        outputs = torch.cat(outputs, dim = 2).cpu()
        totaldata = torch.cat((totaldata, outputs), dim=0)
        totallabel += labels
    dataset = metaDataset(totaldata[1:], totallabel)
    torch.save(dataset, path)
if meta_save:
    converting(trainloader, model_list, "/scratch/hh3043/ML_contest/separate/train_meta.pt")
    converting(valloader, model_list, "/scratch/hh3043/ML_contest/separate/val_meta.pt")
    converting_test(model_list, "/scratch/hh3043/ML_contest/separate/test_meta.pt")

### load the outputs
train_data = torch.load('/scratch/hh3043/ML_contest/separate/train_meta.pt')
val_data = torch.load('/scratch/hh3043/ML_contest/separate/val_meta.pt')
test_data =  torch.load("/scratch/hh3043/ML_contest/separate/test_meta.pt")

trainloader = DataLoader(train_data, batch_size=16, collate_fn=collate_fn, shuffle=True, num_workers=3)
valloader = DataLoader(val_data, batch_size=16, collate_fn=collate_fn, shuffle=False, num_workers=3)


def collate_fn_test(batch):
    print(batch.size)
    image = torch.stack([x for x in batch])
    print(image.shape)
    return Image


test_loader = DataLoader(test_data, batch_size=16, shuffle=False, num_workers=1)

for data in tqdm(pre_valloader):
    correct = 0
    total = 0
    outputs = torch.zeros((data[0].shape[0], 4)).to(device)
    for model in model_list:
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            outputs += output
    _, predicted = torch.max(outputs.data, 1)

    total += labels.size(0)
    correct += predicted.eq(labels).sum().item()
    accu=100.*correct/total
print("Validation Accuracy without reweights: %.3f "%(accu))


model = meta()
if predict:

    model.load_state_dict(torch.load(PATH)['state_dict'])
    model = model.to(device)
    predicting(model, transform)
    print("Finish saving")

else:

    if continue_train:
        model.load_state_dict(torch.load(PATH)['state_dict'])
    model.to(device)

    criterion = nn.CrossEntropyLoss().to(device)
    max_lr = 5e-4
    optimizer = optim.SGD(model.parameters(), lr = max_lr, weight_decay = 1.0e-3, momentum = 0.9) 
    grad_clip = 0.1
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = epochs*len(trainloader))

    best_val = 100
    cnt = 0
    ### train the meta model
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct=0
        total=0
        for i, data in enumerate(tqdm(list(trainloader))):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels) 
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

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
        
    my_lr = scheduler.get_last_lr()[0]
    print('Finished Training', "last_learning_rate", my_lr)


### use the meta model on the final prediction
    model.load_state_dict(torch.load(PATH)['state_dict'])
    predicting(model, test_loader)
    print("Finish saving")