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
from torch.utils.data import DataLoader, ConcatDataset
from model_zoo import resnet18
from data_util import CustomImageDataset
import torch.optim as optim
from predict import prediction
import torch.nn as nn
from tqdm import tqdm
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(torch.version.cuda)
print(device)

######## set up ##########
save = True
patience = 1000
predict = True
epochs = 20


##########################

transform = transforms.Compose(
    [
    # transforms.Resize((128, 128)),
    # transforms.CenterCrop()
    # transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, )) 
    ])

transform_1 = transforms.Compose(
    [
    # transforms.Resize((128, 128)),
    # transforms.RandomHorizontalFlip(1),
    # transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) 
    ])

if predict:
    model = resnet18.resnet18()
    PATH = "/scratch/hh3043/ML_contest/checkpoint_7.pth"
    model.load_state_dict(torch.load(PATH)['state_dict'])
    model = model.to(device)
    prediction(model, transform)
    print("Finish saving")


def save_checkpoint(model, optimizer):

    checkpoint = {'model': resnet18.resnet18(),
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict()}
    # if str_name = "accu":
    #     torch.save(checkpoint, '/scratch/hh3043/ML_contest/checkpoint_7_accu.pth')
    torch.save(checkpoint, '/scratch/hh3043/ML_contest/checkpoint_7.pth')


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



if save:
    train_data = CustomImageDataset("/scratch/hh3043/ML_contest/dataset/train_label.txt", "/scratch/hh3043/ML_contest/dataset/train_gray_img", transform = transform, mask = None, balance = True)
    # train_data_1 = CustomImageDataset("/scratch/hh3043/ML_contest/dataset/train_label.txt", "/scratch/hh3043/ML_contest/dataset/train_img", transform = transform_1, mask = None, balance = True)
    val_data = CustomImageDataset("/scratch/hh3043/ML_contest/dataset/train_label.txt", "/scratch/hh3043/ML_contest/dataset/val_gray_img", transform = transform, mask = None, balance = False)
    torch.save(train_data,'/scratch/hh3043/ML_contest/dataset/train_64_gray.pt')
    torch.save(val_data,'/scratch/hh3043/ML_contest/dataset/val_64_gray.pt')
    # torch.save(train_data_1,'/scratch/hh3043/ML_contest/dataset/train_64_hflip.pt')
    trainloader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=3)
    valloader = DataLoader(val_data, batch_size=16, shuffle=False, num_workers=3)

else:
    train_data = torch.load('/scratch/hh3043/ML_contest/dataset/train_64_gray.pt')
    # train_data_1 = torch.load('/scratch/hh3043/ML_contest/dataset/train_64_hflip.pt')
    val_data = torch.load('/scratch/hh3043/ML_contest/dataset/val_64_gray.pt')
    trainloader = DataLoader(train_data, batch_size=16, collate_fn=collate_fn, shuffle=True, num_workers=3)
    valloader = DataLoader(val_data, batch_size=16, collate_fn=collate_fn, shuffle=False, num_workers=3)



model = resnet18.resnet18()
model.to(device)

criterion = nn.CrossEntropyLoss().to(device)
max_lr = 5e-3
optimizer = optim.SGD(model.parameters(), lr = max_lr, weight_decay = 1.0e-3, momentum = 0.9) 
grad_clip = 0.1
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = epochs*len(trainloader))
# sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(trainloader))
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = [epochs // 3, epochs * 2 // 3], gamma = 0.1, last_epoch = -1)




best_val = 100
cnt = 0
best_accu_val = 0

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct=0
    total=0
    for i, data in enumerate(tqdm(list(trainloader))):

        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        #inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, 0.2)

        optimizer.zero_grad()
        outputs = model(inputs)
        
        loss = criterion(outputs, labels) 
        # loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        loss.backward()
        nn.utils.clip_grad_value_(model.parameters(), grad_clip)
        optimizer.step()
        #sched.step()
        scheduler.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)   #
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    # scheduler.step()

    train_loss =running_loss/len(trainloader)
    accu=100.*correct/total

    my_lr = scheduler.get_last_lr()[0]
    print('Train Loss: %.3f | Accuracy: %.3f | lr: %f'%(train_loss,accu, my_lr), end = " | ")
    val_loss, val_accu = testing(model, valloader, criterion)
    # if val_loss < best_val:
    #     best_val = val_loss
    #     save_checkpoint(model, optimizer)
    if best_accu_val < val_accu:
        best_accu_val = val_accu
        save_checkpoint(model, optimizer)
    else:
        cnt += 1
        if cnt >= patience:
            break
    
my_lr = scheduler.get_last_lr()[0]
print('Finished Training', "last_learning_rate", my_lr)



model = resnet18.resnet18() #resnet18.resnet18()
PATH = "/scratch/hh3043/ML_contest/checkpoint_7.pth"
model.load_state_dict(torch.load(PATH)['state_dict'])
model = model.to(device)
prediction(model, transform)
print("Finish saving")