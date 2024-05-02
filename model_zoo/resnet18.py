import torch.nn as nn
import torch.nn.functional as F

def res(input_channel):
    block = nn.Sequential(
          nn.Conv2d(input_channel,input_channel,3, padding = 1, bias=False),
          nn.BatchNorm2d(input_channel),
          nn.ReLU(True),
          nn.Conv2d(input_channel,input_channel,3, padding = 1, bias=False),
          nn.BatchNorm2d(input_channel),
        )
    
    return nn.Sequential(*block)

def conv_block(input_channel, output_channel, filter_size = 3,padding = 1):
    block = nn.Sequential(
          nn.Conv2d(input_channel,output_channel,filter_size, stride = 2, padding = padding, bias = False),
          nn.BatchNorm2d(output_channel),
          nn.ReLU(True),
          nn.Conv2d(output_channel,output_channel,filter_size, stride = 1, padding = padding, bias = False),
          nn.BatchNorm2d(output_channel),
        )
    
    return nn.Sequential(*block)

def start(input_channel, output_channel, filter_size = 3, stride = 1, padding = 1):
    block = nn.Sequential(
          nn.Conv2d(input_channel,output_channel,filter_size, stride = stride, padding = padding, bias = False),
          nn.BatchNorm2d(output_channel),
          nn.ReLU(True),
        )
    
    return nn.Sequential(*block)

def downsample(input_channel, output_channel):
    block = nn.Sequential(
          nn.Conv2d(input_channel,output_channel,1, stride = 2, bias = False),
          nn.BatchNorm2d(output_channel),
        )
    
    return nn.Sequential(*block)


class resnet18(nn.Module):
    def __init__(self):
        super(resnet18, self).__init__()
        self.conv1 = start(1, 64)
        self.pool_1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res1_1 = res(64)
        self.res1_2 = res(64)
        
        self.conv2 = conv_block(64, 128, 3, padding = 1)
        self.sample_1 = downsample(64, 128)
        self.res2_1 = res(128)
        
        self.conv3 = conv_block(128, 256, 3, padding = 1)
        self.res3_1 = res(256)
        self.sample_2 = downsample(128, 256)
      
        self.conv4 = conv_block(256, 512, 3, padding = 1)
        self.res4_1 = res(512)
        self.sample_3 = downsample(256, 512)
        self.drop = nn.Dropout()
        
        self.averagepool = nn.AdaptiveAvgPool2d(output_size = (1, 1))   #fghjkl
        self.fc1 = nn.Linear(512, 4)   ###how many labels are needed in this task?
        
        
    def forward(self, x):
        x = self.conv1(x)
        #x = self.pool_1(x)
        x = F.relu(self.res1_1(x) + x)
        x = F.relu(self.res1_2(x) + x)
        
        
        x = F.relu(self.conv2(x) + self.sample_1(x))
        x = F.relu(self.res2_1(x) + x)


        x = F.relu(self.conv3(x) + self.sample_2(x))
        x = F.relu(self.res3_1(x) + x)


        x = F.relu(self.conv4(x) + self.sample_3(x))
        x = F.relu(self.res4_1(x) + x)

        x = self.averagepool(x)
        x = x.reshape((-1, 512))
        x = self.fc1(x)
        return x