import torch.nn as nn
import torch.nn.functional as F
from .resnet18 import res, conv_block, start, downsample

class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=128, hidden_size=512, num_layers=2, batch_first=True, bidirectional = True)
        self.linear = nn.Linear(512*2, 4)
    def forward(self, x):
        x = x.permute(0, 2, 3, 1).squeeze(-1)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.linear(x)
        return x


class AttentionBlock(nn.Module):
    def __init__(self, in_features_l, in_features_g, attn_features, up_factor, normalize_attn=True):
        super(AttentionBlock, self).__init__()
        self.up_factor = up_factor
        self.normalize_attn = normalize_attn
        self.W_l = nn.Conv2d(in_channels=in_features_l, out_channels=attn_features, kernel_size=1, padding=0, bias=False)
        self.W_g = nn.Conv2d(in_channels=in_features_g, out_channels=attn_features, kernel_size=1, padding=0, bias=False)
        self.phi = nn.Conv2d(in_channels=attn_features, out_channels=1, kernel_size=1, padding=0, bias=True)
    def forward(self, l, g):
        N, C, W, H = l.size()
        l_ = self.W_l(l)
        g_ = self.W_g(g)
        if self.up_factor > 1:
            g_ = F.interpolate(g_, scale_factor=self.up_factor, mode='bilinear', align_corners=False)
        c = self.phi(F.relu(l_ + g_)) # batch_sizex1xWxH
        
        # compute attn map
        if self.normalize_attn:
            a = F.softmax(c.view(N,1,-1), dim=2).view(N,1,W,H)
        else:
            a = torch.sigmoid(c)
        # re-weight the local feature
        f = torch.mul(a.expand_as(l), l) # batch_sizexCxWxH
        if self.normalize_attn:
            output = f.view(N,C,-1).sum(dim=2) # weighted sum
        else:
            output = F.adaptive_avg_pool2d(f, (1,1)).view(N,C) # global average pooling
        return a, output




class attention_resnet18(nn.Module):
    def __init__(self):
        super(attention_resnet18, self).__init__()
        self.atten = nn.MultiheadAttention(128, 1, batch_first= True)
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
        # self.fc = nn.Linear(512 * 16 * 16, 512)
        
        
    def forward(self, x):
        x = x.permute(0, 2, 3, 1).squeeze(-1)
        x, _ = self.atten(x, x, x)
        x = x.unsqueeze(1)
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

        # x = x.reshape((-1, 512*16*16))
        x = self.averagepool(x)
        # x = self.fc(x)
        x = x.reshape((-1, 512))
        x = self.fc1(x)
        return x


