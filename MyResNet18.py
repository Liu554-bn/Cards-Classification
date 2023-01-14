import torch.nn as nn
import torch.nn.functional as F


class basic_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(basic_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = self.bn1(y)
        y = self.conv2(y)
        y = self.bn2(y)
        return F.relu(x + y)

class basic_block2(nn.Module):
    def __init__(self,in_channels,out_channels):# 3-64 64-128 128-256
        super(basic_block2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=2,padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=2,padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels,out_channels,kernel_size=1,stride=1,padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels)
    def forward(self, x):
        z = self.conv1(x)
        z = self.bn1(z)
        y = F.relu(self.bn2(self.conv2(x)))
        y = self.conv3(y)
        y = self.bn3(y)
        return F.relu(y+z)

class Resnet18(nn.Module):
    '''按照网络结构图直接连接，确定好通道数量就可以'''

    def __init__(self):
        super(Resnet18, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.res2 = basic_block(64, 64)
        self.res3 = basic_block(64, 64)
        self.res4 = basic_block2(64, 128)
        self.res5 = basic_block(128, 128)
        self.res6 = basic_block2(128, 256)
        self.res7 = basic_block(256, 256)
        self.res8 = basic_block2(256, 512)
        self.res9 = basic_block(512, 512)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(512, 53)

    def forward(self, x):
        x = self.conv1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        x = self.res6(x)
        x = self.res7(x)
        x = self.res8(x)
        x = self.res9(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

