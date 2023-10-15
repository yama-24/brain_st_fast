# 必要なモジュールのインポート
from torchvision import transforms
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

# 前処理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])

class Net(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.fc = nn.Linear(50176, 4)

    def forward(self, x):

        # 1層目
        h = self.conv1(x)
        h = F.relu(h)
        h = self.bn1(h)
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        # 2層目
        h = self.conv2(h)
        h = F.relu(h)
        h = self.bn2(h)
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        # 3層目
        h = self.conv3(h)
        h = F.relu(h)
        h = self.bn3(h)
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        # 全結合層
        h = h.view(-1, 50176)
        h = self.fc(h)
        return h