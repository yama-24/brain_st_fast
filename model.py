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
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class Net(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(100352, 128)
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):

        # 1層目
        h = self.conv1(x)
        h = F.relu(h)
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        # 2層目
        h = self.conv2(h)
        h = F.relu(h)
        h = F.max_pool2d(h, kernel_size=2, stride=2)

        # 全結合層
        h = h.view(-1, 100352)
        h = self.fc1(h)
        h = F.relu(h)
        h = self.fc2(h)
        return h