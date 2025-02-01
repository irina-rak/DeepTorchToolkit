import torch
import torch.nn as nn

from torch.nn import functional as F
from torchvision.models import resnet18, ResNet18_Weights, resnet50

from src.ml.models.cnn2d.GroupNorm2d import GroupNorm2d


class ResNet18(nn.Module):
    def __init__(self, in_channels=3, n_classes=10):
        super(ResNet18, self).__init__()

        # self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model = resnet18()

        self.model.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        self.model.fc = nn.Linear(512, n_classes)

        # Sigmoid and Softmax are not used here since they are included in the loss function
        # self.softmax = nn.Softmax(dim=1)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        # return self.softmax(x)
        return x
        # return self.sigmoid(x)
