import torch
import torch.nn as nn
from torchvision.models import resnet18

def get_model(model_name, num_classes=10, dropout=0.5):
    if model_name == "simpleCNN":
        return CNN(num_classes=num_classes, dropout=dropout)
    if model_name == "resnet18":
        model = resnet18(weights=None)
        model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        model.maxpool = nn.Identity()
        model.fc = nn.Linear(512, num_classes)
        # ``dropout`` used to be ignored for resnet18. Insert it after global pooling so
        # the classifier keys (``fc.weight``) stay unchanged and old checkpoints still load.
        model.avgpool = nn.Sequential(model.avgpool, nn.Dropout(dropout))
        return model
    else:
        raise ValueError("Unsupported model name")

class CNN(torch.nn.Module):
    def __init__(self, num_classes=10, dropout=0.5):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.adaptive_pool = torch.nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = torch.nn.Linear(64 * 4 * 4, 128)
        self.fc2 = torch.nn.Linear(128, num_classes)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x