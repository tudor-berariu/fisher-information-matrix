import torch.nn as nn
import torch.nn.functional as F

from kfac import KFACModule


class ConvNet(KFACModule):
    def __init__(self, **kwargs):
        super(ConvNet, self).__init__(**kwargs)
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.drop_conv2 = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.drop_fc1 = nn.Dropout()
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.drop_conv2(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)
        x = x.view(-1, 320)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop_fc1(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
