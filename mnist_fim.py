import argparse
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module
from torchvision import datasets, transforms

from fim import fim_diag


class ConvNet(Module):
    def __init__(self):
        super(ConvNet, self).__init__()
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


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--samples-no', type=int, default=64, metavar='S',
                        help='Samples to use for estimating Fisher')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device("cuda:0" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=1, shuffle=True, **kwargs)

    model = ConvNet().to(device)

    fim_diag(model, train_loader, args.samples_no, device, verbose=True)


if __name__ == "__main__":
    main()
