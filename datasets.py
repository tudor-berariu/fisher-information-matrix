from typing import List, Optional, Tuple
from multiprocessing.pool import ThreadPool
from argparse import Namespace
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


ORIGINAL_SIZE = {
    "mnist": torch.Size((1, 28, 28)),
    "fashion": torch.Size((1, 28, 28)),
    "cifar10": torch.Size((3, 32, 32)),
    "svhn": torch.Size((3, 32, 32)),
    "cifar100": torch.Size((3, 32, 32)),
    "fake":  torch.Size((3, 32, 32)),
}

MEAN_STD = {
    "mnist": {(3, 32, 32): (0.10003692801078261, 0.2752173485400458)},
    "fashion": {(3, 32, 32): (0.21899983604159193, 0.3318113789274)},
    "cifar10": {(3, 32, 32): (0.4733630111949825, 0.25156892869250536)},
    "cifar100": {(3, 32, 32): (0.478181, 0.268192)},
    "svhn": {(3, 32, 32): (0.451419, 0.199291)}
}

DATASETS = {
    "mnist": datasets.MNIST,
    "fashion": datasets.FashionMNIST,
    "cifar10": datasets.CIFAR10,
    "svhn": datasets.SVHN,
    "cifar100": datasets.CIFAR100
}


class InMemoryDataLoader(object):

    def __init__(self, data: Tensor, target: Tensor,
                 batch_size: int, shuffle: bool = True) -> None:
        self.data, self.target = data, target
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.__index = None

    def __len__(self) -> int:
        return self.data.size(0)

    def __iter__(self):
        randperm = torch.randperm(self.data.size(0)).to(self.data.device)
        self.data = self.data.index_select(0, randperm)
        self.target = self.target.index_select(0, randperm)
        self.__index = 0
        return self

    def __next__(self) -> Tuple[Tensor, Tensor]:
        start = self.__index
        if self.__index >= self.data.size(0):
            raise StopIteration
        end = min(start + self.batch_size, self.data.size(0))
        batch = self.data[start:end], self.target[start:end]
        self.__index = end
        return batch


Padding = Tuple[int, int, int, int]


def get_padding(in_size: torch.Size, out_size: torch.Size) -> Padding:
    assert len(in_size) == len(out_size)
    d_h, d_w = out_size[-2] - in_size[-2], out_size[-1] - in_size[-1]
    p_h1, p_w1 = d_h // 2, d_w // 2
    p_h2, p_w2 = d_h - p_h1, d_w - p_w1
    return (p_h1, p_h2, p_w1, p_w2)


def load_data_async(dataset_name: str,
                    in_size: Optional[torch.Size] = None):

    original_size = ORIGINAL_SIZE[dataset_name]
    in_size = in_size if in_size is not None else original_size
    padding = get_padding(original_size, in_size)
    mean, std = MEAN_STD[dataset_name][tuple(in_size)]

    if dataset_name == "svhn":
        train_data = DATASETS[dataset_name](
            f'./.data/.{dataset_name:s}_data',
            split="train", download=True,
            transform=transforms.Compose([
                transforms.Pad(padding),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: t.expand(in_size)),
                transforms.Normalize((mean,), (std,))
            ]))
    else:
        train_data = DATASETS[dataset_name](
            f'./.data/.{dataset_name:s}_data',
            train=True, download=True,
            transform=transforms.Compose([
                transforms.Pad(padding),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: t.expand(in_size)),
                transforms.Normalize((mean,), (std,))
            ]))
    if dataset_name == "svhn":
        test_data = DATASETS[dataset_name](
            f'./.data/.{dataset_name:s}_data',
            split="test", download=True,
            transform=transforms.Compose([
                transforms.Pad(padding),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: t.expand(in_size)),
                transforms.Normalize((mean,), (std,))
            ]))
    else:
        test_data = DATASETS[dataset_name](
            f'./.data/.{dataset_name:s}_data',
            train=False, download=True,
            transform=transforms.Compose([
                transforms.Pad(padding),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: t.expand(in_size)),
                transforms.Normalize((mean,), (std,))
            ]))

    loader = DataLoader(train_data, batch_size=len(train_data),
                        num_workers=4)
    train_data, train_target = next(iter(loader))
    del loader

    loader = DataLoader(test_data, batch_size=len(test_data),
                        num_workers=4)
    test_data, test_target = next(iter(loader))
    del loader

    return train_data, train_target, test_data, test_target


class DataSetFactory(object):

    def __init__(self, all_datasets: List[str],
                 in_size: Optional[torch.Size] = None) -> None:
        self.full_data = {}
        pool = ThreadPool(processes=len(all_datasets))
        for dataset_name in all_datasets:
            self.full_data[dataset_name] = pool.apply_async(
                load_data_async, (dataset_name, in_size))

    def get_datasets(self, dataset_name: str,
                     device: torch.device,
                     args: Namespace):

        train_data, train_target, test_data, test_target = \
            self.full_data[dataset_name].get()
        train_loader = InMemoryDataLoader(train_data.to(device),
                                          train_target.to(device),
                                          shuffle=True,
                                          batch_size=args.batch_size)
        test_loader = InMemoryDataLoader(test_data.to(device),
                                         test_target.to(device),
                                         shuffle=False,
                                         batch_size=args.test_batch_size)

        return train_loader, test_loader
