from typing import Dict
import sys
import argparse
from argparse import Namespace
import math
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.optim as optim
import torch.autograd as autograd
from torch import Tensor
from torchvision import datasets, transforms

from fim import fim_diag
from kfac import kfac
from models import ConvNet, MLP


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--samples-no', type=int, default=10000, metavar='S',
                        help='Samples to use for estimating Fisher')
    parser.add_argument('--every', type=int, default=False,
                        help='Look at several Fisher matrices')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='batch_size')
    parser.add_argument('--epochs-no', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA')
    parser.add_argument('--empirical', action="store_true", default=False,
                        help="Empirical FIM.")
    parser.add_argument('--mode', type=str, default="diag",
                        choices=["diag", "block", "triblock"],
                        help="Fisher approximation.")

    parser.add_argument('--model', type=str, default="mlp",
                        choices=["mlp", "conv"],
                        help="Model to be used.")

    # Sparsity
    parser.add_argument('--sparsity-mode', type=str, default="LG1",
                        choices=["lg1", "l1"],
                        help="Sparsity stategy")
    parser.add_argument('--sparsity-scale', type=float, default=0,
                        help='Scale for sparsity')
    parser.add_argument('--sparsity-norm', type=float, default=1,
                        help='Norm for sparsity')

    return parser.parse_args()


def train(model, train_loader, args, device=None):
    optimizer = optim.Adam(model.parameters(), lr=.001)
    print("Start training...")
    total_no, correct_no = 0, 0
    for epoch_no in range(args.epochs_no):
        for idx, (data, target) in enumerate(train_loader):
            if device:
                data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            correct_no += (output.argmax(dim=1) == target).sum()
            total_no += output.size(0)
            loss = -output.gather(1, target.unsqueeze(1)).mean()
            grads = autograd.grad([loss], model.parameters(), retain_graph=True)
            sparsity_loss = torch.norm(torch.cat(tuple([g.view(-1) for g in grads])),
                                       p=args.sparsity_norm)
            loss += args.sparsity_scale * sparsity_loss
            loss.backward()
            optimizer.step()
            if total_no > 1000:
                acc = float(correct_no) / float(total_no)
                correct_no, total_no = 0, 0
                sys.stdout.write(f"\rEpoch {epoch_no:2d}."
                                 f"  Loss: {loss.item():2.5f}."
                                 f"  Acc: {acc*100:2.2f}%")
    print("\nEnd training.")


def show_fisher(fim: Dict[str, Tensor], samples_no, args, scale=3):
    name = f"{args.mode:s}_{args.model:s}_{args.epochs_no:d}"
    if args.sparsity_scale > 0:
        name = name + f"_{args.sparsity_scale:f}_{args.sparsity_norm:f}"
    if args.empirical:
        name = f"empirical_{name:s}"
    for p_name, params in fim.items():
        if params.ndimension() == 1:
            height, width = params.size(0), 1
        elif params.ndimension() == 2:
            height, width = params.size()
        else:
            raise NotImplementedError
        img = params.view(height, 1, width, 1)\
                    .repeat(1, scale, 1, scale)\
                    .view(height * scale, width * scale)\
                    .to(torch.device("cpu"))\
                    .numpy()
        plt.figure(figsize=(24, 24))
        sns.heatmap(img, cmap='Greys')
        # plt.show()
        plt.savefig(f"figs/{name:s}_{p_name:s}_{samples_no:d}.png")
        plt.clf()


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
        batch_size=args.batch_size, shuffle=True, **kwargs)

    if args.model == "mlp":
        Model = MLP
    elif args.model == "conv":
        Model = ConvNet

    model = Model().to(device)
    train(model, train_loader, args, device=device)

    if args.mode == "diag":
        all_fims = fim_diag(model, train_loader, samples_no=args.samples_no,
                            empirical=args.empirical, device=device, verbose=True,
                            every_n=args.every)
        for samples_no, fisher in all_fims.items():
            show_fisher(fisher, samples_no, args)

    elif args.mode in ["block", "triblock"]:
        model.tridiag = (args.mode == "triblock")
        gaussian_prior = kfac(model, train_loader, samples_no=args.samples_no,
                              empirical=args.empirical, device=device, verbose=True)

        new_model = Model().to(device)

        crt_params = {name: param for (name, param) in new_model.named_parameters()}
        gaussian_prior(crt_params)


if __name__ == "__main__":
    main()
