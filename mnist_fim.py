import argparse
from argparse import Namespace

import torch
from torchvision import datasets, transforms

from fim import fim_diag
from kfac import kfac
from models import ConvNet, MLP


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--samples-no', type=int, default=60000, metavar='S',
                        help='Samples to use for estimating Fisher')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='batch_size')
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
        batch_size=args.batch_size, shuffle=True, **kwargs)

    if args.model == "mlp":
        Model = MLP
    elif args.model == "conv":
        Model = ConvNet

    model = Model().to(device)

    if args.mode == "diag":
        fim_diag(model, train_loader, samples_no=args.samples_no,
                 empirical=args.empirical, device=device, verbose=True)
    elif args.mode in ["block", "triblock"]:
        model.tridiag = (args.mode == "triblock")
        gaussian_prior = kfac(model, train_loader, samples_no=args.samples_no,
                              empirical=args.empirical, device=device, verbose=True)

        new_model = Model().to(device)

        crt_params = {name: param for (name, param) in new_model.named_parameters()}
        gaussian_prior(crt_params)


if __name__ == "__main__":
    main()
