from typing import Dict
from os.path import isfile
import sys
import argparse
from argparse import Namespace
import matplotlib.pyplot as plt
import seaborn as sns
from termcolor import colored as clr

import numpy as np

import torch
import torch.optim as optim
import torch.autograd as autograd
from torch import Tensor
from torchvision import datasets, transforms

from fim import fim_diag
from models import ConvNet, MLP
from fisher_metrics import frechet_diags, cosine_distance,\
    dot_product, unit_trace_diag, unit_trace_diag_
from datasets import DataSetFactory


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(description="Continuous learning")

    parser.add_argument("--no-cuda", action="store_true", default=False,
                        help="disables CUDA")

    # ---------- Script control
    parser.add_argument("--skip-b", action="store_true", default=False,
                        help="Forces train")
    parser.add_argument("--force-train-a", action="store_true", default=False,
                        help="Forces train")
    parser.add_argument("--trials-no", type=int, default=5,
                        help="Number of trials.")

    # ---------- Tasks
    parser.add_argument("--datasets", type=str, default="mf",
                        help="Tasks order.")

    # ---------- Model
    parser.add_argument("--model", type=str, default="mlp", choices=["mlp", "conv"],
                        help="Model to be used.")
    parser.add_argument("--hidden-units", type=int, default=200,
                        help="No. of hidden units.")

    # ---------- Training
    parser.add_argument("--batch-size", type=int, default=128, metavar="N",
                        help="batch_size")
    parser.add_argument("--test-batch-size", type=int, default=1000, metavar="N",
                        help="test batch_size")
    parser.add_argument("--epochs-no", type=int, default=50,
                        help="Number of training epochs")

    # ---------- EWC
    parser.add_argument("--ewc", action="store_true", default=False,
                        help="Use EWC while training for the 2nd task")
    parser.add_argument("--ewc-scale", type=float, default=400,
                        help="Scale for EWC")

    # ---------- Fisher computation
    parser.add_argument("--empirical", action="store_true", default=False,
                        help="Empirical FIM.")
    parser.add_argument("--mode", type=str, default="diag", choices=["diag"],
                        help="Fisher approximation.")
    parser.add_argument("--samples-no", type=int, default=15000, metavar="S",
                        help="Samples to use for estimating Fisher")
    parser.add_argument("--every", type=int, default=False,
                        help="Look at several Fisher matrices")

    # ---------- Sparsity
    parser.add_argument("--sparsity-strategy", type=str, default="no",
                        choices=["lg1", "l1", "no"],
                        help="Sparsity stategy")
    parser.add_argument("--sparsity-scale", type=float, default=1,
                        help="Scale for sparsity")
    parser.add_argument("--sparsity-norm", type=float, default=.9,
                        help="Norm for sparsity")

    # ---------- Fisher overlap
    parser.add_argument("--overlap-strategy", type=str, default="frechet",
                        choices=["frechet", "cosine", "dot", "no"],
                        help="Overlap strategy")
    parser.add_argument("--overlap-samples-no", type=int, default=32,
                        help="Samples to use for overlap")
    parser.add_argument("--overlap-scale", type=float, default=1,
                        help="Overlap scale")

    return parser.parse_args()


def get_results_name(args):
    names = sorted(list(args.__dict__.keys()))
    name = "_".join(f"{n:s}_{str(args.__dict__[n]):s}" for n in names)
    return f"fisher_results/results_{name:s}.pkl"


def test(model, test_loader, name="Test", device=None):
    model.eval()
    total_no, correct_no = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            if device:
                data, target = data.to(device), target.to(device)
            output = model(data)
            correct_no += (output.argmax(dim=1) == target).sum()
            total_no += output.size(0)
    acc = float(correct_no) / float(total_no)
    correct_no, total_no = 0, 0
    sys.stdout.write(clr(f" | [Eval - {name:s}]  Acc: {acc*100:2.2f}%\n",
                         attrs=['bold']))
    model.train()


def train(model, optimizer, loaders, name, args, old_fim=None, device=None):
    print("Start training...")
    total_no, correct_no = 0, 0
    train_loader, test_loader = loaders

    ordered_names, ordered_parameters = [], []
    for p_name, parameters in model.named_parameters():
        ordered_names.append(p_name)
        ordered_parameters.append(parameters)

    sparsity_loss = torch.tensor(.0, device=device)
    overlap_loss = torch.tensor(.0, device=device)

    if old_fim is not None and args.overlap_strategy == "frechet":
        # We normalize and take square root of F2
        old_fim_ut = clone_fim(old_fim)
        trace = sum(t.sum() for t in old_fim_ut.values()).detach_()
        for t in old_fim_ut.values():
            t.div_(trace).sqrt_()

    for epoch_no in range(args.epochs_no):
        model.train()
        for _idx, (data, target) in enumerate(train_loader):
            if device:
                data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            logits = model(data)
            correct_no += (logits.argmax(dim=1) == target).sum()
            total_no += logits.size(0)

            # NLL
            nll_loss = -logits.gather(1, target.unsqueeze(1)).mean()

            # Sparsity
            if args.sparsity_strategy == "l1":
                allvalues = tuple(t.view(-1) for t in model.parameters())
                sparsity_loss = torch.norm(torch.cat(allvalues), p=args.sparsity_norm)
            elif args.sparsity_strategy == "lg1":
                grads = autograd.grad([nll_loss], model.parameters(),
                                      create_graph=True, retain_graph=True)
                allvalues = tuple(t.view(-1) for t in grads)
                sparsity_loss = torch.norm(torch.cat(allvalues), p=args.sparsity_norm)

            # Overlap
            if old_fim is not None and args.overlap_strategy != "no":
                o_logits = logits[0:min(args.overlap_samples_no, logits.size(0))]
                o_fisher = dict({})
                for o_idx in range(args.overlap_samples_no):
                    o_grads = autograd.grad(o_logits[o_idx, target[o_idx]],
                                            ordered_parameters,
                                            create_graph=True, retain_graph=True)
                    if o_fisher:
                        for p_name, matrix in zip(ordered_names, o_grads):
                            o_fisher[p_name] = o_fisher[p_name] + matrix * matrix
                    else:
                        for p_name, matrix in zip(ordered_names, o_grads):
                            o_fisher[p_name] = matrix * matrix
                if args.overlap_strategy == "frechet":
                    o_fisher = unit_trace_diag(o_fisher)
                    overlap_loss = frechet_diags(o_fisher, old_fim_ut)
                    frechet_norm = 0
                    for p_name, matrix in o_fisher.items():
                        diff = (matrix + 1e-8).sqrt() - old_fim_ut[p_name]
                        frechet_norm += (diff * diff).sum()
                    overlap_loss = 1 - frechet_norm / 2.0

                elif args.overlap_strategy == "cosine":
                    overlap_loss = cosine_distance(o_fisher, old_fim)
                elif args.overlap_strategy == "dot":
                    overlap_loss = dot_product(o_fisher, old_fim)

            loss = nll_loss + sparsity_loss * args.sparsity_scale \
                + overlap_loss * args.overlap_scale

            loss.backward()

            for p in model.parameters():
                # print(p.grad)
                if torch.isnan(p.grad).any():
                    print("UPS")
                    exit(1)

            optimizer.step()
            if total_no > -1:  # 1000:
                acc = float(correct_no) / float(total_no)
                correct_no, total_no = 0, 0
                sys.stdout.write(f"\r[Train - {name:s}]"
                                 f" Epoch {epoch_no:2d}"
                                 f"; NLL: {nll_loss.item():2.3f}"
                                 f"; L1: {sparsity_loss.item():2.3f}"
                                 f"; Overlap: {overlap_loss.item():2.3f}"
                                 f"; Loss: {loss.item():2.3f}"
                                 f"; Acc: {acc*100:2.2f}%")
        test(model, test_loader, name=name, device=device)
    print("Trained.")


def save_fisher(fim: Dict[str, Tensor], name, scale=3):
    for p_name, params in fim.items():
        if params.ndimension() == 1:
            height, width = params.size(0), 1
        elif params.ndimension() == 2:
            height, width = params.size()
        else:
            raise NotImplementedError
        img = params.view(height, 1, width, 1) \
                    .repeat(1, scale, 1, scale) \
                    .view(height * scale, width * scale) \
                    .to(torch.device("cpu")) \
                    .numpy()
        plt.figure(figsize=(24, 24))
        sns.heatmap(img, cmap='Greys')
        plt.savefig(f"fisher_results/{p_name:s}_{name:s}.png")
        plt.close()


def clone_fim(fim: Dict[str, Tensor]):
    return {name: matrix.clone().detach_() for (name, matrix) in fim.items()}


def main() -> None:
    args = parse_args()
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    device = torch.device("cuda:0" if use_cuda else "cpu")

    all_datasets = {"m": "mnist", "f": "fashion", "c": "cifar10"}
    crt_datasets = [all_datasets[d_name] for d_name in args.datasets]
    factory = DataSetFactory(crt_datasets, torch.Size([3, 32, 32]))

    if args.model == "mlp":
        Model = MLP
    elif args.model == "conv":
        Model = ConvNet

    print(f"\n>> I. Train a model on  {crt_datasets[0]:s}!\n")

    model = Model().to(device)
    optimizer = optim.SGD(model.parameters(), lr=.01)

    path = f"./saved_models/{args.model}"
    if not args.force_train_a and \
       all(isfile(f"{path:s}_{end:s}.pkl") for end in ['params', 'optim', 'fim']):
        model_state_dict = torch.load(f"{path:s}_params.pkl")
        model.load_state_dict(model_state_dict)

        fim_a = torch.load(f"{path:s}_fim.pkl")

        optimizer_state_dict = torch.load(f"{path:s}_optim.pkl")
        optimizer.load_state_dict(optimizer_state_dict)
        print("Loaded params, optimizer and FIM from disk.")

    else:
        first_loaders = factory.get_datasets(crt_datasets[0], device, args)

        train(model, optimizer, first_loaders, crt_datasets[0], args, device=device)
        all_fims = fim_diag(model, first_loaders[0], samples_no=args.samples_no,
                            empirical=args.empirical, device=device, verbose=True)
        fim_a = all_fims[args.samples_no]
        model_state_dict = model.state_dict()
        optimizer_state_dict = optimizer.state_dict()

        torch.save(model.state_dict(), f"{path:s}_params.pkl")
        torch.save(optimizer.state_dict(), f"{path:s}_optim.pkl")
        torch.save(fim_a, f"{path:s}_fim.pkl")

    save_fisher(fim_a, f"{args.model:s}_fim_A", scale=3)

    fim_a_unit = clone_fim(fim_a)
    unit_trace_diag_(fim_a_unit)

    print(f"\n>> B. Train a second model on {crt_datasets[1]:s} with no restrictions!\n")

    snd_loaders = factory.get_datasets(crt_datasets[1], device, args)

    if args.skip_b:
        print("Skip.")
    else:
        overlaps_b = []

        for idx in range(args.trials_no):
            model_state_dict = torch.load(f"{path:s}_params.pkl")
            optimizer_state_dict = torch.load(f"{path:s}_optim.pkl")
            model.load_state_dict(model_state_dict)
            optimizer.load_state_dict(optimizer_state_dict)

            train(model, optimizer, snd_loaders, crt_datasets[1], args, device=device)
            all_fims = fim_diag(model, snd_loaders[0], samples_no=args.samples_no,
                                empirical=args.empirical, device=device, verbose=True)
            fim_b = all_fims[args.samples_no]

            # save_fisher(fim_b, f"{args.model:s}_fim_B_{idx:d}")

            unit_trace_diag_(fim_b)
            overlap = frechet_diags(fim_a_unit, fim_b)
            print("Overlap: ", overlap)
            overlaps_b.append(overlap)
            del fim_b

    print(f"\n>> C. Train a second model on {crt_datasets[1]:s} with restrictions :)!\n")

    overlaps_c = []

    for idx in range(args.trials_no):
        model_state_dict = torch.load(f"{path:s}_params.pkl")
        optimizer_state_dict = torch.load(f"{path:s}_optim.pkl")
        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)

        train(model, optimizer, snd_loaders, crt_datasets[1], args, device=device,
              old_fim=fim_a)
        all_fims = fim_diag(model, snd_loaders[0], samples_no=args.samples_no,
                            empirical=args.empirical, device=device, verbose=True)
        fim_c = all_fims[args.samples_no]

        # save_fisher(fim_c, f"{args.model:s}_fim_C_{idx:d}")

        unit_trace_diag_(fim_c)
        overlap = frechet_diags(fim_a_unit, fim_c)
        print("Overlap: ", overlap)
        overlaps_c.append(overlap)
        del fim_c

    print("\n>> D. Compute average Frechet distance.")
    if not args.skip_b:
        print("[A-B] Minimum overlap: ", np.min(overlaps_b))
        print("[A-B] Maximum overlap: ", np.max(overlaps_b))
        print("[A-B] Mean    overlap: ", np.mean(overlaps_b))
    print("[A-C] Minimum overlap: ", np.min(overlaps_c))
    print("[A-C] Maximum overlap: ", np.max(overlaps_c))
    print("[A-C] Mean    overlap: ", np.mean(overlaps_c))

    results = dict({})
    if not args.skip_b:
        results["min_overlap_a_b"]: np.min(overlaps_b)
        results["max_overlap_a_b"]: np.max(overlaps_b)
        results["mean_overlap_a_b"]: np.mean(overlaps_b)
        results["improvement"]: np.mean(overlaps_c) - np.mean(overlaps_b)
    results["min_overlap_a_c"]: np.min(overlaps_c)
    results["max_overlap_a_c"]: np.max(overlaps_c)
    results["mean_overlap_a_c"]: np.mean(overlaps_c)

    torch.save(results, get_results_name(args))

    if not args.skip_b:
        print("\nImprovement: ", np.mean(overlaps_c) - np.mean(overlaps_b))

    print("\nFin.")


if __name__ == "__main__":
    main()
