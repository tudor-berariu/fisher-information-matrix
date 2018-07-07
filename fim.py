import time
import sys

import torch
from torch.distributions import Categorical
from torch.nn import Module
from torch.utils.data import DataLoader

assert int(torch.__version__.split(".")[1]) >= 4, "PyTorch 0.4+ required"


def fim_diag(model: Module,
             data_loader: DataLoader,
             samples_no: int = None,
             empirical: bool = False,
             device: torch.device = None,
             verbose: bool = False):
    fim = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fim[name] = torch.zeros_like(param)

    seen_no = 0
    last = 0
    tic = time.time()

    while samples_no is None or seen_no < samples_no:
        data_iterator = iter(data_loader)
        try:
            data, target = next(data_iterator)
        except StopIteration:
            if samples_no is None:
                break
            data_iterator = iter(data_loader)
            data, target = next(data_loader)

        if device is not None:
            data = data.to(device)
            if empirical:
                target = target.to(device)

        logits = model(data)
        if empirical:
            outdx = target.unsqueeze(1)
        else:
            outdx = Categorical(logits=logits).sample().unsqueeze(1).detach()
        samples = logits.gather(1, outdx)

        idx, batch_size = 0, data.size(0)
        while idx < batch_size and (samples_no is None or seen_no < samples_no):
            model.zero_grad()
            torch.autograd.backward(samples[idx], retain_graph=True)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    fim[name] += (param.grad * param.grad)
                    fim[name].detach_()
            seen_no += 1
            idx += 1

            if verbose and seen_no % 100 == 0:
                toc = time.time()
                fps = float(seen_no - last) / (toc - tic)
                tic, last = toc, seen_no
                sys.stdout.write(f"\rSamples: {seen_no:5d}. Fps: {fps:2.4f} samples/s.")

    if verbose:
        if seen_no > last:
            toc = time.time()
            fps = float(seen_no - last) / (toc - tic)
        sys.stdout.write(f"\rSamples: {seen_no:5d}. Fps: {fps:2.5f} samples/s.\n")

    for name, grad2 in fim.items():
        grad2 /= float(seen_no)

    return fim
