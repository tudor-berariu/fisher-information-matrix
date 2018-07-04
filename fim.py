import time
import sys

import torch
from torch.distributions import Categorical
from torch.nn import Module
from torch.utils.data import DataLoader

assert int(torch.__version__.split(".")[1]) >= 4, "PyTorch 0.4+ required"


def fim_diag(model: Module,
             data_loader: DataLoader,
             samples_no: int,
             device: torch.device,
             verbose: bool = False):
    fim = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fim[name] = torch.zeros_like(param)

    seen_no = 0
    last = 0
    tic = time.time()

    while seen_no < samples_no:
        data_iterator = iter(data_loader)
        try:
            data, _ = next(data_iterator)
        except StopIteration:
            data_iterator = iter(data_loader)
            data, _ = next(data_loader)

        data = data.to(device)
        idx, batch_size = 0, data.size(0)
        while idx < batch_size and seen_no < samples_no:

            model.zero_grad()
            logits = model(data[idx:idx+1])
            sample = Categorical(logits=logits).sample().unsqueeze(1).detach()
            logits.gather(1, sample).backward()
            for name, param in model.named_parameters():
                if param.requires_grad:
                    fim[name] += (param.grad * param.grad)
                    fim[name].detach()

            seen_no += 1

            if verbose and seen_no % 2000 == 0:
                toc = time.time()
                fps = float(seen_no - last) / (toc - tic)
                tic, last = toc, seen_no
                sys.stdout.write(f"\rSamples: {seen_no:d}. Fps: {fps:2.4f} samples/s.")

    if verbose:
        toc = time.time()
        fps = float(seen_no - last) / (toc - tic)
        sys.stdout.write(f"\rSamples: {seen_no:d}. Fps: {fps:2.5f} s.\n")

    for name, grad2 in fim.items():
        grad2 /= float(seen_no)

    return fim
