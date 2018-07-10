from typing import Dict

import torch
import torch.nn.functional as Functional
from torch import Tensor


def unit_trace_diag(fim: Dict[str, Tensor]) -> Dict[str, Tensor]:
    trace = sum(t.sum() for t in fim.values())
    return {n: t / trace for (n, t) in fim.items()}


def unit_trace_diag_(fim: Dict[str, Tensor]) -> None:
    trace = sum(t.sum() for t in fim.values()).detach_()
    for t in fim.values():
        t.div_(trace)


def frechet_diags(fim_1: Dict[str, Tensor], fim_2: Dict[str, Tensor]):
    assert len(fim_1) == len(fim_2)
    frechet_norm = 0
    for name, values1 in fim_1.items():
        values2 = fim_2[name]
        diff = (values1.sqrt() - values2.sqrt())
        frechet_norm += (diff * diff).sum()
    return 1 - frechet_norm / 2.0


def cosine_distance(fim_1: Dict[str, Tensor], fim_2: Dict[str, Tensor]):
    return Functional.cosine_similarity(
        torch.cat(tuple([t.view(-1) for t in fim_1.values()])),
        torch.cat(tuple([t.view(-1) for t in fim_2.values()])),
        dim=0
    )


def dot_product(fim_1: Dict[str, Tensor], fim_2: Dict[str, Tensor]):
    return torch.dot(
        torch.cat(tuple([t.view(-1) for t in fim_1.values()])),
        torch.cat(tuple([t.view(-1) for t in fim_2.values()]))
    )


def main():
    import torch.optim as optim

    params = torch.rand(50, requires_grad=True)
    w2 = {"w": torch.rand(50).exp()}
    unit_trace_diag_(w2)
    optimizer = optim.SGD([params], lr=.001, momentum=.99, nesterov=True)
    scale = 10
    for step in range(10000):
        optimizer.zero_grad()
        w1 = {"w": params.exp()}
        w1 = unit_trace_diag(w1)
        overlap = frechet_diags(w1, w2)
        cos = cosine_distance(w1, w2)
        dot = dot_product(w1, w2)

        loss = dot * scale

        loss.backward()
        optimizer.step()
        if (step + 1) % 25 == 0:
            print("Step", step, ":", overlap.item(), cos.item(), dot.item())
    print(w1)


if __name__ == "__main__":
    main()
