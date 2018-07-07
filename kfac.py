import time
import sys
from copy import copy
from typing import Tuple

import torch
import torch.nn as nn

from torch import Tensor
from torch.nn import Module
from torch.distributions import Categorical
from torch.utils.data import DataLoader


class CannotChangeDuringFisherEstimation(Exception):
    pass


class KFACModule(Module):

    def __init__(self, average_factors: bool = True):
        super(KFACModule, self).__init__()
        self.__kf_mode = False
        self.__avergage_factors = average_factors
        self.__my_handles = []
        self.__name_to_id = dict({})
        self.__factors = dict({})
        self.__samples_no = dict({})
        self.__inputs_cov = dict({})
        self.__do_checks = True
        self.__conv_special_inputs = dict({})

    @property
    def kf_mode(self) -> bool:
        return self.__kf_mode

    @property
    def average_factors(self) -> bool:
        return self.__avergage_factors

    @average_factors.setter
    def average_factors(self, value: bool) -> None:
        if self.__kf_mode:
            raise CannotChangeDuringFisherEstimation
        self.__avergage_factors = bool(value)

    def reset(self):
        for handle in self.__my_handles:
            handle.remove()
        self.__my_handles.clear()
        self.__factors.clear()
        self.__name_to_id.clear()
        self.__samples_no.clear()
        self.__inputs_cov.clear()
        self.__conv_special_inputs.clear()

    def start_kf(self):
        self.reset()
        self.__kf_mode = True
        for name, module in self.named_modules():
            if list(module.children()):
                continue
            if not any(p.requires_grad for p in module.parameters()):
                continue
            self.__name_to_id[name] = id(module)
            self.__samples_no[id(module)] = 0
            if isinstance(module, nn.Conv2d):
                self.__my_handles.extend([
                    module.register_forward_hook(self.conv2d_fwd_hook),
                    module.register_backward_hook(self.conv2d_bwd_hook)
                ])
            elif isinstance(module, nn.Linear):
                self.__my_handles.extend([
                    module.register_forward_hook(self.linear_fwd_hook),
                    module.register_backward_hook(self.linear_bwd_hook)
                ])
            else:
                self.__my_handles.append(
                    module.register_backward_hook(self.general_bwd_hook)
                )

    def __extract_factors(self):
        for module_id, factors in self.__factors.items():
            factors['in_cov'].div_(self.samples_no[module_id])
            factors['out_grad_cov'].div_(self.samples_no[module_id])
        return copy(self.__factors)

    def end_kf(self):
        factors = self.__extract_factors()
        self.__kf_mode = False
        self.reset()
        return factors

    def conv2d_fwd_hook(self, module: Module, inputs, output) -> None:
        module_id = id(module)

        assert isinstance(inputs, tuple) and len(inputs) == 1
        assert isinstance(output, Tensor)
        inputs, = inputs

        ch_out, ch_in, k_h, k_w = module.weight.size()
        s_h, s_w = module.stride
        b_sz, ch_in_, h_in, w_in = inputs.size()
        h_out = (h_in - k_h + 0) // s_h + 1
        w_out = (w_in - k_w + 0) // s_w + 1
        b_sz_, ch_out_, h_out_, w_out_ = output.size()

        assert ch_in_ == ch_in
        assert h_out_ == h_out
        assert w_out == w_out_ and \
            ch_out_ == ch_out and b_sz_ == b_sz

        x = inputs.new().resize_(b_sz, h_out, w_out, ch_in, k_h, k_w)
        for idx_h in range(0, h_out):
            start_h = idx_h * s_h
            for idx_w in range(0, w_out):
                start_w = idx_w * s_w
                x[:, idx_h, idx_w, :, :, :].copy_(
                    inputs[:, :, start_h:(start_h + k_h), start_w:(start_w + k_w)]
                )

        x = x.view(b_sz * h_out * w_out, ch_in * k_h * k_w)
        if self.__do_checks:
            # Keep them until bwd pass
            self.__conv_special_inputs[module_id] = x

        x = torch.cat([x, x.new_ones(b_sz * h_out * w_out, 1)], dim=1)

        if self.__do_checks:
            weight_extra = torch.cat([module.weight.view(ch_out, -1),
                                      module.bias.view(ch_out, -1)], dim=1)
            y = (x @ weight_extra.t()).view(b_sz, h_out * w_out, ch_out)\
                                      .transpose(1, 2)\
                                      .view(b_sz, ch_out, h_out, w_out)
            assert (y - output).abs().max() < 1e-5  # assert torch.allclose(y, output)

        inputs_cov = (x.t() @ x).div_(b_sz)
        self.__inputs_cov[module_id] = (x.t() @ x).div_(b_sz)

    def conv2d_bwd_hook(self, module: Module, grad_input, grad_output) -> None:
        module_id = id(module)
        self.__samples_no[module_id] += 1
        assert isinstance(grad_input, tuple) and len(grad_input) == 3
        assert isinstance(grad_output, tuple) and len(grad_output) == 1
        dx, dw, db = grad_input
        dy, = grad_output
        b_sz, ch_out, h_out, w_out = dy.size()
        dy = dy.view(b_sz, ch_out, -1).transpose(1, 2)\
                                      .contiguous().view(-1, ch_out)

        if self.__do_checks:
            ch_out_, ch_in, _k_h, _k_w = module.weight.size()
            assert ch_out == ch_out_
            x = self.__conv_special_inputs[module_id]
            b_sz = dy.size(0)
            ch_out = dy.size(1)
            dw_ = torch.mm(dy.t(), x).view_as(dw)
            assert (dw - dw_).sum().item() < 1e-5

        in_cov = self.__inputs_cov[module_id]
        del self.__inputs_cov[module_id]

        grad_output_cov = (dy.t() @ dy).div_(b_sz * h_out * w_out).detach_()

        if self.average_factors:
            if module_id in self.__factors:
                self.__factors[module_id]['in_cov'] += in_cov
                self.__factors[module_id]['grad_out_cov'] += grad_output_cov
            else:
                self.__factors[module_id] = {}
                self.__factors[module_id]['in_cov'] = in_cov
                self.__factors[module_id]['grad_out_cov'] = grad_output_cov
        else:
            crt_factors = self.__factors.setdefault(module_id, dict({}))
            crt_factors.setdefault('in_cov', ()).append(in_cov)
            crt_factors.setdefault('grad_out_cov', ()).append(grad_output_cov)

    def linear_fwd_hook(self, module: Module, inputs: Tuple[Tensor], _output) -> None:
        module_id = id(module)
        data, = inputs  # extract from tuple
        b_sz = data.size(0)
        if module.bias is not None:
            data = torch.cat([data, data.new_ones(b_sz, 1)], dim=1)  # add 1s
        self.__inputs_cov[module_id] = (data.t() @ data).detach_().div_(b_sz)

    def linear_bwd_hook(self, module: Module, grad_input, grad_output) -> None:
        module_id = id(module)
        self.__samples_no[module_id] += 1
        in_cov = self.__inputs_cov[module_id]
        del self.__inputs_cov[module_id]
        dy, = grad_output  # extract from tuple
        grad_output_cov = (dy.t() @ dy).div_(dy.size(0)).detach_()

        if self.average_factors:
            if module_id in self.__factors:
                self.__factors[module_id]['in_cov'].add_(in_cov).detach_()
                self.__factors[module_id]['grad_out_cov'].add_(grad_output_cov).detach_()
            else:
                self.__factors[module_id] = {}
                self.__factors[module_id]['in_cov'] = in_cov
                self.__factors[module_id]['grad_out_cov'] = grad_output_cov
        else:
            crt_factors = self.__factors.setdefault(module_id, dict({}))
            crt_factors.setdefault('in_cov', ()).append(in_cov)
            crt_factors.setdefault('grad_out_cov', ()).append(grad_output_cov)

    def general_bwd_hook(self, module: Module, grad_input, grad_output) -> None:
        module_id = id(module)
        self.__samples_no[module_id] += 1
        raise NotImplementedError


def kfac(model: KFACModule,
         data_loader: DataLoader,
         samples_no: int = None,
         empirical: bool = False,
         device: torch.device = None,
         use_batches: bool = True,
         verbose: bool = False):

    if verbose:
        print("Start KFAC.")

    model.start_kf()

    seen_no = 0
    last = 0
    tic = time.time()

    while not samples_no or seen_no < samples_no:
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

        if use_batches:
            logits = model(data)
            if empirical:
                outdx = target.unsqueeze(1)
            else:
                outdx = Categorical(logits=logits).sample().unsqueeze(1).detach()
            samples = logits.gather(1, outdx)
            model.zero_grad()
            torch.autograd.backward(samples.mean(), retain_graph=True)
            seen_no += samples.size(0)

            if verbose and seen_no - last >= 100:
                toc = time.time()
                fps = float(seen_no - last) / (toc - tic)
                tic, last = toc, seen_no
                sys.stdout.write(f"\rSamples: {seen_no:5d}. Fps: {fps:2.4f} samples/s.")
        else:
            idx, batch_size = 0, data.size(0)
            while idx < batch_size and (not samples_no or seen_no < samples_no):
                logits = model(data[idx:idx+1])
                if empirical:
                    outdx = target[idx:idx+1].unsqueeze(1)
                else:
                    outdx = Categorical(logits=logits).sample().unsqueeze(1).detach()
                sample = logits.gather(1, outdx)
                model.zero_grad()
                torch.autograd.backward(sample, retain_graph=True)
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

    factors = model.end_kf()
    return factors
