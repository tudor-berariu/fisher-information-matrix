import time
import sys
from copy import copy
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from torch import Tensor
from torch.nn import Module
from torch.distributions import Categorical
from torch.utils.data import DataLoader


class CannotChangeDuringFisherEstimation(Exception):
    pass


class GaussianPrior(object):

    def __init__(self,
                 params: Dict[str, Tensor],
                 factors: Dict[Tuple[int, int], List[Tuple[Tensor, Tensor]]],
                 name_to_id: Dict[str, int]):
        self.name_to_id = copy(name_to_id)
        self.mode = dict({})
        for module_name, module_id in name_to_id.items():
            weight = params[module_name + "." + "weight"]
            bias = params[module_name + "." + "bias"]
            out_no = weight.size(0)
            weight, bias = weight.view(out_no, -1), bias.view(out_no, 1)
            self.mode[module_id] = torch.cat([weight, bias], dim=1).detach_()

        self.factors = copy(factors)
        for keys in factors.keys():
            left_id, right_id = keys
            assert left_id in name_to_id.values() and right_id in name_to_id.values()

    def __call__(self, params) -> Tensor:
        diff = {}
        for module_name, module_id in self.name_to_id.items():
            weight = params[module_name + "." + "weight"]
            bias = params[module_name + "." + "bias"]
            out_no = weight.size(0)
            weight, bias = weight.view(out_no, -1), bias.view(out_no, 1)
            param = torch.cat([weight, bias], dim=1)
            diff[module_id] = (param - self.mode[module_id]).detach_()

        total_loss = None
        for (ids, fs) in self.factors.items():
            left_id, right_id = ids
            coeff = 1 if left_id == right_id else 2
            if isinstance(fs["in_cov"], list):
                for (in_cov_t, grad_out_cov) in zip(fs["in_cov"], fs["grad_out_cov"]):
                    print(grad_out_cov.size(), diff[right_id].size(), in_cov_t.size())
                    loss = torch.dot(diff[left_id].view(-1),
                                     (grad_out_cov @ diff[right_id] @ in_cov_t).view(-1))
                    loss *= coeff
                    loss.detach_()
                    total_loss = loss if total_loss is None else (total_loss + loss)
            else:
                in_cov_t, grad_out_cov = fs["in_cov"], fs["grad_out_cov"]
                loss = torch.dot(diff[left_id].view(-1),
                                 (grad_out_cov @ diff[right_id] @ in_cov_t).view(-1))
                loss *= coeff
                loss.detach_()
                total_loss = loss if total_loss is None else (total_loss + loss)

        return total_loss.detach_()


class KFACModule(Module):

    def __init__(self,
                 average_factors: bool=True,
                 tridiag: bool=False):
        super(KFACModule, self).__init__()
        self.__kf_mode = False
        self.__avergage_factors = average_factors
        self.__my_handles = []
        self.__name_to_id = dict({})
        self.__factors = dict({})
        self.__samples_no = dict({})
        self.__inputs = dict({})
        self.__do_checks = True
        self.__conv_special_inputs = dict({})
        self.__tridiag = tridiag
        self.__next_layer_stats = None
        self.__backward_phase = False

    @property
    def kf_mode(self) -> bool:
        return self.__kf_mode

    @property
    def tridiag(self) -> bool:
        return self.__tridiag

    @tridiag.setter
    def tridiag(self, value) -> None:
        self.__tridiag = value

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
        self.__inputs.clear()
        self.__conv_special_inputs.clear()
        self.__backward_phase = False
        self.__next_layer_stats = None

    def start_kf(self):
        self.reset()
        self.__kf_mode = True
        for name, module in self.named_modules():
            if list(module.children()):
                continue
            if not any(p.requires_grad for p in module.parameters()):
                continue
            self.__name_to_id[name] = id(module)
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
        for idid, factors in self.__factors.items():
            factors['in_cov'].div_(self.__samples_no[idid]).transpose_(0, 1)
            factors['grad_out_cov'].div_(self.__samples_no[idid])

        crt_params = {name: param for (name, param) in self.named_parameters()}

        return GaussianPrior(crt_params, self.__factors, self.__name_to_id)

    def end_kf(self):
        factors = self.__extract_factors()
        self.__kf_mode = False
        self.reset()
        return factors

    def conv2d_fwd_hook(self, module: Module, inputs, output) -> None:
        self.__backward_phase = False
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
        assert w_out == w_out_ and ch_out_ == ch_out and b_sz_ == b_sz

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

        self.__inputs[module_id] = x

    def conv2d_bwd_hook(self, module: Module, grad_input, grad_output) -> None:
        module_id = id(module)
        assert isinstance(grad_input, tuple) and len(grad_input) == 3
        assert isinstance(grad_output, tuple) and len(grad_output) == 1
        _dx, dw, _db = grad_input
        dy, = grad_output
        b_sz, ch_out, h_out, w_out = dy.size()
        if self.__tridiag:
            # should do something with dy
            raise NotImplementedError
        dy = dy.view(b_sz, ch_out, -1).transpose(1, 2)\
            .contiguous().view(-1, ch_out)

        if self.__do_checks:
            ch_out_, _ch_in, _k_h, _k_w = module.weight.size()
            assert ch_out == ch_out_
            x = self.__conv_special_inputs[module_id]
            b_sz = dy.size(0)
            ch_out = dy.size(1)
            dw_ = torch.mm(dy.t(), x).view_as(dw)
            assert (dw - dw_).abs().sum().item() < 1e-5

        x = self.__inputs[module_id]
        in_cov = (x.t() @ x).div_(b_sz)
        del self.__inputs[module_id]

        grad_output_cov = (dy.t() @ dy).div_(b_sz * h_out * w_out).detach_()

        idid = module_id, module_id
        self.__samples_no[idid] = self.__samples_no.get(idid, 0) + 1
        if self.average_factors:
            if idid in self.__factors:
                self.__factors[idid]['in_cov'] += in_cov
                self.__factors[idid]['grad_out_cov'] += grad_output_cov
            else:
                self.__factors[idid] = {}
                self.__factors[idid]['in_cov'] = in_cov
                self.__factors[idid]['grad_out_cov'] = grad_output_cov
        else:
            crt_factors = self.__factors.setdefault(idid, dict({}))
            crt_factors.setdefault('in_cov', ()).append(in_cov)
            crt_factors.setdefault('grad_out_cov', ()).append(grad_output_cov)

        if self.__tridiag:
            raise NotImplementedError
            if self.__backward_phase:
                # TODO
                pass
            self.__backward_phase = True

    def linear_fwd_hook(self, module: Module, inputs: Tuple[Tensor], _output) -> None:
        self.__backward_phase = False
        module_id = id(module)
        data, = inputs  # extract from tuple
        b_sz = data.size(0)
        if module.bias is not None:
            data = torch.cat([data, data.new_ones(b_sz, 1)], dim=1)  # add 1s
        self.__inputs[module_id] = data

    def linear_bwd_hook(self, module: Module, _grad_input, grad_output) -> None:
        module_id = id(module)
        self.__samples_no[module_id] = self.__samples_no.get(module_id, 0) + 1
        data = self.__inputs[module_id]
        b_sz = data.size(0)
        in_cov = (data.t() @ data).detach_().div_(b_sz)
        del self.__inputs[module_id]
        dy, = grad_output  # extract from tuple
        grad_output_cov = (dy.t() @ dy).div_(dy.size(0)).detach_()

        idid = module_id, module_id
        self.__samples_no[idid] = self.__samples_no.get(idid, 0) + 1

        if self.average_factors:
            if idid in self.__factors:
                self.__factors[idid]['in_cov'].add_(in_cov).detach_()
                self.__factors[idid]['grad_out_cov'].add_(grad_output_cov).detach_()
            else:
                self.__factors[idid] = {}
                self.__factors[idid]['in_cov'] = in_cov
                self.__factors[idid]['grad_out_cov'] = grad_output_cov
        else:
            crt_factors = self.__factors.setdefault(idid, dict({}))
            crt_factors.setdefault('in_cov', ()).append(in_cov)
            crt_factors.setdefault('grad_out_cov', ()).append(grad_output_cov)

        if self.__tridiag:
            if self.__backward_phase:
                next_id, next_in, next_grad_out = self.__next_layer_stats

                tin_cov = (data.t() @ next_in).detach_().div_(b_sz)
                tgrad_output_cov = (dy.t() @ next_grad_out).div_(dy.size(0)).detach_()

                tidid = module_id, next_id
                self.__samples_no[tidid] = self.__samples_no.get(tidid, 0) + 1

                if self.average_factors:
                    if tidid in self.__factors:
                        self.__factors[tidid]['in_cov'].add_(tin_cov).detach_()
                        self.__factors[tidid]['grad_out_cov'].add_(tgrad_output_cov).detach_()
                    else:
                        self.__factors[tidid] = {}
                        self.__factors[tidid]['in_cov'] = tin_cov
                        self.__factors[tidid]['grad_out_cov'] = tgrad_output_cov
                else:
                    crt_factors = self.__factors.setdefault(tidid, dict({}))
                    crt_factors.setdefault('in_cov', ()).append(tin_cov)
                    crt_factors.setdefault('grad_out_cov', ()).append(tgrad_output_cov)

            self.__next_layer_stats = (module_id, data.clone(), dy.clone().detach_())
            self.__backward_phase = True

    def general_bwd_hook(self, module: Module, _grad_input, _grad_output) -> None:
        module_id = id(module)
        self.__samples_no[module_id] += 1
        raise NotImplementedError


def kfac(model: KFACModule,
         data_loader: DataLoader,
         samples_no: int=None,
         empirical: bool=False,
         device: torch.device=None,
         use_batches: bool=True,
         verbose: bool=False):

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

    gaussian_prior = model.end_kf()

    return gaussian_prior
