import argparse
import importlib
import torch
from einops import rearrange
import io
import math
from copy import deepcopy
from torch import nn
import math
from inspect import isfunction


##########################################################
def make_beta_schedule(
    schedule, num_timesteps, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3
):
    if schedule == "quad":
        betas = (
            torch.linspace(
                linear_start**0.5,
                linear_end**0.5,
                num_timesteps,
                dtype=torch.float64,
            )
            ** 2
        )

    elif schedule == "linear":
        betas = torch.linspace(
            linear_start, linear_end, num_timesteps, dtype=torch.float64
        )

    elif schedule == "cosine":
        timesteps = (
            torch.arange(num_timesteps + 1, dtype=torch.float64) / num_timesteps
            + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)

    return betas


def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)

    return out


def noise_like(shape, noise_fn, device, repeat=False):
    if repeat:
        resid = [1] * (len(shape) - 1)
        shape_one = (1, *shape[1:])

        return noise_fn(*shape_one, device=device).repeat(shape[0], *resid)

    else:
        return noise_fn(*shape, device=device)


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


##########################################################
class ModelEmaV2(nn.Module):
    def __init__(self, model, decay=0.9999, device=None):
        super(ModelEmaV2, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(
                self.module.state_dict().values(), model.state_dict().values()
            ):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(
            model, update_fn=lambda e, m: self.decay * e + (1.0 - self.decay) * m
        )

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


##########################################################
def convert_to_coord_format(h, w, device="cpu"):
    x_channel = torch.linspace(0, 1, w, device=device).view(1, 1, -1).repeat(1, w, 1)
    y_channel = torch.linspace(0, 1, h, device=device).view(1, -1, 1).repeat(1, 1, h)
    return torch.cat((x_channel, y_channel), dim=0)


def get_random_coordinates(resolution=64, npoints=1024, ndim=2):
    coord = convert_to_coord_format(resolution, resolution)
    coord = rearrange(coord, "c h w -> (h w) c")
    sampled_indices = torch.randperm(coord.shape[0])[:npoints]
    coord = coord[sampled_indices, :]
    return coord, sampled_indices


def get_obj_from_str(string):
    # From https://github.com/CompVis/taming-transformers
    module, cls = string.rsplit(".", 1)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    # From https://github.com/CompVis/taming-transformers
    if "target" not in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
