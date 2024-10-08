import os
import sys
import torch
import importlib.util
import importlib


def modelrunner(ckpt_path):
    spec = importlib.util.spec_from_file_location(
        "MCF", os.path.join("models", "mcf.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    checkpoint = torch.load(ckpt_path)
    
    opt = checkpoint["opt"]

    opt.device = "cuda"
    model = module.MCF.load_from_checkpoint(ckpt_path)

    model.eval()

    return model, opt
