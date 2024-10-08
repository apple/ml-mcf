import os
import pickle
import torch
import numpy as np
from copy import deepcopy
from torch.utils.data import Dataset
import torch.nn.functional as F


class DRUGSDataset(Dataset): 
    def __init__(
        self,
        path='./data/',
        train=True, 
        download=True,
        n_eigenfuncs=128,
        mode="train",
        max_confs=10,
        n_molecules=None,
        **kwargs
    ):
        self.n_eigenfuncs = n_eigenfuncs
        self.mode = mode
        self.max_confs = max_confs
        assert self.mode in ["train", "val", "test"]
        self.sub_dir = self.mode
        if self.mode == "test":
            self.sub_dir += "_1000"
        self.path = path
        
        self.all_files = []
        if n_molecules:
            unique_mols = set()
            for fn in os.listdir(os.path.join(self.path, self.sub_dir)):
                if fn.endswith(".pkl"):
                    fn_tag = fn.replace(".pkl", "").split("_")[-2]
                    unique_mols.add(fn_tag)
            unique_mols = list(unique_mols)
            unique_mols.sort()
            print("number of unique molecules:", len(unique_mols))
            n_molecules = min(n_molecules, len(unique_mols))
            sample_mols = unique_mols[:n_molecules]
            sample_mols = set(sample_mols)
            for fn in os.listdir(os.path.join(self.path, self.sub_dir)):
                if fn.endswith(".pkl"):
                    fn_tag = fn.replace(".pkl", "").split("_")[-2]
                    conf_idx = int(fn.split(".")[-2].split("_")[-1])
                    if fn_tag in sample_mols and conf_idx < max_confs:
                        self.all_files.append(fn)
        else:
            for fn in os.listdir(os.path.join(self.path, self.sub_dir)):
                if fn.endswith(".pkl"):
                    conf_idx = int(fn.split(".")[-2].split("_")[-1])
                    if conf_idx < max_confs or self.mode == "test":
                        self.all_files.append(fn)
        print("number of processed molecules:", len(self.all_files))

        self.normalizer = torch.tensor([-16.8567, 16.6408, -10.4640, 10.6586, -7.1461, 7.4427])
        self.min_x, self.max_x, self.min_y, self.max_y, self.min_z, self.max_z = self.normalizer
        print("position normalizer:", self.normalizer)

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, index):
        fn = self.all_files[index]
        conf_idx = fn.split(".")[-2].split("_")[-1]

        f_path = os.path.join(self.path, self.sub_dir, fn)
        with open(f_path, 'rb') as f:
            item = pickle.load(f)
        f.close()
        
        x = item['x']
        mol = item['mol']
        smi = item['smi']
        eig_vecs = item['eig_vecs'][:, :self.n_eigenfuncs]
        pos = deepcopy(item['pos'])
        
        # scale the coordinates to [0, 1]
        pos[:, 0] = (pos[:, 0] - self.min_x) / (self.max_x - self.min_x)
        pos[:, 1] = (pos[:, 1] - self.min_y) / (self.max_y - self.min_y)
        pos[:, 2] = (pos[:, 2] - self.min_z) / (self.max_z - self.min_z)
        # scale the coordinates to [-1, 1]
        pos = pos * 2 - 1
        # CoM
        pos = pos - pos.mean(dim=0, keepdim=True)

        num_nodes = x.shape[0]
        if num_nodes <= self.n_eigenfuncs:
            # padding by zeros
            zero_vec = torch.zeros(num_nodes, self.n_eigenfuncs - num_nodes + 1)
            zero_vec = torch.clamp(zero_vec, min=1e-8)
            eig_vecs = torch.cat((eig_vecs, zero_vec), dim = 1)

        coord = torch.cat([eig_vecs, x, pos], dim=1)

        return coord.float(), mol, self.normalizer, smi, conf_idx