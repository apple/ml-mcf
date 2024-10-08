import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from datasets.geom_qm9 import QM9Dataset
from datasets.geom_drugs import DRUGSDataset
from datasets.geom_xl import XLDataset


def mol_collate_fn(batch):
    coords, mols, normalizer, smiles, conf_indices = zip(*batch)
    n_samples = len(coords)
    coord_dim = coords[0].shape[1]
    max_num_atoms = max([coord.shape[0] for coord in coords])
    padded_coords = torch.zeros(n_samples, max_num_atoms, coord_dim)
    mask = torch.zeros(n_samples, max_num_atoms)
    num_atoms = []
    for i, coord in enumerate(coords):
        padded_coords[i, :coord.shape[0], :] = coord
        mask[i, :coord.shape[0]] = 1
        num_atoms.append(coord.shape[0])
    return padded_coords, mask, mols, normalizer[0], smiles, conf_indices, num_atoms


def build_dataloader(data_config, verbose=True):
    datasets = {
        "geom_qm9": QM9Dataset,
        "geom_drugs": DRUGSDataset,
        'geom_xl': XLDataset,
    }

    if "train_set_config" not in data_config.keys():
        data_config.train_set_config = {}

    if "val_set_config" not in data_config.keys():
        data_config.val_set_config = {}

    data_loader_args = [
        "batch_size",
        "shuffle",
        "num_workers",
        "drop_last",
        "pin_memory",
        "persistent_workers"
    ]
    train_loader_defaults = {
        "shuffle": True,
        "num_workers": 1,
        "drop_last": True,
        "pin_memory": True,
        "persistent_workers": True,
    }
    val_loader_defaults = {
        "shuffle": False,
        "num_workers": 1,
        "drop_last": False,
        "pin_memory": True,
        "persistent_workers": True,
    }

    # combine all configs (configs on right have priority if configs share args)
    train_set_config = {
        **train_loader_defaults,
        **data_config,
        **data_config.train_set_config,
    }
    train_set_config = {
        k: train_set_config[k]
        for k in train_set_config
        if k not in {"train_set_config", "val_set_config"}
    }
    train_set = datasets[data_config.dataset](**train_set_config)
    # get only the args for the DataLoader class, since it can't deal with extra args
    train_loader_config = {k: train_set_config[k] for k in data_loader_args}
    if data_config.dataset in ["geom_qm9", "geom_drugs", "geom_xl", "geom_qm9_rot"]:
        # collate function to return both coordinates and molecules (list of RDkit Mol Objects)
        train_loader_config["collate_fn"] = mol_collate_fn
    train_loader = DataLoader(dataset=train_set, **train_loader_config)

    val_set_config = {
        **val_loader_defaults,
        **data_config,
        **data_config.val_set_config,
    }
    val_set_config = {
        k: val_set_config[k]
        for k in val_set_config
        if k not in {"train_set_config", "val_set_config"}
    }
    val_set = datasets[data_config.dataset](**val_set_config)
    val_loader_config = {k: val_set_config[k] for k in data_loader_args}
    if data_config.dataset in ["geom_qm9", "geom_drugs", "geom_xl", "geom_qm9_rot"]:
        # collate function to return both coordinates and molecules (list of RDkit Mol Objects)
        val_loader_config["collate_fn"] = mol_collate_fn
    if "nsamples" in val_set_config.keys() and val_set_config["nsamples"] < len(val_set):
        val_set = Subset(val_set, list(range(0, val_set_config["nsamples"])))
    val_loader = DataLoader(dataset=val_set, **val_loader_config)

    if verbose:
        print("")
        print("----------- Train Set Config -----------\n")
        print(OmegaConf.to_yaml(train_set_config))
        print("------------ Val Set Config ------------\n")
        print(OmegaConf.to_yaml(val_set_config))
        print("----------------- End ------------------\n")
        print("")

    data_module = DataModule(train_loader=train_loader, val_loader=val_loader)
    return data_module


class DataModule(pl.LightningDataModule):
    def __init__(self, train_loader, val_loader=None, test_loader=None):
        super().__init__()

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader
