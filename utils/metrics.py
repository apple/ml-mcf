import torch
import numpy as np
import torch.nn as nn
from torch import Tensor

from tqdm import trange
import torch.nn.functional as F
from typing import List

from copy import deepcopy
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolAlign
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule
from rdkit.Geometry import Point3D


def set_rdmol_positions(mol, pos):
    """
    Args:
        rdkit_mol:  An `rdkit.Chem.rdchem.Mol` object.
        pos: (N_atoms, 3)
    """
    mol_ = deepcopy(mol)
    for i in range(pos.shape[0]):
        # mol_.GetConformer().SetAtomPosition(i, pos[i].tolist())
        conf = mol_.GetConformers()[0]
        x, y, z = pos[i].tolist()
        conf.SetAtomPosition(i, Point3D(x, y, z))
    return mol_


def calc_rmsd(mol, ref_mol):
    mol = Chem.RemoveHs(mol)
    ref_mol = Chem.RemoveHs(ref_mol)
    try:
        rmsd = rdMolAlign.GetBestRMS(mol, ref_mol)
    except:
        # print("Can't match molecules!", mol, ref_mol)
        rmsd = None
    return rmsd


def calc_confution_mat(sampled_mols, gt_mols, useFF=False):
    n_confs = len(gt_mols)
    n_samples = len(sampled_mols)
    rmsd_confusion_mat = np.nan * np.ones([n_confs, n_samples],dtype=np.float64)

    for i in range(n_samples):
        if useFF:
            #print('Applying FF on generated molecules...')
            MMFFOptimizeMolecule(sampled_mols[i])

        for j in range(n_confs):
            mol_ref = deepcopy(gt_mols[j])
            mol_sampled = deepcopy(sampled_mols[i])
            rmsd_confusion_mat[j,i] = calc_rmsd(mol_sampled, mol_ref)
    return rmsd_confusion_mat


def molecule_gen_metric(sample_mols_list, gt_mols_list, threshold=0.5):
    b = len(gt_mols_list)
    rmsd_confusion_matrices = []
    covr_scores = []
    matr_scores = []
    covp_scores = []
    matp_scores = []

    for i in range(b):
        gt_mols = gt_mols_list[i]
        sampled_mols = sample_mols_list[i]
        rmsd_conf_mat = calc_confution_mat(sampled_mols, gt_mols)

        if np.isnan(np.sum(rmsd_conf_mat)):
            print("NAN in RMSD confusion matrix!")

        rmsd_ref_min = np.nanmin(rmsd_conf_mat, axis=-1)    # np (num_ref, )
        rmsd_gen_min = np.nanmin(rmsd_conf_mat, axis=0)     # np (num_gen, )
        rmsd_cov_thres = rmsd_ref_min.reshape(-1, 1) <= threshold # np (num_ref, )
        rmsd_jnk_thres = rmsd_gen_min.reshape(-1, 1) <= threshold # np (num_gen, )

        matr_scores.append(rmsd_ref_min.mean())
        covr_scores.append(rmsd_cov_thres.mean())
        matp_scores.append(rmsd_gen_min.mean())
        covp_scores.append(rmsd_jnk_thres.mean())
        rmsd_confusion_matrices.append(rmsd_conf_mat)
    
    # covr_scores = np.vstack(covr_scores)  # np (num_mols, num_thres)
    covr_scores = np.array(covr_scores)   # np (num_mols, )
    matr_scores = np.array(matr_scores)   # np (num_mols, )
    # covp_scores = np.vstack(covp_scores)  # np (num_mols, num_thres)
    covp_scores = np.array(covp_scores)   # np (num_mols, )
    matp_scores = np.array(matp_scores)   # np (num_mols, )

    return rmsd_confusion_matrices, covr_scores, matr_scores, covp_scores, matp_scores
