import os
import argparse
import glob
import pickle
import scipy
import math
import torch
import random
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, get_laplacian 
from multiprocess import Pool

import torch.nn.functional as F
from torch_scatter import scatter

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import ChiralType


results = {}
jobs = []
failures = []

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--dataset',
                default='qm9',
                const='qm9',
                nargs='?',
                choices=['qm9', 'drugs'],
                help='dataset to process (default: %(default)s)'
                )
parser.add_argument('--max_confs', type=int, default=20)
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--n_workers', type=int, default=32)

args = parser.parse_args()
dataset = args.dataset
mode = args.mode
max_confs = args.max_confs
n_workers = args.n_workers

if dataset == 'qm9':
    types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
elif dataset == 'drugs':
    types = {
        'H': 0, 'Li': 1, 'B': 2, 'C': 3, 'N': 4, 'O': 5, 'F': 6, 'Na': 7, 'Mg': 8, 'Al': 9, 'Si': 10,
        'P': 11, 'S': 12, 'Cl': 13, 'K': 14, 'Ca': 15, 'V': 16, 'Cr': 17, 'Mn': 18, 'Cu': 19, 'Zn': 20,
        'Ga': 21, 'Ge': 22, 'As': 23, 'Se': 24, 'Br': 25, 'Ag': 26, 'In': 27, 'Sb': 28, 'I': 29, 'Gd': 30,
        'Pt': 31, 'Au': 32, 'Hg': 33, 'Bi': 34
    }

chirality = {
    ChiralType.CHI_TETRAHEDRAL_CW: -1.,
    ChiralType.CHI_TETRAHEDRAL_CCW: 1.,
    ChiralType.CHI_UNSPECIFIED: 0,
    ChiralType.CHI_OTHER: 0
}
bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}


def one_k_encoding(value, choices):
    """
    Creates a one-hot encoding with an extra category for uncommon values.
    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the :code:`value` in a list of length :code:`len(choices) + 1`.
             If :code:`value` is not in :code:`choices`, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding


def featurize_mol(mol_dic ,bonds, types, mode, max_confs=None):
    confs = mol_dic['conformers']
    random.shuffle(confs)  # shuffle confs
    name = mol_dic["smiles"]
    if max_confs is not None:
        max_confs = min(max_confs, len(confs))

    # filter mols rdkit can't intrinsically handle
    mol_ = Chem.MolFromSmiles(name)
    if mol_:
        canonical_smi = Chem.MolToSmiles(mol_)
    else:
        return None

    # skip conformers with fragments
    if '.' in name:
        return None

    # skip conformers without dihedrals
    N = confs[0]['rd_mol'].GetNumAtoms()

    pos = torch.zeros([max_confs, N, 3])
    pos_mask = torch.zeros(max_confs, dtype=torch.int64)
    k = 0
    mols_list = []
    for conf in confs:
        mol = conf['rd_mol']

        # skip mols with atoms with more than 4 neighbors for now
        n_neighbors = [len(a.GetNeighbors()) for a in mol.GetAtoms()]
        if np.max(n_neighbors) > 4:
            continue

        pos[k] = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float)
        pos_mask[k] = 1
        k += 1
        correct_mol = mol
        mols_list.append(mol)
        if k == max_confs:
            break

    # return None if no non-reactive conformers were found
    if k == 0:
        return None

    type_idx = []
    atomic_number = []
    atom_features = []
    chiral_tag = []
    neighbor_dict = {}
    ring = correct_mol.GetRingInfo()
    for i, atom in enumerate(correct_mol.GetAtoms()):
        type_idx.append(types[atom.GetSymbol()])
        n_ids = [n.GetIdx() for n in atom.GetNeighbors()]
        if len(n_ids) > 1:
            neighbor_dict[i] = torch.tensor(n_ids)
        chiral_tag.append(chirality[atom.GetChiralTag()])
        atomic_number.append(atom.GetAtomicNum())
        atom_features.append(1 if atom.GetIsAromatic() else 0)
        atom_features.extend(one_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6]))
        atom_features.extend(one_k_encoding(atom.GetHybridization(), [
                                Chem.rdchem.HybridizationType.SP,
                                Chem.rdchem.HybridizationType.SP2,
                                Chem.rdchem.HybridizationType.SP3,
                                Chem.rdchem.HybridizationType.SP3D,
                                Chem.rdchem.HybridizationType.SP3D2]))
        atom_features.extend(one_k_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]))
        atom_features.extend(one_k_encoding(atom.GetFormalCharge(), [-1, 0, 1]))
        atom_features.extend([int(ring.IsAtomInRingOfSize(i, 3)),
                                int(ring.IsAtomInRingOfSize(i, 4)),
                                int(ring.IsAtomInRingOfSize(i, 5)),
                                int(ring.IsAtomInRingOfSize(i, 6)),
                                int(ring.IsAtomInRingOfSize(i, 7)),
                                int(ring.IsAtomInRingOfSize(i, 8))])
        atom_features.extend(one_k_encoding(int(ring.NumAtomRings(i)), [0, 1, 2, 3]))

    z = torch.tensor(atomic_number, dtype=torch.long)
    chiral_tag = torch.tensor(chiral_tag, dtype=torch.float)

    row, col, edge_type, bond_features = [], [], [], []
    for bond in correct_mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        row += [start, end]
        col += [end, start]
        edge_type += 2 * [bonds[bond.GetBondType()]]
        bt = tuple(sorted([bond.GetBeginAtom().GetAtomicNum(), bond.GetEndAtom().GetAtomicNum()])), bond.GetBondTypeAsDouble()
        bond_features += 2 * [int(bond.IsInRing()),
                                int(bond.GetIsConjugated()),
                                int(bond.GetIsAromatic())]

    edge_index = torch.tensor([row, col], dtype=torch.long)
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    edge_attr = F.one_hot(edge_type, num_classes=len(bonds)).to(torch.float)

    perm = (edge_index[0] * N + edge_index[1]).argsort()
    edge_index = edge_index[:, perm]
    edge_type = edge_type[perm]
    edge_attr = edge_attr[perm]

    row, col = edge_index
    hs = (z == 1).to(torch.float)
    num_hs = scatter(hs[row], col, dim_size=N).tolist()

    x1 = F.one_hot(torch.tensor(type_idx), num_classes=len(types))
    x2 = torch.tensor(atom_features).view(N, -1)
    x = torch.cat([x1.to(torch.float), x2], dim=-1)


    data = Data(x=x, z=z, pos=[pos], edge_index=edge_index, edge_attr=edge_attr, neighbors=neighbor_dict,
                chiral_tag=chiral_tag, name=name, boltzmann_weight=conf['boltzmannweight'],
                degeneracy=conf['degeneracy'], mol=mols_list, pos_mask=pos_mask)
    return data


def process_data(f_path, save_dir, mode, max_confs=20):
    with open(f_path, "rb") as f:
        mol_dic = pickle.load(f)
    smi = mol_dic["smiles"]
    item = featurize_mol(mol_dic, bonds, types, mode, max_confs)

    if item != None:
        ones = (item.pos_mask == 1).sum(dim=0)
        k_eig_vec, k_eig_val = calculate_eigenfuncs(item.edge_index, item.num_nodes)
        if k_eig_vec is None or k_eig_val is None:
            return

        for j in range(ones.item()):
            data = Data(x=item.x, pos=item.pos[0][j], smi=smi,
                edge_index=item.edge_index, edge_attr=item.edge_attr,
                eig_vecs = k_eig_vec, eig_vals = k_eig_val, mol=item.mol)

            smi_name = smi.replace('/', '_')
            smi_name = smi_name[:245]
            save_path = os.path.join(save_dir, f"{smi_name}_{j}.pkl")
            with open(save_path, 'wb') as f:
                pickle.dump(data, f)
    return


def calculate_eigenfuncs(edge_index, num_nodes):
    adj = to_dense_adj(edge_index)[0]
    node_deg_vec = adj.sum(axis=1, keepdim=True)
    node_deg_mat = torch.diag(node_deg_vec[:, 0])
    lap_mat = get_laplacian(edge_index)
    L = to_dense_adj(edge_index = lap_mat[0], edge_attr = lap_mat[1])[0]

    # calculate the  eigenvalues and the eigenvectors for the lap_matrix
    w = scipy.linalg.eigh(L.cpu(), b = node_deg_mat.cpu())
    eigenvalues, eigenvectors = w[0], w[1]

    # calculate the k lowers eigenvalues and the corresponding eigenvectors
    eigenvectors = eigenvectors[:, eigenvalues.argsort()]  # increasing order
    k_eig_vec = torch.from_numpy(np.real(eigenvectors[:, 1:])).float()
    eigenvalues.sort()
    k_eig_val = torch.from_numpy(np.real(eigenvalues[1:])).float()

    # normalize eigenvecs and eigenvals
    k_eig_vec = k_eig_vec * math.sqrt(num_nodes)
    k_eig_val = k_eig_val * math.sqrt(num_nodes)

    try:
        assert k_eig_vec.shape[0] == num_nodes
    except:
        print("Invalid eigenvectors!", "k_eig_vec:", k_eig_vec.shape, "num_nodes", num_nodes)
        return None, None
    
    return k_eig_vec, k_eig_val


def worker_fn(job):
    idx, f_path, save_dir, mode, max_confs = job
    print(f"Processing {idx}th file")
    process_data(f_path, save_dir, mode, max_confs)
    return


def populate_results(res):
    pass


path = f'data/rdkit_folder/{dataset}'
split_path = f'data/rdkit_folder/{dataset}/split0.npy'
save_dir = f'data/processed_{dataset}/{mode}'
os.makedirs(save_dir, exist_ok=True)

split_idx = 0 if mode == 'train' else 1 if mode == 'val' else 2
split = np.load(split_path, allow_pickle=True)[split_idx]

all_files = sorted(glob.glob(os.path.join(path, '*.pickle')))
pickle_files = [f for i, f in enumerate(all_files) if i in split]
print(f"Number of files: {len(pickle_files)}")

jobs = [(idx, f, save_dir, mode, max_confs) for idx, f in enumerate(pickle_files)]

p = Pool(n_workers)
map_fn = p.imap_unordered
p.__enter__()

for res in map_fn(worker_fn, jobs):
    populate_results(res)

p.close()