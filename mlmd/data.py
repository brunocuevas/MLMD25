import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import MDAnalysis as mda
from typing import Generator, List
import numpy as np
import logging


def trajectory_to_data(
        u: mda.Universe, selection_string: str, 
        residue_index, atom_type_index, label
):
    """
    Args:
    
        u: mdanalysis universe containing a trajectory, as obtained from MDAnalysis
        selection_string: e.g. "name CA"
        residue_index: a dict mapping each residue with a number
        atom_type_index: a dict mapping each atom type with a number
        label: a numeric label for each prediction class (e.g. 0, 1, ...)
    """
    out = []
    for fr in u.trajectory:
        x = torch.from_numpy(u.select_atoms(selection_string).positions)
        residues = list(map(lambda x: residue_index[x], u.select_atoms(selection_string).resnames))
        atom_types = list(map(lambda x: atom_type_index[x], u.select_atoms(selection_string).atoms.types.tolist()))
        d = Data(pos=x, res=torch.tensor(residues, dtype=torch.long), types=torch.tensor(atom_types, dtype=torch.long), label=torch.tensor(label, dtype=torch.long))
        out.append(d)
    return out


def split_trajectory(trj: List[Data], index=List[int])->List[List[Data]]:
    """
    Args: 
        trj:  as obtained from `trajectory_to_data`.
        index: it indicates the location of each frame

    Notes:

        It places each frame in the n position of the output,
        allowing a flexible split of the data.
    """
    index_values = np.unique(index)
    if 1 + max(index_values) != len(index_values):
        raise IOError("the results cannot have empty frames")
    out = []
    for i in index_values:
        out.append([])

    for i, j in enumerate(index):
        out[j].append(trj[i])
    return out



def trajectories_to_data(data_files, selection_string, residue_index, atom_type_index):
    """
    """
    trajectories = []
    path = data_files['path']
    logging.error(path)
    for file in data_files['trajectories']:
        logging.error("reading {0}".format(file['topology']))
        u = mda.Universe(
            path + '/' + file['topology'], 
            path + '/' + file['trajectory']
        )
        trajectories.append(list(trajectory_to_data(u, selection_string, residue_index, atom_type_index, file['label'])))
    return trajectories

def random_split_index(n, test_fraction) -> List[int]:
    print("foo!")
    u = np.random.rand(n)
    v = np.zeros_like(u)
    v[u > test_fraction] = 1
    return v.astype(int).tolist()

def start_end_split_index(n, test_fraction) -> List[int]:
    u = np.arange(n) / n
    v = np.zeros(n)
    v[u > (1 - test_fraction)] = 1
    return v.astype(int).tolist()