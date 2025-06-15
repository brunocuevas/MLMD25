import unittest
import json
from typing import List
import numpy as np


class TestData(unittest.TestCase):

    def test_start_end_split(self):
        from mlmd.data import start_end_split_index
        u = start_end_split_index(n=50, test_fraction=0.5)
        m = np.array(u).sum()
        self.assertEqual(1,1)

    def test_split_trajectory(self):
        from mlmd.data import start_end_split_index
        from mlmd.data import trajectory_to_data
        from mlmd.data import split_trajectory
        import MDAnalysis as mda
        u = mda.Universe(
            "data/md-2025-06-09/FP11ns.prmtop",
            "data/md-2025-06-09/FP11_R0_TRAJ_150ns.nc",
        )
        with open("data/params/tokkens.v01.json") as f: 
            tokkens_dict: dict = json.load(f)
        m = trajectory_to_data(
            u, selection_string='name CA',
            residue_index=tokkens_dict['residues'],
            atom_type_index=tokkens_dict['atoms'],
            label=0
        )
        index = start_end_split_index(n=len(m), test_fraction=0.25)
        n = split_trajectory(m, index=index)

    def test_batch_load(self):
        from mlmd import data as mlmdata
        from torch_geometric.loader import DataLoader


        data = {
            "trajectories": [
                {
                    "topology": "FP11ns.prmtop",
                    "trajectory": "FP11_R0_TRAJ_150ns.nc",
                    "label": 0
                }
            ],
            "path": "/home/bcz/research/mlmd25/data/md-2025-06-09/"
        }
        
        with open("data/params/tokkens.v01.json") as f: 
            tokkens_dict: dict = json.load(f)

        trajectories = mlmdata.trajectories_to_data(
            data_files=data, 
            selection_string="name CA", 
            residue_index=tokkens_dict['residues'],
            atom_type_index=tokkens_dict['atoms'],
        )

        train = []
        test = []

        for trj in trajectories:
            index = mlmdata.start_end_split_index(len(trj), test_fraction=0.25)
            out = mlmdata.split_trajectory(
                trj=trj, index=index
            )
            train += out[0]
            test += out[1]
            
        batch_size = 25
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test, batch_size=batch_size, shuffle=True)

        for d in train_loader:
            pass

        for d in test_loader:
            pass

        self.assertEqual(1, 1)