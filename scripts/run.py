import json
import click
import torch
import torch.nn.functional as F
import numpy as np
import tqdm
from torch.nn import CrossEntropyLoss
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from mlmd import data as mlmdata
from mlmd import models as mlmodels
from typing import List
import yaml
import os
import pandas as pd


@click.command()
@click.option("--epochs", "-e", default=20)
@click.option("--learning_rate", "-l", default=1e-4)
@click.option("--test_fraction", "-t", default=0.2)
@click.option("--test_split_method", "-s", type=click.Choice(['random', 'start_end']), default='start_end')
@click.option("--selection_string", "-u", default="name CA")
@click.option("--radius", "-r", default=4.0)
@click.option("--batch_size", "-b", default=32)
@click.argument("DATA")
@click.argument("TOKKENS")
@click.argument("OUTPUT")
def train(
    data: str, tokkens:str, output: str, epochs: int, learning_rate: float, test_fraction: float, test_split_method: str,
    selection_string: str, radius: float, batch_size: int
) -> None:
    """
    Trains a GCN on the given data. Features such as the
    radius, the batch size, or the selection string can be
    declared in the command line.

    Data must be in YAML format, containing trajectories
    as a list of ´topology´, ´trajectory´, ´label´. 
    
    The data YAML file must be located at the same place that
    data
    """

    with open(tokkens) as f: 
        tokkens_dict: dict = json.load(f)

    path = os.path.abspath(data)
    path = "/".join(path.split("/")[:-1])
    print(path)
    with open(data) as f:
        data: dict = yaml.load(f, yaml.Loader)
    data['path'] = path

    atom_type_index: dict = tokkens_dict['atoms']
    residue_type_index: dict = tokkens_dict['residues']

    trajectories = mlmdata.trajectories_to_data(
        data_files=data, 
        selection_string=selection_string, 
        residue_index=residue_type_index, 
        atom_type_index=atom_type_index
    )

    train = []
    test = []

    for trj in trajectories:
        
        if test_split_method == 'random':
            index = mlmdata.random_split_index(len(trj), test_fraction=test_fraction)
        elif test_split_method == 'start_end': 
            index = mlmdata.start_end_split_index(len(trj), test_fraction=test_fraction)
        else:
            raise RuntimeError("unknown split method")
        
        out: List[List[Data]] = mlmdata.split_trajectory(
            trj=trj, index=index
        )
        train += out[0]
        test += out[1]

    
    gcn = mlmodels.GCN(radius=radius).to(torch.device('cuda:0'))
    
    loss = CrossEntropyLoss()
    optimizer = torch.optim.Adam(gcn.parameters(), lr=learning_rate, weight_decay=5e-4)
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=True)
    history = []
    for i in tqdm.tqdm(range(epochs)):
        statistics = dict()
        statistics['training_loss'] = 0.0
        gcn.train()
        for d in train_loader:
            d = d.to(torch.device("cuda:0"))
            optimizer.zero_grad()
            z = gcn(d)
            l = loss(z, d.label)
            l.backward()
            optimizer.step()
        statistics['training_loss'] += l.item()
        gcn.eval()
        with torch.no_grad():
            statistics['test_accuracy'] = 0.0
            statistics['test_loss'] = 0.0
            for d in test_loader:
                d = d.to(torch.device("cuda:0"))
                z = gcn(d)
                l = loss(z, d.label)
                a = ((z.argmax(1) == d.label).sum())
                statistics['test_accuracy'] += a.item()
                statistics['test_loss'] += l.item()
                    
        statistics['test_accuracy'] /= len(test)
        statistics['test_loss'] /= ((len(test) // batch_size) + 1)

        # print(f"Epoch {i+1:2d} | Accuracy {statistics['test_accuracy']:.4f}")
        history.append(statistics)

    pd.DataFrame.from_records(history).to_csv(output, sep=';', index=None)


if __name__ == "__main__":
    train()