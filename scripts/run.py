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
import comet_ml


@click.command()
@click.option("--epochs", "-e", default=20)
@click.option("--learning_rate", "-l", default=1e-4)
@click.option("--test_fraction", "-t", default=0.2)
@click.option("--test_split_method", "-s", type=click.Choice(['random', 'start_end']), default='start_end')
@click.option("--selection_string", "-u", default="name CA")
@click.option("--radius", "-r", default=4.0)
@click.option("--batch_size", "-b", default=32)
@click.option("--save_each", default=5)
@click.option("--comet_api", default=None)
@click.argument("DATA")
@click.argument("TOKKENS")
@click.argument("OUTPUT")
def train(
    data: str, tokkens:str, output: str, epochs: int, learning_rate: float, test_fraction: float, test_split_method: str,
    selection_string: str, radius: float, batch_size: int, save_each, comet_api: str | None
) -> None:
    """
    Trains a GCN on the given data. Features such as the
    radius, the batch size, or the selection string can be
    declared in the command line.

    Data must be in YAML format, containing trajectories
    as a list of ´topology´, ´trajectory´, ´label´. 
    
    The data YAML file must be located at the same place that
    data.
    """
    if comet_api is not None:
        comet_ml.login(api_key=comet_api)
        exp = comet_ml.start()
        exp.log_parameters({"batch_size": batch_size})
        exp.log_parameters({"selection_string": selection_string})
        exp.log_parameters({"learning_rate": learning_rate})
        exp.log_parameters({"test_split_method": test_split_method})
        exp.log_parameters({"name": output})


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
    index = None
    for trj in trajectories:
        
        if test_split_method == 'random':
            index = mlmdata.random_split_index(len(trj), test_fraction=test_fraction)
        elif test_split_method == 'start_end': 
            index = mlmdata.start_end_split_index(len(trj), test_fraction=test_fraction)
        else:
            raise RuntimeError("unknown split method")
        
        out: List[List[Data]] = mlmdata.split_trajectory(
            trj=trj, index=index # type: ignore
        )
        train += out[0]
        test += out[1]

    
    gcn = mlmodels.SimpleGCN(radius=radius).to(torch.device('cuda:0'))
    
    loss = CrossEntropyLoss()
    optimizer = torch.optim.Adam(gcn.parameters(), lr=learning_rate, weight_decay=5e-4)
    # reduce LR when test loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

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

        # compute average validation loss for the scheduler (avoid changing statistics['test_loss'] here)
        n_test_batches = len(test_loader) if len(test_loader) > 0 else 1
        val_loss_avg = statistics['test_loss'] / n_test_batches
        scheduler.step(val_loss_avg)
                    
        statistics['test_accuracy'] /= len(test)
        # average test loss over batches
        statistics['test_loss'] /= n_test_batches
        # record current learning rate
        statistics['learning_rate'] = optimizer.param_groups[0]['lr']
        if comet_api is not None:
            exp.log_metric(name="learning_rate", step=i, value=statistics['learning_rate'])
            exp.log_metric(name="training_loss", step=i, value=statistics['training_loss'])
            exp.log_metric(name="test_accuracy", step=i, value=statistics['test_accuracy'])
            exp.log_metric(name="test_loss", step=i, value=statistics['test_loss'])

        # print(f"Epoch {i+1:2d} | Accuracy {statistics['test_accuracy']:.4f}")
        history.append(statistics)
        if (i + 1) % save_each == 0:
            base = os.path.splitext(output)[0]
            fname = f"{base}.{i+1}.pt"
            torch.save(gcn.state_dict(), fname)

    pd.DataFrame.from_records(history).to_csv(output + '.csv', sep=';', index=None)


if __name__ == "__main__":
    train()