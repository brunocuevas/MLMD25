# MLMD25

## Summary

The goal of this project is the prediction of agonism/antagonism in TLR receptors

## Content 

- *notebooks*
- *scripts* This folder contains the new script that we will us to run stuff.
- *data*
- *results*

## Installation

The project requires a valid CUDA installation and a PYG (PyTorch Graph) installation.
Doing so can be challening, considering the differences between CUDA versions and
machines. 

The following bash command allowed us to install PyG.

    pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.5.0+cu124.html


To install the library, run the following command at
the top of the folder.

    pip install -e .

## Train

**IMPORTANT**. New addition. The following command runs a 20 epochs training on the trajectories.yaml file.


    mlmd-train trajectories.yaml ../params/tokkens.v01.json run01


`trajectories.yaml` must be located at the folder
where the MD simulations are. A very basic example of
this file is:

    trajectories: 
  
        - topology: "FP11ns.prmtop"
            trajectory: "FP11_R0_TRAJ_150ns.nc"
            label: 0

        - topology: "FP11ns.prmtop"
            trajectory: "FP11_R1_TRAJ_150ns.nc"
            label: 0


## Contact


