# MLMD25

Machine learning on molecular-dynamics (MD) trajectories to predict **agonism vs antagonism** for TLR4/MD-2 ligands, using graph neural networks (GCN / GAT).


## Repository map

| Path | Role |
|------|------|
| `GCNN_TLR4_MD2.ipynb` | Main validation notebook (frame splits + manual simulation checks) |
| `GAT_MANUAL_VALIDATION.ipynb` | GAT-focused manual validation notebook |
| `notebooks/` | Secondary / newer experiment notebooks |
| `mlmd/` | Installable Python package (`data`, `models`) |
| `scripts/` | CLI entry point (`mlmd-train`) |
| `TRAJECTORIES/` | In-repo **sample** MD (≈15 ns slices, 85–100 ns) for GitHub |
| `data/params/` | Token maps (e.g. `tokkens.v01.json`) |
| `results/` | Recorded validation accuracy tables and notes |
| `docs/` | Organization notes for the published layout |
| `archive/` | Historical notebook drafts and presentation exports |
| `test/` | Basic package tests |

**Important:** the primary notebooks do **not** load `TRAJECTORIES/` by default. They expect full external MD trees next to the repo (see below). `TRAJECTORIES/` is the portable sample described in `docs/organization.md`.

## Requirements

- Python ≥ 3.9
- A working **PyTorch** install (GPU recommended; the CLI currently trains on `cuda:0`)
- **PyTorch Geometric** and its sparse/scatter extensions
- **MDAnalysis** for trajectory I/O

Installing PyG correctly depends on your Torch/CUDA build and can be the hard part.

### 1. Create an environment and install PyTorch

Follow the official PyTorch instructions for your platform: https://pytorch.org/get-started/locally/

### 2. Install PyG extensions

This repo includes a known-good example for Torch 2.5 + CUDA 12.4:

```bash
bash environment.sh
# equivalent to:
# pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
#   -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
```

Pick the wheel index that matches **your** `torch` build from https://data.pyg.org/whl/ if versions differ. On CPU-only machines, use the corresponding CPU wheels from that site.

### 3. Install this project

From the repository root:

```bash
pip install -r requirements.txt
pip install -e .
```

Optional Jupyter / test tooling:

```bash
pip install -r requirements-dev.txt
```

## Data

### In-repo sample (`TRAJECTORIES/`)

Four systems, three replicas each, plus topology:

- **FP11**, **FP18** — agonists (label `0`)
- **FP12**, **FP7** — antagonists (label `1`)

See `TRAJECTORIES/trajectories.example.yaml` for a ready-made CLI manifest.

### External full trajectories (what the main notebooks use)

The trajectories can be reproduced from our initial structures, found in the project's [Zenodo repository](https://doi.org/10.5281/zenodo.22796892)

### Parameters

Token dictionary used by the library/CLI:

- `data/params/tokkens.v01.json`

## Run the notebooks

1. Install dependencies (above).
2. Place full MD data where the notebook paths expect it (or edit those path cells).
3. Open the entry-point notebook:

## Results

Historical validation tables and commentary live in:

- `results/validation_results.md`


## Contact

For ML-code queries, bruno.czuviria at upm.es 
For data and MD-simulations, olmomart@ucm.es 
