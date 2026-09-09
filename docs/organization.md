# GitHub organization notes

Original lab notes (Spanish), lightly updated for the current paths.

## Jupyter notebook

`GCNN_TLR4_MD2.ipynb` (repository root): full pipeline covering

- first 750 frames for train / last 750 for test
- train/test split every 100 frames
- manual validation with simulations

Related: `GAT_MANUAL_VALIDATION.ipynb` at the repository root.

## Results of test accuracy

`results/validation_results.md`: history of validation runs as tables from the code, with `test_accuracy` in the center column and loss columns on either side.

## TRAJECTORIES

Sample folder (maximum size practical for GitHub): ~15 ns of each simulation, taken from 85–100 ns. Layout:

- `FP11` — replicas R0, R1, R2 + `SYSTEMns.prmtop` (agonist, label 0)
- `FP12` — replicas R0, R1, R2 + `SYSTEMns.prmtop` (antagonist, label 1)
- `FP18` — replicas R0, R1, R2 + `SYSTEMns.prmtop` (agonist, label 0)
- `FP7` — replicas R0, R1, R2 + `SYSTEMns.prmtop` (antagonist, label 1)

CLI example manifest: `TRAJECTORIES/trajectories.example.yaml`.

## Other folders

- `archive/` — historical notebook drafts and presentations
- `notebooks/` — secondary experiments
- `mlmd/`, `scripts/` — installable library and `mlmd-train` CLI
- `data/params/` — token maps for atom/residue features
