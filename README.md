# SGS Flow Matching

SE(3)-Equivariant Flow Matching for Sub-Grid Scale Closure in Large Eddy Simulation — EPITA DSA 2026.

## Structure

```
sgs-flow-matching/
├── src/
│   ├── aposteriori/      # A posteriori LES validation (spectralDNS)
│   ├── baselines/        # Smagorinsky, WALE, Beck MLP baselines
│   ├── data/             # JHTDB loader, dataset, augmentation
│   ├── evaluation/       # Metrics, visualisation, equivariance audit
│   ├── models/           # SE(3)-EGNN, MLP, conditional flow matching
│   └── training/         # Trainer, loss functions
├── configs/              # YAML experiment configs
├── notebooks/            # Task notebooks
├── figures/              # Generated plots
├── results/              # Evaluation outputs
├── train.py              # Training entry point
└── evaluate.py           # Evaluation entry point
```

## Setup

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[jhtdb]"
```

## Usage

```bash
sgs-train --config configs/v1.yaml
sgs-eval  --run results/run_v1 --data data/les_snapshots.h5
```

## A posteriori validation

```bash
python -m aposteriori.solver_setup
```

Runs a Taylor-Green vortex at Re=1600 on a 64³ grid with dynamic Smagorinsky SGS closure and prints PASS/FAIL.
