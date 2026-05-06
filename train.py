"""
Training entry point.

Usage:
    python train.py --config configs/v1.yaml
    python train.py --config configs/v2.yaml
    python train.py --config configs/v3.yaml
    python train.py --config configs/v1.yaml --resume runs/v1/best.pt
"""

import argparse
import yaml
import torch
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to YAML config')
    parser.add_argument('--resume', default='', help='Path to checkpoint to resume from')
    parser.add_argument('--device', default='', help='cuda / cpu (auto-detected if empty)')
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    variant = cfg['variant']
    device = torch.device(
        args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    print(f"Variant: {variant.upper()} | Device: {device}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    data_cfg = cfg['data']

    if data_cfg.get('use_jhtdb', False):
        from data.jhtdb import JHTDBLoader
        loader = JHTDBLoader(
            token=data_cfg.get('jhtdb_token', ''),
            cache_dir=data_cfg.get('jhtdb_cache_dir', './jhtdb_cache'),
        )
        tau_field, grad_field = loader.prepare_les_data(
            time_idx=data_cfg.get('jhtdb_time_idx', 0)
        )
    else:
        print("Using synthetic LES data (set use_jhtdb: true for real JHTDB data).")
        from data.jhtdb import generate_synthetic_les_data
        tau_field, grad_field = generate_synthetic_les_data(n_les=64, seed=0)

    from data.dataset import build_dataloaders
    train_loader, val_loader, test_loader, stats = build_dataloaders(
        tau_field=tau_field,
        grad_field=grad_field,
        n_train=data_cfg['n_train'],
        n_val=data_cfg['n_val'],
        n_test=data_cfg['n_test'],
        patch_size=data_cfg['patch_size'],
        k_neighbours=data_cfg['k_neighbours'],
        batch_size=cfg['training']['batch_size'],
        num_workers=data_cfg['num_workers'],
        seed=data_cfg['seed'],
    )
    print(f"Data: {data_cfg['n_train']} train / {data_cfg['n_val']} val / "
          f"{data_cfg['n_test']} test sub-cubes")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    from models.flow_matching import build_model
    model = build_model(variant, cfg['model'])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    from training.trainer import Trainer
    trainer = Trainer(
        model=model,
        variant=variant,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg['training'],
        device=device,
        out_dir=cfg['out_dir'],
    )

    if args.resume:
        trainer.load_checkpoint(args.resume)

    trainer.train()

    # Save stats for evaluation
    import torch
    torch.save(stats, Path(cfg['out_dir']) / 'stats.pt')
    print(f"Stats saved to {cfg['out_dir']}/stats.pt")


if __name__ == '__main__':
    main()
