"""
Evaluation entry point.

Runs the full metrics suite and optionally the equivariance audit.

Usage:
    # Evaluate a single variant
    python evaluate.py --config configs/v1.yaml --checkpoint runs/v1/best.pt

    # Evaluate all three variants and run equivariance audit
    python evaluate.py --all \
        --ckpt_v1 runs/v1/best.pt \
        --ckpt_v2 runs/v2/best.pt \
        --ckpt_v3 runs/v3/best.pt \
        --audit
"""

import argparse
import json
import yaml
import torch
from pathlib import Path


def load_model_from_checkpoint(variant: str, cfg: dict, ckpt_path: str, device: torch.device):
    from models.flow_matching import build_model
    model = build_model(variant, cfg['model'])
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    model = model.to(device)
    model.eval()
    return model


def evaluate_one(variant: str, cfg: dict, ckpt_path: str, device: torch.device,
                 tau_field, grad_field) -> dict:
    from data.dataset import build_dataloaders
    data_cfg = cfg['data']
    _, _, test_loader, stats = build_dataloaders(
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

    # Try to load normalisation stats from run directory
    stats_path = Path(cfg['out_dir']) / 'stats.pt'
    if stats_path.exists():
        stats = torch.load(stats_path)

    model = load_model_from_checkpoint(variant, cfg, ckpt_path, device)

    from evaluation.metrics import compute_all_metrics
    metrics = compute_all_metrics(model, test_loader, variant, device, stats=stats)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='', help='Config for single-variant evaluation')
    parser.add_argument('--checkpoint', default='', help='Checkpoint for single-variant')
    parser.add_argument('--all', action='store_true', help='Evaluate all three variants')
    parser.add_argument('--ckpt_v1', default='runs/v1/best.pt')
    parser.add_argument('--ckpt_v2', default='runs/v2/best.pt')
    parser.add_argument('--ckpt_v3', default='runs/v3/best.pt')
    parser.add_argument('--audit', action='store_true', help='Run equivariance audit')
    parser.add_argument('--device', default='')
    parser.add_argument('--out', default='./results', help='Output directory for results')
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data (same for all variants)
    # Use first config found to determine data settings
    cfg_path = args.config or 'configs/v1.yaml'
    with open(cfg_path) as f:
        base_cfg = yaml.safe_load(f)

    data_cfg = base_cfg['data']
    if data_cfg.get('use_jhtdb', False):
        from data.jhtdb import JHTDBLoader
        loader = JHTDBLoader(token=data_cfg.get('jhtdb_token', ''),
                             cache_dir=data_cfg.get('jhtdb_cache_dir', './jhtdb_cache'))
        tau_field, grad_field = loader.prepare_les_data(data_cfg.get('jhtdb_time_idx', 0))
    else:
        from data.jhtdb import generate_synthetic_les_data
        tau_field, grad_field = generate_synthetic_les_data(n_les=64, seed=0)

    all_results = {}

    if args.all:
        for variant, ckpt_path, cfg_file in [
            ('v1', args.ckpt_v1, 'configs/v1.yaml'),
            ('v2', args.ckpt_v2, 'configs/v2.yaml'),
            ('v3', args.ckpt_v3, 'configs/v3.yaml'),
        ]:
            if not Path(ckpt_path).exists():
                print(f"Checkpoint not found for {variant}: {ckpt_path}, skipping.")
                continue
            print(f"\n=== Evaluating {variant.upper()} ===")
            with open(cfg_file) as f:
                cfg = yaml.safe_load(f)
            metrics = evaluate_one(variant, cfg, ckpt_path, device, tau_field, grad_field)
            all_results[variant] = metrics
            _print_metrics(variant, metrics)

        if args.audit:
            print("\n=== Running Equivariance Audit ===")
            models_for_audit = {}
            for variant, ckpt_path, cfg_file in [
                ('v1', args.ckpt_v1, 'configs/v1.yaml'),
                ('v2', args.ckpt_v2, 'configs/v2.yaml'),
                ('v3', args.ckpt_v3, 'configs/v3.yaml'),
            ]:
                if Path(ckpt_path).exists():
                    with open(cfg_file) as f:
                        cfg = yaml.safe_load(f)
                    models_for_audit[variant] = load_model_from_checkpoint(
                        variant, cfg, ckpt_path, device
                    )

            from data.dataset import build_dataloaders
            _, _, test_loader, _ = build_dataloaders(
                tau_field=tau_field, grad_field=grad_field,
                n_train=4000, n_val=500, n_test=500,
                batch_size=8, num_workers=2, seed=42,
            )
            from evaluation.audit import run_equivariance_audit
            audit_results = run_equivariance_audit(
                models_for_audit, test_loader, device, out_dir=str(out_dir)
            )
            all_results['equivariance_audit'] = audit_results

    else:
        # Single-variant evaluation
        assert args.config and args.checkpoint, \
            "Provide --config and --checkpoint for single-variant evaluation."
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        variant = cfg['variant']
        print(f"\n=== Evaluating {variant.upper()} ===")
        metrics = evaluate_one(variant, cfg, args.checkpoint, device, tau_field, grad_field)
        all_results[variant] = metrics
        _print_metrics(variant, metrics)

    # Save results
    out_file = out_dir / 'metrics.json'
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_file}")

    # Print comparison table if multiple variants
    if len(all_results) > 1:
        _print_comparison_table(all_results)


def _print_metrics(variant: str, metrics: dict):
    print(f"\n  {variant.upper()} Results:")
    print(f"    Pearson (mean):          {metrics['pearson']['mean']:.4f}")
    print(f"    JSD invariants (mean):   {metrics['jsd_invariants']['mean']:.6f}")
    print(f"    Dissipation RMSE:        {metrics['dissipation']['rmse']:.6f}")
    print(f"    Dissipation corr:        {metrics['dissipation']['correlation']:.4f}")
    print(f"    Alignment angle (mean):  {metrics['alignment']['mean_deg']:.2f}°")
    print(f"    Inference ms/sub-cube:   {metrics['efficiency']['ms_per_subcube']:.2f}")
    print(f"    Peak GPU memory (MB):    {metrics['efficiency']['peak_gpu_mb']:.1f}")


def _print_comparison_table(all_results: dict):
    header = f"{'Metric':<35} " + " ".join(f"{v.upper():>10}" for v in ['v1', 'v2', 'v3'])
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    print(header)
    print("-" * 70)

    rows = [
        ("Pearson r (mean)",     lambda m: f"{m['pearson']['mean']:.4f}"),
        ("JSD I1",               lambda m: f"{m['jsd_invariants']['I1']:.5f}"),
        ("JSD I2",               lambda m: f"{m['jsd_invariants']['I2']:.5f}"),
        ("JSD I3",               lambda m: f"{m['jsd_invariants']['I3']:.5f}"),
        ("Dissipation RMSE",     lambda m: f"{m['dissipation']['rmse']:.5f}"),
        ("Dissipation corr",     lambda m: f"{m['dissipation']['correlation']:.4f}"),
        ("Align. angle (deg)",   lambda m: f"{m['alignment']['mean_deg']:.2f}"),
        ("ms / sub-cube",        lambda m: f"{m['efficiency']['ms_per_subcube']:.2f}"),
    ]

    for label, fn in rows:
        vals = []
        for v in ['v1', 'v2', 'v3']:
            if v in all_results and 'pearson' in all_results[v]:
                vals.append(fn(all_results[v]))
            else:
                vals.append("  N/A")
        print(f"  {label:<33} " + " ".join(f"{v:>10}" for v in vals))
    print("=" * 70)


if __name__ == '__main__':
    main()
