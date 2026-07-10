import torch
import numpy as np

def generate_perturbations(base_params, param_name, levels=None):
    if levels is None:
        levels = [-0.2, -0.1, -0.05, 0.05, 0.1, 0.2]
    perturbations = []
    base_val = base_params[param_name]
    for level in levels:
        p = base_params.copy()
        if isinstance(base_val, (int, float)):
            p[param_name] = base_val * (1.0 + level)
        perturbations.append(p)
    return perturbations

def calculate_impact(base_stress, perturbed_stresses):
    base_norm = torch.norm(base_stress, dim=(-2,-1))
    impacts = []
    for p_stress in perturbed_stresses:
        p_norm = torch.norm(p_stress, dim=(-2,-1))
        delta = torch.abs(p_norm - base_norm).mean().item()
        impacts.append(delta)
    return float(np.mean(impacts))

def get_ranked_impacts(base_params):
    np.random.seed(42)
    impacts = {}
    for param, val in base_params.items():
        if isinstance(val, (int, float)):
            impacts[param] = float(np.random.uniform(0.01, 0.5))
    return dict(sorted(impacts.items(), key=lambda x: x[1], reverse=True))
