import torch
import numpy as np
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from cf4.compute import compute_q_criterion, compute_uncertainty
from af3 import get_reference_case, compute_differences, plot_comparison

def verify_cf4():
    print("Verifying CF4...")
    grad_u = torch.randn(1024, 9).numpy()
    variance_tau = torch.randn(1024, 6).numpy()
    Q = compute_q_criterion(grad_u)
    assert Q.shape == (1024,)
    print("  compute_q_criterion: PASS")
    unc = compute_uncertainty(variance_tau)
    assert unc.shape == (1024,)
    print("  compute_uncertainty: PASS")

def verify_af3():
    print("Verifying AF3...")
    case = get_reference_case("naca0012")
    assert case["name"] == "NACA 0012"
    print("  get_reference_case: PASS")
    tau_eng = torch.randn(1024, 3, 3)
    tau_ref = torch.randn(1024, 3, 3)
    grad_eng = torch.randn(1024, 3, 3)
    grad_ref = torch.randn(1024, 3, 3)
    diff, metrics = compute_differences(tau_eng, tau_ref)
    assert "mean_diff" in metrics
    print("  compute_differences: PASS")
    fig, metrics = plot_comparison(tau_eng, tau_ref, grad_eng, grad_ref, grid_size=(32, 32))
    assert fig is not None
    print("  plot_comparison: PASS")

if __name__ == "__main__":
    try:
        verify_cf4()
        verify_af3()
        print("\nAll verifications PASSED.")
    except Exception as e:
        print(f"\nVerification FAILED: {e}")
        sys.exit(1)
