/**
 * CF4 diagnostics — JavaScript port for frontend mock/offline mode.
 * Matches the computation in src/cf4/compute.py.
 *
 * The irreps-to-stress conversion (needed for exact Π = -τ:S) is approximated
 * here using the isotropic part of τ; the backend /api/v1/cf4 endpoint gives
 * the exact result when available.
 */

function percentile(arr, p) {
  const sorted = [...arr].sort((a, b) => a - b);
  const idx = Math.floor((p / 100) * (sorted.length - 1));
  return sorted[idx];
}

export function computeCF4(meanTau, varianceTau, gradU) {
  const N = meanTau.length;

  // --- Q-criterion -----------------------------------------------------------
  const qValues = gradU.map(g => {
    // g = [9] flat row-major 3×3
    let omNorm2 = 0, sNorm2 = 0;
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        const Gij = g[i * 3 + j];
        const Gji = g[j * 3 + i];
        const Om = 0.5 * (Gij - Gji);
        const S  = 0.5 * (Gij + Gji);
        omNorm2 += Om * Om;
        sNorm2  += S * S;
      }
    }
    return 0.5 * (omNorm2 - sNorm2);
  });

  // --- Regime labels (percentile thresholds) ---------------------------------
  const p33 = percentile(qValues, 33.3);
  const p66 = percentile(qValues, 66.7);
  const regimeLabels = qValues.map(q => (q < p33 ? 0 : q < p66 ? 1 : 2));

  // --- Uncertainty (L2 norm of per-component variance) -----------------------
  const uncertainty = varianceTau.map(v => Math.sqrt(v.reduce((s, x) => s + x * x, 0)));
  const uncThresh   = percentile(uncertainty, 75);
  const highUncMask = uncertainty.map(u => u >= uncThresh);

  // --- SGS dissipation (approximate: isotropic τ contracted with trace(S)) ---
  // τ_iso ≈ mean_tau[:,0] / √3  (0e irrep = isotropic / √3 in e3nn convention)
  // tr(S) = (∂u_x/∂x + ∂u_y/∂y + ∂u_z/∂z) / 1
  // Π ≈ -τ_iso × tr(S)  (leading-order approximation)
  const invSqrt3 = 1 / Math.sqrt(3);
  const dissipation = meanTau.map((tau, i) => {
    const g = gradU[i];
    const tauIso = tau[0] * invSqrt3;
    return -(tauIso * (g[0] + g[4] + g[8]));
  });

  // --- Backscatter -----------------------------------------------------------
  const backscatterMask = dissipation.map(pi => pi < 0);
  const backscatterFrac = backscatterMask.filter(Boolean).length / N;

  return {
    q_values:           qValues,
    regime_labels:      regimeLabels,
    regime_thresholds:  [p33, p66],
    uncertainty,
    high_unc_mask:      highUncMask,
    dissipation,
    backscatter_mask:   backscatterMask,
    backscatter_frac:   backscatterFrac,
  };
}
