"""
WALE (Wall-Adapting Local Eddy-viscosity) SGS closure.
Nicoud & Ducros (Flow, Turbulence and Combustion, 1999).

Formula:
    ν_t = (C_w Δ)² * (S^d_ij S^d_ij)^(3/2)
          / [(S_ij S_ij)^(5/2) + (S^d_ij S^d_ij)^(5/4)]

    τ_ij = -2 ν_t S_ij

where S^d_ij is the traceless symmetric part of g² (g = velocity gradient).
Constant C_w = 0.325 (calibrated for isotropic HIT, Ducros et al. 2000).
"""

import numpy as np


class WALE:
    """
    WALE SGS model.  Input/output in 3×3 tensor form.

    Args:
        delta: LES filter width (grid spacing).
        Cw:    WALE constant (default 0.325).
    """

    def __init__(self, delta: float = 1.0, Cw: float = 0.325):
        self.delta = delta
        self.Cw = Cw

    # ------------------------------------------------------------------

    def _strain_rate(self, grad_u: np.ndarray) -> np.ndarray:
        """S_ij = (g_ij + g_ji) / 2.  [N, 3, 3]"""
        return 0.5 * (grad_u + grad_u.swapaxes(-1, -2))

    def _traceless_sym_sq(self, grad_u: np.ndarray) -> np.ndarray:
        """
        Traceless symmetric part of g² (g_ik g_kj).

        S^d_ij = 0.5 (g²_ij + g²_ji) - (1/3) δ_ij g²_kk

        Returns [N, 3, 3].
        """
        g2 = grad_u @ grad_u                              # [N, 3, 3]
        sym = 0.5 * (g2 + g2.swapaxes(-1, -2))           # symmetric part
        trace = np.trace(g2, axis1=-2, axis2=-1)          # [N]
        I = np.eye(3, dtype=grad_u.dtype)
        Sd = sym - (trace / 3.0)[:, np.newaxis, np.newaxis] * I
        return Sd

    def eddy_viscosity(self, grad_u: np.ndarray) -> np.ndarray:
        """
        Compute WALE eddy viscosity ν_t for each point.

        Args:
            grad_u: [N, 3, 3] velocity gradient.

        Returns:
            nu_t: [N] non-negative eddy viscosity.
        """
        S  = self._strain_rate(grad_u)          # [N, 3, 3]
        Sd = self._traceless_sym_sq(grad_u)     # [N, 3, 3]

        SS   = (S  * S ).sum(axis=(-2, -1))     # [N]
        SdSd = (Sd * Sd).sum(axis=(-2, -1))     # [N]

        num   = np.clip(SdSd, 0.0, None) ** 1.5                     # (S^d:S^d)^(3/2)
        denom = (np.clip(SS, 0.0, None) ** 2.5                      # (S:S)^(5/2)
                 + np.clip(SdSd, 0.0, None) ** 1.25)                # (S^d:S^d)^(5/4)
        denom = np.where(denom < 1e-30, 1e-30, denom)

        nu_t = (self.Cw * self.delta) ** 2 * num / denom
        return np.clip(nu_t, 0.0, None)

    def predict(self, grad_u: np.ndarray) -> np.ndarray:
        """
        Predict SGS stress τ_ij = -2 ν_t S_ij.

        Args:
            grad_u: [N, 3, 3] velocity gradient.

        Returns:
            tau: [N, 3, 3] SGS stress.
        """
        nu_t = self.eddy_viscosity(grad_u)       # [N]
        S    = self._strain_rate(grad_u)         # [N, 3, 3]
        return -2.0 * nu_t[:, np.newaxis, np.newaxis] * S

    def validate(self):
        """Quick self-consistency check on synthetic strain-dominated data."""
        from scipy.stats import pearsonr

        rng = np.random.default_rng(0)
        N = 512
        grad_u = rng.standard_normal((N, 3, 3)).astype(np.float32)
        tau_pred = self.predict(grad_u)

        # For WALE: eddy viscosity should be non-negative everywhere
        nu_t = self.eddy_viscosity(grad_u)
        assert (nu_t >= 0).all(), "WALE: eddy viscosity went negative!"

        # Dissipation should be non-positive (WALE cannot represent backscatter)
        S = self._strain_rate(grad_u)
        Pi = (tau_pred * S).sum(axis=(-2, -1))
        bs_frac = float((Pi > 0).mean())
        print(f"WALE validate: backscatter fraction = {bs_frac:.4f}  "
              f"({'PASS' if bs_frac < 1e-6 else 'WARN — unexpected backscatter'})")
        return bs_frac


if __name__ == '__main__':
    model = WALE(delta=1.0)
    model.validate()
