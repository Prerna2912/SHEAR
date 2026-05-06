"""
JHTDB data access for the forced isotropic turbulence (isotropic1024coarse) dataset.
Supports both the pyJHTDB web service and pre-downloaded local HDF5 files.
DNS: 1024^3, Gaussian-filtered to 64^3 LES grid.
"""

import os
import numpy as np
import torch
import h5py
from pathlib import Path


# ------------------------------------------------------------------
# Spectral Gaussian filter + downsampling
# ------------------------------------------------------------------

def gaussian_filter_3d(u: np.ndarray, filter_width: int) -> np.ndarray:
    """
    Apply a Gaussian spectral filter to a 3-D field and downsample.

    Args:
        u: [N, N, N] DNS velocity component (single component).
        filter_width: number of DNS cells per LES cell (e.g. 16 for 1024->64).

    Returns:
        u_les: [N//filter_width, ...] filtered + downsampled field.
    """
    N = u.shape[0]
    N_les = N // filter_width

    # Fourier transform
    u_hat = np.fft.rfftn(u)

    # Build Gaussian kernel in spectral space
    # G(k) = exp(-k^2 * Delta^2 / 24)  where Delta = filter_width * dx, dx=2pi/N
    kx = np.fft.fftfreq(N, d=1.0 / N).astype(np.float32)
    ky = kx.copy()
    kz = np.fft.rfftfreq(N, d=1.0 / N).astype(np.float32)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
    k2 = KX**2 + KY**2 + KZ**2
    sigma2 = (filter_width / (2 * np.pi))**2 / 24.0
    G = np.exp(-k2 * sigma2)

    u_hat_filtered = u_hat * G

    # Truncate to LES wavenumber range (sharp spectral cutoff)
    k_max = N_les // 2
    u_hat_les = u_hat_filtered[:k_max, :k_max, :k_max // 2 + 1] * (N_les / N)**3

    u_les = np.fft.irfftn(u_hat_les, s=(N_les, N_les, N_les)).real
    return u_les.astype(np.float32)


def compute_velocity_gradient(u: np.ndarray, v: np.ndarray, w: np.ndarray,
                               dx: float) -> np.ndarray:
    """
    Compute 3x3 velocity gradient tensor at each LES grid point via
    second-order central differences.

    Args:
        u, v, w: [N_les, N_les, N_les] LES velocity components.
        dx: LES grid spacing (physical units).

    Returns:
        grad_u: [N_les, N_les, N_les, 3, 3] where grad_u[..., i, j] = du_i/dx_j.
    """
    vels = np.stack([u, v, w], axis=-1)  # [N, N, N, 3]
    N = u.shape[0]
    grad = np.zeros((*u.shape, 3, 3), dtype=np.float32)

    for i in range(3):
        for j in range(3):
            # Periodic central difference along axis j
            grad[..., i, j] = (
                np.roll(vels[..., i], -1, axis=j)
                - np.roll(vels[..., i], 1, axis=j)
            ) / (2.0 * dx)

    return grad  # [N, N, N, 3, 3]


def compute_sgs_stress(u_dns: np.ndarray, v_dns: np.ndarray, w_dns: np.ndarray,
                        filter_width: int, dx_dns: float) -> tuple:
    """
    Compute filtered velocities and SGS stress tensor from DNS fields.

    tau_{ij} = bar(u_i * u_j) - bar(u_i) * bar(u_j)

    Returns:
        u_les, v_les, w_les: [N_les^3] filtered velocities.
        tau: [N_les, N_les, N_les, 3, 3] SGS stress (symmetric).
        grad_u: [N_les, N_les, N_les, 3, 3] filtered velocity gradient.
    """
    # Filter velocities
    u_les = gaussian_filter_3d(u_dns, filter_width)
    v_les = gaussian_filter_3d(v_dns, filter_width)
    w_les = gaussian_filter_3d(w_dns, filter_width)

    vels_les = [u_les, v_les, w_les]

    # Filter velocity products and compute tau_{ij}
    tau = np.zeros((*u_les.shape, 3, 3), dtype=np.float32)
    dns_vels = [u_dns, v_dns, w_dns]

    for i in range(3):
        for j in range(i, 3):
            uiuj_filtered = gaussian_filter_3d(dns_vels[i] * dns_vels[j], filter_width)
            tau_ij = uiuj_filtered - vels_les[i] * vels_les[j]
            tau[..., i, j] = tau_ij
            tau[..., j, i] = tau_ij  # symmetry

    dx_les = dx_dns * filter_width
    grad_u = compute_velocity_gradient(u_les, v_les, w_les, dx_les)

    return u_les, v_les, w_les, tau, grad_u


# ------------------------------------------------------------------
# JHTDB API wrapper
# ------------------------------------------------------------------

class JHTDBLoader:
    """
    Loads velocity data from JHTDB (online API or local HDF5 cache).

    Usage:
        loader = JHTDBLoader(token="your_token", cache_dir="./jhtdb_cache")
        u, v, w = loader.load_snapshot(time_idx=0)
    """

    DATASET = "isotropic1024coarse"
    DNS_N = 1024
    FILTER_WIDTH = 16      # 1024 -> 64
    DX_DNS = 2 * np.pi / 1024

    def __init__(self, token: str = "", cache_dir: str = "./jhtdb_cache"):
        self.token = token or os.environ.get("JHTDB_TOKEN", "")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._api = None

    def _get_api(self):
        if self._api is None:
            try:
                import pyJHTDB
                self._api = pyJHTDB.libJHTDB()
                self._api.initialize()
                if self.token:
                    self._api.add_token(self.token)
            except ImportError:
                raise ImportError("Install pyJHTDB: pip install pyJHTDB")
        return self._api

    def _cache_path(self, time_idx: int) -> Path:
        return self.cache_dir / f"isotropic1024_t{time_idx:04d}.h5"

    def load_snapshot(self, time_idx: int = 0) -> tuple:
        """
        Load DNS velocity snapshot, using cache if available.

        Returns:
            u, v, w: [1024, 1024, 1024] float32 DNS velocity components.
        """
        cache = self._cache_path(time_idx)
        if cache.exists():
            with h5py.File(cache, 'r') as f:
                return f['u'][:], f['v'][:], f['w'][:]

        # Download from JHTDB
        api = self._get_api()
        time = time_idx * 0.002  # JHTDB time step

        # Download in spatial chunks to avoid memory issues
        chunk = 256
        N = self.DNS_N
        u = np.zeros((N, N, N), dtype=np.float32)
        v = np.zeros_like(u)
        w = np.zeros_like(u)

        for iz in range(0, N, chunk):
            for iy in range(0, N, chunk):
                for ix in range(0, N, chunk):
                    x0 = np.array([ix, iy, iz], dtype=np.float32)
                    result = api.getVelocity(
                        self.DATASET, time,
                        x0[0], x0[1], x0[2],
                        min(chunk, N - ix),
                        min(chunk, N - iy),
                        min(chunk, N - iz),
                    )
                    xe = min(ix + chunk, N)
                    ye = min(iy + chunk, N)
                    ze = min(iz + chunk, N)
                    u[ix:xe, iy:ye, iz:ze] = result[..., 0]
                    v[ix:xe, iy:ye, iz:ze] = result[..., 1]
                    w[ix:xe, iy:ye, iz:ze] = result[..., 2]

        with h5py.File(cache, 'w') as f:
            f.create_dataset('u', data=u, compression='gzip')
            f.create_dataset('v', data=v, compression='gzip')
            f.create_dataset('w', data=w, compression='gzip')

        return u, v, w

    def prepare_les_data(self, time_idx: int = 0) -> tuple:
        """
        Load DNS snapshot and return LES-level fields.

        Returns:
            tau:    [64, 64, 64, 3, 3] SGS stress.
            grad_u: [64, 64, 64, 3, 3] filtered velocity gradient.
        """
        u, v, w = self.load_snapshot(time_idx)
        _, _, _, tau, grad_u = compute_sgs_stress(
            u, v, w, self.FILTER_WIDTH, self.DX_DNS
        )
        return tau, grad_u


# ------------------------------------------------------------------
# Synthetic data generator (for development / testing without JHTDB)
# ------------------------------------------------------------------

def generate_synthetic_les_data(
    n_les: int = 64,
    seed: int = 0,
    return_vels: bool = False,
) -> tuple:
    """
    Generate physically plausible synthetic LES data using random Fourier modes.
    Satisfies approximate incompressibility (div u ≈ 0 in spectral space).

    Returns:
        tau:    [n_les, n_les, n_les, 3, 3] SGS stress (symmetric).
        grad_u: [n_les, n_les, n_les, 3, 3] velocity gradient.
    """
    rng = np.random.default_rng(seed)
    N = n_les

    # Build a random solenoidal velocity field via projection
    u_hat = rng.standard_normal((N, N, N // 2 + 1, 3)).astype(np.complex64)
    u_hat[..., 1] = 1j * rng.standard_normal(u_hat[..., 1].shape).astype(np.float32)

    # Energy spectrum E(k) ~ k^{-5/3} (Kolmogorov)
    kx = np.fft.fftfreq(N, d=1.0 / N).astype(np.float32)
    kz = np.fft.rfftfreq(N, d=1.0 / N).astype(np.float32)
    KX, KY, KZ = np.meshgrid(kx, kx, kz, indexing='ij')
    k = np.sqrt(KX**2 + KY**2 + KZ**2)
    k[0, 0, 0] = 1.0
    E_k = k**(-5.0 / 3.0)
    E_k[0, 0, 0] = 0.0
    amp = np.sqrt(E_k / (4 * np.pi * k**2 + 1e-10))[..., np.newaxis]
    u_hat = u_hat * amp

    # Project onto solenoidal subspace (remove divergence)
    K = np.stack([KX, KY, KZ], axis=-1)  # [N,N,N/2+1, 3]
    k2 = np.sum(K**2, axis=-1, keepdims=True) + 1e-10
    u_hat = u_hat - (np.sum(u_hat * K, axis=-1, keepdims=True) / k2) * K

    # Inverse FFT to get velocity
    vels = np.zeros((N, N, N, 3), dtype=np.float32)
    for i in range(3):
        vels[..., i] = np.fft.irfftn(u_hat[..., i], s=(N, N, N)).real

    dx = 2 * np.pi / N

    # Velocity gradient
    grad_u = np.zeros((N, N, N, 3, 3), dtype=np.float32)
    for i in range(3):
        for j in range(3):
            grad_u[..., i, j] = (
                np.roll(vels[..., i], -1, axis=j)
                - np.roll(vels[..., i], 1, axis=j)
            ) / (2.0 * dx)

    # SGS stress: use Smagorinsky model as a proxy for ground truth
    # tau_{ij} = -2 * (C_s * Delta)^2 * |S| * S_{ij}
    # S_{ij} = (grad_u_{ij} + grad_u_{ji}) / 2
    S = 0.5 * (grad_u + grad_u.transpose(0, 1, 2, 4, 3))
    Smag = np.sqrt(2.0 * np.sum(S**2, axis=(-2, -1)))  # |S|
    C_s = 0.17
    Delta = dx
    tau = -2.0 * (C_s * Delta)**2 * Smag[..., np.newaxis, np.newaxis] * S

    # Add realistic fluctuations
    tau += 0.1 * rng.standard_normal(tau.shape).astype(np.float32) * tau.std()

    if return_vels:
        return tau.astype(np.float32), grad_u.astype(np.float32), vels.astype(np.float32)
    return tau.astype(np.float32), grad_u.astype(np.float32)
