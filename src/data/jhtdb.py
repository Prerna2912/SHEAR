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
    Loads LES-resolution velocity data from JHTDB via givernylocal.

    Strategy: getCutout with stride=FILTER_WIDTH to download the 1024^3 DNS
    directly at 64^3 LES resolution. The testing token allows ≤4096 points
    per call; the full 64^3 grid is split into 64 batches of 16^3=4096 each.

    SGS stress is computed with dynamic Smagorinsky on the filtered field
    (exact Germano stress requires the full 1024^3 DNS, impractical over API).

    Usage:
        loader = JHTDBLoader(token="your_token", cache_dir="./jhtdb_cache")
        tau, grad_u = loader.prepare_les_data(time_idx=0)
    """

    DATASET     = "isotropic1024coarse"
    DNS_N       = 1024
    LES_N       = 64
    FILTER_WIDTH = 16          # stride: 1024 → 64
    DX_DNS      = 2 * np.pi / 1024
    DX_LES      = 2 * np.pi / 64
    TIMEPOINT   = 0.0          # t=0 snapshot

    # Testing token max datapoints per getCutout call
    _BATCH_POINTS = 4096       # 16^3

    def __init__(self, token: str = "", cache_dir: str = "./jhtdb_cache"):
        self.token = token or os.environ.get("JHTDB_TOKEN", "")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cube(self):
        try:
            from givernylocal.turbulence_dataset import turb_dataset
        except ImportError:
            raise ImportError("Install givernylocal: pip install givernylocal")
        return turb_dataset(
            dataset_title=self.DATASET,
            output_path=str(self.cache_dir),
            auth_token=self.token,
        )

    def _cache_path(self, time_idx: int) -> Path:
        return self.cache_dir / f"les_t{time_idx:04d}.h5"

    @staticmethod
    def _extract_velocity(ds, t_step: int) -> np.ndarray:
        """Extract velocity array from xarray Dataset returned by getCutout."""
        key = f"velocity_{t_step:04d}"
        return ds[key].values.astype(np.float32)  # [z, y, x, 3]

    def test_connection(self) -> bool:
        """
        Verify API connectivity using a minimal 16^3 = 4096-point query.
        Works with the public testing token.

        Returns True on success, raises on failure.
        """
        from givernylocal.turbulence_toolkit import getCutout

        cube    = self._get_cube()
        t_step  = np.int32(1)
        axes    = np.array([[1, 16], [1, 16], [1, 16],
                            [t_step, t_step]], dtype=np.int32)
        strides = np.array([1, 1, 1, 1], dtype=np.int32)

        result = getCutout(cube, 'velocity', axes, strides, verbose=False)
        vel    = self._extract_velocity(result, int(t_step))  # [16, 16, 16, 3]
        assert vel.shape == (16, 16, 16, 3), f"Unexpected shape: {vel.shape}"
        print(f"Connection OK — u[0,0,0] = {vel[0,0,0,0]:.4f}  "
              f"v[0,0,0] = {vel[0,0,0,1]:.4f}  w[0,0,0] = {vel[0,0,0,2]:.4f}")
        return True

    def load_les_velocities(self, time_idx: int = 0) -> tuple:
        """
        Download the full 64^3 LES-resolution velocity field via a single
        getCutout call with stride=16 on the 1024^3 DNS.

        Requires an authorized token (testing token is limited to 4096 raw
        points per call, which is insufficient for a strided 1024^3 query).
        Request a token at: turbulence@lists.johnshopkins.edu

        Returns:
            u, v, w: [64, 64, 64] float32 arrays.
        """
        cache = self._cache_path(time_idx)
        if cache.exists():
            with h5py.File(cache, 'r') as f:
                if 'u_les' in f:
                    return f['u_les'][:], f['v_les'][:], f['w_les'][:]

        from givernylocal.turbulence_toolkit import getCutout

        cube   = self._get_cube()
        t_step = np.int32(time_idx + 1)    # JHTDB uses 1-based time indices
        axes   = np.array([[1, self.DNS_N], [1, self.DNS_N], [1, self.DNS_N],
                           [t_step, t_step]], dtype=np.int32)
        stride = np.array([self.FILTER_WIDTH, self.FILTER_WIDTH,
                           self.FILTER_WIDTH, 1], dtype=np.int32)

        print(f"Downloading 64³ LES velocities (stride {self.FILTER_WIDTH} "
              f"on {self.DNS_N}³ DNS, t_step={t_step}) …")
        result = getCutout(cube, 'velocity', axes, stride, verbose=False)
        vel    = self._extract_velocity(result, int(t_step))  # [64, 64, 64, 3]

        u, v, w = vel[..., 0], vel[..., 1], vel[..., 2]
        return u, v, w

    def prepare_les_data(self, time_idx: int = 0) -> tuple:
        """
        Download the full 1024^3 DNS velocity and compute the TRUE Leonard SGS stress:
            τᵢⱼ = bar(uᵢ·uⱼ) − bar(uᵢ)·bar(uⱼ)
        using a box filter of width filter_width=16 DNS cells per LES cell.

        Downloads in 64 z-slabs (~200 MB each, ~12.9 GB total) to stay within
        memory limits. Requires an authorised JHTDB token for large cutouts.
        Results are cached to HDF5 after the first call.

        Returns:
            tau:    [64, 64, 64, 3, 3] true SGS stress.
            grad_u: [64, 64, 64, 3, 3] filtered velocity gradient.
        """
        cache = self._cache_path(time_idx)
        if cache.exists():
            with h5py.File(cache, 'r') as f:
                if ('tau' in f and 'grad_u' in f
                        and f.attrs.get('filter_type') == 'box_leonard'):
                    print(f"Loaded from cache: {cache}")
                    return f['tau'][:], f['grad_u'][:]
            print("Stale cache detected (old Smagorinsky approximation) — "
                  "recomputing with true Leonard stress.")

        tau, u, v, w = self._compute_leonard_stress_chunked(time_idx)
        grad_u = compute_velocity_gradient(u, v, w, self.DX_LES)

        with h5py.File(cache, 'w') as f:
            f.create_dataset('u_les',  data=u,      compression='gzip')
            f.create_dataset('v_les',  data=v,      compression='gzip')
            f.create_dataset('w_les',  data=w,      compression='gzip')
            f.create_dataset('tau',    data=tau,    compression='gzip')
            f.create_dataset('grad_u', data=grad_u, compression='gzip')
            f.attrs['time_idx']     = time_idx
            f.attrs['filter_width'] = self.FILTER_WIDTH
            f.attrs['les_n']        = self.LES_N
            f.attrs['filter_type']  = 'box_leonard'

        print(f"Saved LES cache → {cache}")
        return tau, grad_u

    def _compute_leonard_stress_chunked(self, time_idx: int) -> tuple:
        """
        Download the 1024^3 DNS velocity one z-slab at a time and compute
        the true Leonard SGS stress with a box filter.

        For each LES z-slice k, downloads DNS z ∈ [k·fw, (k+1)·fw) at full
        x-y resolution (slab shape: [fw, 1024, 1024, 3] ≈ 200 MB).

        Box filter: each LES cell averages fw^3 = 4096 DNS cells.
        Reshape trick: [fw, 1024, 1024] → [fw, 64, fw, 64, fw] then
        mean over axes (0, 2, 4) gives the filtered value at [64, 64].

        Total download: 64 slabs × 200 MB ≈ 12.9 GB (cached after first run).
        """
        from givernylocal.turbulence_toolkit import getCutout

        cube   = self._get_cube()
        t_step = np.int32(time_idx + 1)
        fw = self.FILTER_WIDTH    # 16
        N  = self.DNS_N           # 1024
        M  = self.LES_N           # 64

        PAIRS = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]

        u_les    = np.zeros((M, M, M, 3),          dtype=np.float32)
        bar_uiuj = np.zeros((M, M, M, len(PAIRS)), dtype=np.float64)

        for k in range(M):
            z0 = k * fw + 1          # JHTDB 1-based index
            z1 = (k + 1) * fw

            axes = np.array(
                [[1, N], [1, N], [z0, z1], [t_step, t_step]],
                dtype=np.int32,
            )
            strides = np.ones(4, dtype=np.int32)

            print(f"\r  z-slab {k + 1:2d}/{M}  (DNS z={z0}–{z1}, "
                  f"{(k + 1) / M * 100:.0f}% done)    ", end="", flush=True)

            res = getCutout(cube, 'velocity', axes, strides, verbose=False)
            vel = np.ascontiguousarray(
                self._extract_velocity(res, int(t_step))   # [fw, N, N, 3]
            )

            # Box filter via reshape: [fw, N, N] → [fw, M, fw, M, fw]
            # averaging over the three within-cell axes (0, 2, 4)
            vr = vel.reshape(fw, M, fw, M, fw, 3)
            u_les[k] = vr.mean(axis=(0, 2, 4)).astype(np.float32)  # [M, M, 3]

            for idx, (i, j) in enumerate(PAIRS):
                uiuj = (vel[..., i].astype(np.float64)
                        * vel[..., j].astype(np.float64))   # [fw, N, N]
                bar_uiuj[k, :, :, idx] = (
                    uiuj.reshape(fw, M, fw, M, fw).mean(axis=(0, 2, 4))
                )

            del vel, vr

        print("\n  Computing τᵢⱼ = bar(uᵢuⱼ) − bar(uᵢ)·bar(uⱼ) …")
        tau = np.zeros((M, M, M, 3, 3), dtype=np.float32)
        for idx, (i, j) in enumerate(PAIRS):
            t_ij = (bar_uiuj[..., idx]
                    - u_les[..., i].astype(np.float64)
                    * u_les[..., j].astype(np.float64))
            tau[..., i, j] = t_ij.astype(np.float32)
            tau[..., j, i] = tau[..., i, j]

        return tau, u_les[..., 0], u_les[..., 1], u_les[..., 2]


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
