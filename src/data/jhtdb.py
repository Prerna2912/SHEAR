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
    sigma2 = (filter_width * 2 * np.pi / N) ** 2 / 24.0
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
        Compute the true Leonard SGS stress with the Gaussian LES filter:
            G(k) = exp(−k²Δ²/24),  Δ = filter_width × dx_dns

        τᵢⱼ = bar(uᵢ·uⱼ) − bar(uᵢ)·bar(uⱼ)

        The filter is applied as a separable 1D Gaussian along each spatial
        direction (x→y→z), processing one z-slab at a time to avoid loading
        the full 1024³ DNS field.  Peak memory per slab ≈ 300 MB.
        Total download ≈ 12.9 GB; cached to HDF5 after the first call.

        Returns:
            tau:    [64, 64, 64, 3, 3] true SGS stress.
            grad_u: [64, 64, 64, 3, 3] filtered velocity gradient.
        """
        cache = self._cache_path(time_idx)
        if cache.exists():
            with h5py.File(cache, 'r') as f:
                if ('tau' in f and 'grad_u' in f
                        and f.attrs.get('filter_type') == 'gaussian_leonard'):
                    print(f"Loaded from cache: {cache}")
                    return f['tau'][:], f['grad_u'][:]
            print("Stale cache detected — recomputing with Gaussian-filtered "
                  "Leonard stress.")

        tau, u, v, w = self._compute_leonard_stress_spectral(time_idx)
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
            f.attrs['filter_type']  = 'gaussian_leonard'

        print(f"Saved LES cache → {cache}")
        return tau, grad_u

    # ------------------------------------------------------------------
    # Gaussian-filtered Leonard stress (separable 1D filter, slab-by-slab)
    # ------------------------------------------------------------------

    def _compute_leonard_stress_spectral(self, time_idx: int) -> tuple:
        """
        Download 1024³ DNS velocity in z-slabs and compute the true Leonard
        SGS stress using the separable Gaussian spectral filter.

        Algorithm (separability of Gaussian: G_3D = G_x · G_y · G_z):
          For each z-slab [fw, N, N, 3]:
            1. Download in x-tiles (256×1024×16 each, ~48 MB — within API limit)
            2. 1D Gaussian filter + downsample along x → [fw, N, M]
            3. 1D Gaussian filter + downsample along y → [fw, M, M]
            4. Write result to progress HDF5 on Drive (survives session restarts)
          After all slabs:
            5. 1D Gaussian filter + downsample along z → [M, M, M]
          Finally:
            6. τᵢⱼ = bar(uᵢuⱼ) − bar(uᵢ)·bar(uⱼ)

        Progress is checkpointed to Drive after each slab so a disconnected
        session can resume where it stopped rather than re-downloading from slab 1.
        """
        import gc
        from givernylocal.turbulence_toolkit import getCutout

        cube   = self._get_cube()
        t_step = np.int32(time_idx + 1)
        fw      = self.FILTER_WIDTH   # 16
        N       = self.DNS_N          # 1024
        M       = self.LES_N          # 64
        n_slabs = N // fw             # 64

        PAIRS = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]
        n_c, n_p = 3, len(PAIRS)

        # x-axis tiling: 4 tiles of 256×1024×16 = 4.2M pts each (~48 MB),
        # well within givernylocal's safe ~8M point / 100 MB per-call limit.
        X_TILE   = 256
        n_xtiles = N // X_TILE
        strides  = np.ones(4, dtype=np.int32)

        # Progress file lives in cache_dir (Google Drive in Colab) so it
        # survives a session restart.  Each slab's xy-filtered result is
        # written here immediately; on resume, completed slabs are skipped.
        progress_path = self.cache_dir / f"les_t{time_idx:04d}_progress.h5"

        with h5py.File(progress_path, 'a') as pf:
            if 'vel_inter' not in pf:
                pf.create_dataset('vel_inter',
                                  shape=(n_slabs, fw, M, M, n_c),
                                  dtype='float32',
                                  chunks=(1, fw, M, M, n_c))
                pf.create_dataset('prod_inter',
                                  shape=(n_slabs, fw, M, M, n_p),
                                  dtype='float32',
                                  chunks=(1, fw, M, M, n_p))
                pf.create_dataset('slab_done',
                                  data=np.zeros(n_slabs, dtype=bool))
            slab_done = pf['slab_done'][:]

        n_done = int(slab_done.sum())
        if n_done > 0:
            print(f"  Resuming: {n_done}/{n_slabs} slabs already on Drive, "
                  f"starting from slab {n_done + 1}.")

        for k in range(n_slabs):
            if slab_done[k]:
                continue

            z0 = k * fw + 1
            z1 = (k + 1) * fw
            print(f"  z-slab {k + 1:2d}/{n_slabs}  "
                  f"(DNS z={z0}–{z1}, {(k + 1) / n_slabs * 100:.0f}%)",
                  flush=True)

            # Assemble slab from x-tiles
            vel = np.empty((fw, N, N, 3), dtype=np.float32)
            for tx in range(n_xtiles):
                x0 = tx * X_TILE + 1    # JHTDB is 1-indexed
                x1 = (tx + 1) * X_TILE
                tile_axes = np.array(
                    [[x0, x1], [1, N], [z0, z1], [t_step, t_step]],
                    dtype=np.int32,
                )
                tile_res = getCutout(cube, 'velocity', tile_axes, strides,
                                     verbose=False)
                vel[:, :, tx * X_TILE:(tx + 1) * X_TILE, :] = \
                    self._extract_velocity(tile_res, int(t_step))

            # Apply xy Gaussian filter
            vel_xy  = np.stack(
                [self._gaussian_filter_xy(vel[..., c]) for c in range(n_c)],
                axis=-1,
            )
            prod_xy = np.stack(
                [self._gaussian_filter_xy(
                    (vel[..., i].astype(np.float64)
                     * vel[..., j].astype(np.float64)).astype(np.float32))
                 for (i, j) in PAIRS],
                axis=-1,
            )
            del vel
            gc.collect()

            # Checkpoint to Drive
            with h5py.File(progress_path, 'a') as pf:
                pf['vel_inter'][k]  = vel_xy
                pf['prod_inter'][k] = prod_xy
                pf['slab_done'][k]  = True

            del vel_xy, prod_xy

        print()

        # All slabs complete — load from Drive and apply z-filter
        print("  Applying z-direction Gaussian filter …")
        with h5py.File(progress_path, 'r') as pf:
            vel_inter  = pf['vel_inter'][:]   # [n_slabs, fw, M, M, n_c]
            prod_inter = pf['prod_inter'][:]  # [n_slabs, fw, M, M, n_p]

        vel_z  = vel_inter.reshape(N, M, M, n_c)
        prod_z = prod_inter.reshape(N, M, M, n_p)
        del vel_inter, prod_inter

        u_les    = np.stack([self._gaussian_filter_z(vel_z[..., c])
                             for c in range(n_c)], axis=-1)         # [M, M, M, 3]
        bar_uiuj = np.stack([self._gaussian_filter_z(prod_z[..., idx])
                             for idx in range(n_p)], axis=-1)       # [M, M, M, 6]
        del vel_z, prod_z

        print("  Computing τᵢⱼ = bar(uᵢuⱼ) − bar(uᵢ)·bar(uⱼ) …")
        tau = np.zeros((M, M, M, 3, 3), dtype=np.float32)
        for idx, (i, j) in enumerate(PAIRS):
            t_ij = (bar_uiuj[..., idx].astype(np.float64)
                    - u_les[..., i].astype(np.float64)
                    * u_les[..., j].astype(np.float64)).astype(np.float32)
            tau[..., i, j] = t_ij
            tau[..., j, i] = t_ij

        progress_path.unlink(missing_ok=True)
        return tau, u_les[..., 0], u_les[..., 1], u_les[..., 2]

    def _gaussian_filter_xy(self, u_slab: np.ndarray) -> np.ndarray:
        """
        Apply 1D Gaussian LES filter and downsample along x then y.

        G(k) = exp(−k² · (fw/2π)² / 24)  evaluated at wavenumbers k=0..N//2.
        Truncates to |k| ≤ M//2 (LES Nyquist) and scales by M/N.

        Input:  [fw, N, N] DNS slab (one component or product).
        Output: [fw, M, M] Gaussian-filtered at LES (x, y) resolution.
        """
        fw, N, _ = u_slab.shape
        M   = self.LES_N          # 64
        km  = M // 2              # 32
        # Δ = filter_width × dx_dns = filter_width × 2π/N_dns; sig = Δ²/24
        sig = (self.FILTER_WIDTH * 2 * np.pi / self.DNS_N) ** 2 / 24.0
        k   = np.fft.rfftfreq(N, d=1.0 / N).astype(np.float32)   # [N//2+1]
        G   = np.exp(-k**2 * sig).astype(np.float32)

        # Filter along x (axis 2): [fw, N, N] → [fw, N, M]
        Ux    = np.fft.rfft(u_slab, axis=2)            # [fw, N, N//2+1]
        Ux   *= G[np.newaxis, np.newaxis, :]
        Ux_l  = Ux[:, :, :km + 1] * (M / N)           # truncate + scale
        u_x   = np.fft.irfft(Ux_l, n=M, axis=2).astype(np.float32)  # [fw, N, M]
        del Ux, Ux_l

        # Filter along y (axis 1): [fw, N, M] → [fw, M, M]
        Uy    = np.fft.rfft(u_x, axis=1)               # [fw, N//2+1, M]
        Uy   *= G[:, np.newaxis]
        Uy_l  = Uy[:, :km + 1, :] * (M / N)
        u_xy  = np.fft.irfft(Uy_l, n=M, axis=1).astype(np.float32)  # [fw, M, M]
        del Uy, Uy_l, u_x

        return u_xy

    def _gaussian_filter_z(self, u_z: np.ndarray) -> np.ndarray:
        """
        Apply 1D Gaussian LES filter and downsample along z.

        Input:  [N, M, M] field at LES (x,y) but full DNS z-resolution.
        Output: [M, M, M] fully filtered at LES resolution.
        """
        N   = self.DNS_N          # 1024
        M   = self.LES_N          # 64
        km  = M // 2              # 32
        sig = (self.FILTER_WIDTH * 2 * np.pi / self.DNS_N) ** 2 / 24.0
        k   = np.fft.rfftfreq(N, d=1.0 / N).astype(np.float32)
        G   = np.exp(-k**2 * sig).astype(np.float32)

        Uz   = np.fft.rfft(u_z, axis=0)                # [N//2+1, M, M]
        Uz  *= G[:, np.newaxis, np.newaxis]
        Uz_l = Uz[:km + 1, :, :] * (M / N)
        u_out = np.fft.irfft(Uz_l, n=M, axis=0).astype(np.float32)  # [M, M, M]
        del Uz, Uz_l

        return u_out


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
