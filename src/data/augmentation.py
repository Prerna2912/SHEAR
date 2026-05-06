"""
Rotation augmentation for V3 (non-equivariant CFM).
The 24 orientation-preserving symmetries of the cube (chiral octahedral symmetry)
are the SO(3) rotations that map the cubic lattice to itself.
"""

import numpy as np
import torch


def _build_cubic_rotations() -> np.ndarray:
    """
    Enumerate all 24 orientation-preserving rotation matrices of the cube.
    Each is a 3x3 integer matrix with det=+1.
    """
    # Generators: Rx(90°), Ry(90°), Rz(90°)
    def Rx(n):
        c, s = int(np.round(np.cos(n * np.pi / 2))), int(np.round(np.sin(n * np.pi / 2)))
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    def Ry(n):
        c, s = int(np.round(np.cos(n * np.pi / 2))), int(np.round(np.sin(n * np.pi / 2)))
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    def Rz(n):
        c, s = int(np.round(np.cos(n * np.pi / 2))), int(np.round(np.sin(n * np.pi / 2)))
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    mats = set()
    identity = np.eye(3, dtype=np.int32)
    queue = [identity]
    generators = [Rx(1), Ry(1), Rz(1), Rx(-1), Ry(-1), Rz(-1)]

    while queue:
        R = queue.pop()
        key = tuple(R.flatten())
        if key in mats:
            continue
        mats.add(key)
        for G in generators:
            R2 = G @ R
            if tuple(R2.flatten()) not in mats:
                queue.append(R2)

    rotations = np.array([np.array(list(k)).reshape(3, 3) for k in mats])
    # Keep only proper rotations (det = +1)
    rotations = np.array([R for R in rotations if int(np.round(np.linalg.det(R))) == 1])
    assert len(rotations) == 24, f"Expected 24 proper cube rotations, got {len(rotations)}"
    return rotations.astype(np.float32)


# Precomputed at import time — used throughout training for V3
CUBIC_ROTATIONS_SO3: np.ndarray = _build_cubic_rotations()  # [24, 3, 3]


def rotate_velocity_gradient(grad_u: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """
    Apply rotation R to the velocity gradient tensor.

    grad_u[..., i, j] = du_i/dx_j transforms as a rank-2 tensor:
        (R grad_u R^T)[i,j] = R_{ik} R_{jl} grad_u[k, l]

    Args:
        grad_u: [..., 3, 3]
        R:      [3, 3]

    Returns:
        Rotated gradient [..., 3, 3].
    """
    return R @ grad_u @ R.T


def rotate_sgs_stress(tau: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """
    Apply rotation R to the SGS stress tensor (rank-2 symmetric).

    Args:
        tau: [..., 3, 3]
        R:   [3, 3]

    Returns:
        Rotated stress [..., 3, 3].
    """
    return R @ tau @ R.T


def apply_cubic_rotations(
    grad_u: torch.Tensor,
    tau: torch.Tensor,
    n_rotations: int = 24,
    device: torch.device = torch.device('cpu'),
) -> tuple:
    """
    Augment a batch by applying a random subset of the 24 cubic rotations.

    Args:
        grad_u: [B, 3, 3] velocity gradient per sample.
        tau:    [B, 3, 3] SGS stress per sample.
        n_rotations: how many of the 24 to apply (default = all 24).

    Returns:
        aug_grad_u, aug_tau: [B * n_rotations, 3, 3] augmented tensors.
    """
    rots = CUBIC_ROTATIONS_SO3[:n_rotations]  # [n_rotations, 3, 3]
    R_all = torch.from_numpy(rots).to(device)  # [n_rotations, 3, 3]

    aug_grad, aug_tau = [], []
    for R in R_all:
        aug_grad.append(rotate_velocity_gradient(grad_u, R))
        aug_tau.append(rotate_sgs_stress(tau, R))

    return torch.cat(aug_grad, dim=0), torch.cat(aug_tau, dim=0)


def random_so3_rotation(n: int = 1, device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """
    Sample n uniformly random SO(3) rotation matrices via QR decomposition.

    Returns:
        [n, 3, 3] rotation matrices.
    """
    A = torch.randn(n, 3, 3, device=device)
    Q, R = torch.linalg.qr(A)
    # Fix signs so det(Q) = +1
    signs = torch.sign(torch.det(Q)).unsqueeze(-1).unsqueeze(-1)
    Q = Q * signs
    return Q
