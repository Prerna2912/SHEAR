"""
Vectorized 2D panel-method kernels (vortex and source).

All functions operate on (M,) field-point arrays and (N,) panel arrays,
using NumPy broadcasting to produce (M, N) intermediates.

Reference: Katz & Plotkin, "Low-Speed Aerodynamics" (2001), §3.

Notation
--------
gamma : [N] vortex strength (circulation per unit length)
sigma : [N] source strength (volume flux per unit length)
l     : [N] panel lengths
theta : [N] panel angles (atan2(dy, dx))
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def panel_geometry(px: np.ndarray, py: np.ndarray) -> tuple:
    """
    Compute panel geometry from (N+1) node coordinates.

    Args:
        px, py : [N+1] panel node x,y coordinates.

    Returns:
        x1, y1 : [N] panel start points.
        x2, y2 : [N] panel end points.
        xc, yc : [N] control points (midpoints).
        l      : [N] panel lengths.
        theta  : [N] panel angles (atan2(dy, dx)).
        nx, ny : [N] outward panel normals  (-sin θ, cos θ).
    """
    x1, y1 = px[:-1], py[:-1]
    x2, y2 = px[1:],  py[1:]

    dx = x2 - x1
    dy = y2 - y1
    l = np.sqrt(dx**2 + dy**2)

    theta = np.arctan2(dy, dx)
    nx = -np.sin(theta)
    ny =  np.cos(theta)

    xc = 0.5 * (x1 + x2)
    yc = 0.5 * (y1 + y2)

    return x1, y1, x2, y2, xc, yc, l, theta, nx, ny


# ---------------------------------------------------------------------------
# Vortex panel kernel  (lifting surfaces)
# ---------------------------------------------------------------------------

def _vortex_velocity(
    xp: np.ndarray,   # [M]
    yp: np.ndarray,   # [M]
    x1: np.ndarray,   # [N]
    y1: np.ndarray,   # [N]
    l:  np.ndarray,   # [N]
    theta: np.ndarray,# [N]
    gamma: np.ndarray,# [N]  (unit strength when building influence matrix)
) -> tuple:
    """
    Velocity at M field points due to N constant-strength vortex panels.

    Returns (u_x, u_y) in the global frame, each shape [M].
    """
    cos_th = np.cos(theta)   # [N]
    sin_th = np.sin(theta)   # [N]

    # Relative position in global frame  → [M, N]
    dxp = xp[:, None] - x1[None, :]   # [M, N]
    dyp = yp[:, None] - y1[None, :]   # [M, N]

    # Panel-local coordinates
    xi  =  dxp * cos_th[None, :] + dyp * sin_th[None, :]   # [M, N]
    eta = -dxp * sin_th[None, :] + dyp * cos_th[None, :]   # [M, N]

    eps = 1e-10
    r1_sq = xi**2 + eta**2 + eps                              # [M, N]
    r2_sq = (xi - l[None, :])**2 + eta**2 + eps              # [M, N]

    # Induced velocity in panel-local frame (Katz & Plotkin Eq. 3.98)
    g = gamma[None, :]                                         # [1, N]
    u_xi  = g / (4.0 * np.pi) * np.log(r1_sq / r2_sq)
    u_eta = -g / (2.0 * np.pi) * (
        np.arctan2(eta, xi)
        - np.arctan2(eta, xi - l[None, :])
    )

    # Rotate back to global frame and sum over panels  → [M]
    u_x = np.sum(u_xi * cos_th[None, :] - u_eta * sin_th[None, :], axis=1)
    u_y = np.sum(u_xi * sin_th[None, :] + u_eta * cos_th[None, :], axis=1)

    return u_x, u_y


def _vortex_normal_influence(
    xc: np.ndarray,    # [N] control-point x
    yc: np.ndarray,    # [N] control-point y
    x1: np.ndarray,    # [N] panel start x
    y1: np.ndarray,    # [N] panel start y
    l:  np.ndarray,    # [N] panel lengths
    theta: np.ndarray, # [N] panel angles
    nx: np.ndarray,    # [N] control-point normals x
    ny: np.ndarray,    # [N] control-point normals y
) -> np.ndarray:
    """
    Influence matrix A[i,j] = normal velocity at control point i
    due to panel j with unit vortex strength.

    Returns A: [N, N].
    """
    unit_gamma = np.ones(len(x1))
    N = len(xc)
    A = np.zeros((N, N))
    for j in range(N):
        ux, uy = _vortex_velocity(
            xc, yc,
            x1[j:j+1], y1[j:j+1], l[j:j+1], theta[j:j+1],
            unit_gamma[j:j+1],
        )
        A[:, j] = ux * nx + uy * ny
    return A


def solve_vortex_panels(
    px: np.ndarray,
    py: np.ndarray,
    alpha: float,
    U_inf: float = 1.0,
    kutta: bool = True,
) -> np.ndarray:
    """
    Solve for panel vortex strengths γ on a closed 2D lifting body.

    Args:
        px, py : [N+1] panel node coordinates (counterclockwise, TE→upper→LE→lower→TE).
        alpha  : angle of attack (radians).
        U_inf  : freestream speed.
        kutta  : whether to impose the Kutta condition γ₀ + γ_{N-1} = 0.

    Returns:
        gamma : [N] vortex strength per unit length on each panel.
    """
    x1, y1, x2, y2, xc, yc, l, theta, nx, ny = panel_geometry(px, py)
    N = len(xc)

    A = _vortex_normal_influence(xc, yc, x1, y1, l, theta, nx, ny)

    # Freestream normal component at each control point
    Ux = U_inf * np.cos(alpha)
    Uy = U_inf * np.sin(alpha)
    rhs = -(Ux * nx + Uy * ny)

    if kutta and N >= 2:
        # Replace last row with Kutta condition
        A[-1, :]  = 0.0
        A[-1, 0]  = 1.0
        A[-1, -1] = 1.0
        rhs[-1]   = 0.0

    gamma = np.linalg.solve(A, rhs)
    return gamma


def vortex_velocity_field(
    xp: np.ndarray,
    yp: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    gamma: np.ndarray,
    U_inf: float = 1.0,
    alpha: float = 0.0,
) -> tuple:
    """
    Evaluate total (freestream + panel) velocity at field points.

    Args:
        xp, yp : [M] field point coordinates.
        px, py : [N+1] panel node coordinates.
        gamma  : [N] solved panel vortex strengths.
        U_inf  : freestream speed.
        alpha  : angle of attack (radians).

    Returns:
        u_x, u_y : [M] velocity components.
    """
    x1, y1, *_, l, theta, _, _ = panel_geometry(px, py)

    u_x, u_y = _vortex_velocity(xp, yp, x1, y1, l, theta, gamma)
    u_x += U_inf * np.cos(alpha)
    u_y += U_inf * np.sin(alpha)

    return u_x, u_y


# ---------------------------------------------------------------------------
# Source panel kernel  (non-lifting bodies)
# ---------------------------------------------------------------------------

def _source_velocity(
    xp: np.ndarray,
    yp: np.ndarray,
    x1: np.ndarray,
    y1: np.ndarray,
    l:  np.ndarray,
    theta: np.ndarray,
    sigma: np.ndarray,
) -> tuple:
    """
    Velocity at M field points due to N constant-strength source panels.

    Returns (u_x, u_y) in the global frame, each shape [M].
    """
    cos_th = np.cos(theta)
    sin_th = np.sin(theta)

    dxp = xp[:, None] - x1[None, :]
    dyp = yp[:, None] - y1[None, :]

    xi  =  dxp * cos_th[None, :] + dyp * sin_th[None, :]
    eta = -dxp * sin_th[None, :] + dyp * cos_th[None, :]

    eps = 1e-10
    r1_sq = xi**2 + eta**2 + eps
    r2_sq = (xi - l[None, :])**2 + eta**2 + eps

    # Katz & Plotkin Eq. 3.81 for source panel
    s = sigma[None, :]
    u_xi  = s / (4.0 * np.pi) * np.log(r1_sq / r2_sq)
    u_eta =  s / (2.0 * np.pi) * (
        np.arctan2(eta, xi)
        - np.arctan2(eta, xi - l[None, :])
    )

    u_x = np.sum(u_xi * cos_th[None, :] - u_eta * sin_th[None, :], axis=1)
    u_y = np.sum(u_xi * sin_th[None, :] + u_eta * cos_th[None, :], axis=1)

    return u_x, u_y


def solve_source_panels(
    px: np.ndarray,
    py: np.ndarray,
    alpha: float,
    U_inf: float = 1.0,
) -> np.ndarray:
    """
    Solve for source panel strengths σ on a closed 2D non-lifting body.

    Args:
        px, py : [N+1] panel node coordinates (counterclockwise).
        alpha  : flow angle (radians).
        U_inf  : freestream speed.

    Returns:
        sigma : [N] source strength per unit length on each panel.
    """
    x1, y1, x2, y2, xc, yc, l, theta, nx, ny = panel_geometry(px, py)
    N = len(xc)

    unit_sigma = np.ones(N)
    A = np.zeros((N, N))
    for j in range(N):
        ux, uy = _source_velocity(
            xc, yc,
            x1[j:j+1], y1[j:j+1], l[j:j+1], theta[j:j+1],
            unit_sigma[j:j+1],
        )
        A[:, j] = ux * nx + uy * ny

    Ux = U_inf * np.cos(alpha)
    Uy = U_inf * np.sin(alpha)
    rhs = -(Ux * nx + Uy * ny)

    sigma = np.linalg.solve(A, rhs)
    return sigma


def source_velocity_field(
    xp: np.ndarray,
    yp: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    sigma: np.ndarray,
    U_inf: float = 1.0,
    alpha: float = 0.0,
) -> tuple:
    """
    Evaluate total (freestream + panel) velocity at field points.

    Returns:
        u_x, u_y : [M] velocity components.
    """
    x1, y1, *_, l, theta, _, _ = panel_geometry(px, py)

    u_x, u_y = _source_velocity(xp, yp, x1, y1, l, theta, sigma)
    u_x += U_inf * np.cos(alpha)
    u_y += U_inf * np.sin(alpha)

    return u_x, u_y


# ---------------------------------------------------------------------------
# NACA 4-digit profile generator
# ---------------------------------------------------------------------------

def naca4_coords(
    m: float,
    p: float,
    t: float,
    n_panels: int = 80,
    chord: float = 1.0,
    te_bunching: float = 1.0,
) -> tuple:
    """
    Generate NACA 4-digit airfoil coordinates.

    Args:
        m         : maximum camber / chord (e.g. 0.02 for NACA 2412).
        p         : chordwise location of max camber (e.g. 0.4).
        t         : max thickness / chord (e.g. 0.12).
        n_panels  : number of panels (even number → half on each surface).
        chord     : chord length (m).
        te_bunching: cosine bunching exponent (1=uniform, 2=cosine).

    Returns:
        px, py : [n_panels+1] panel node coordinates.
                 Order: TE (1,0) → upper surface → LE (0,0) → lower surface → TE.
    """
    n_half = n_panels // 2

    # Cosine-spaced panel nodes from TE (β=0) to LE (β=π) and back
    beta = np.linspace(0, np.pi, n_half + 1)
    x_upper = 0.5 * (1.0 - np.cos(beta))                    # 0 at LE, 1 at TE

    # NACA thickness distribution
    def thickness(x):
        return t / 0.2 * (
            0.2969 * np.sqrt(np.clip(x, 0, 1))
            - 0.1260 * x
            - 0.3516 * x**2
            + 0.2843 * x**3
            - 0.1015 * x**4
        )

    # Camber line
    def camber(x):
        yc = np.where(
            x < p,
            m / p**2 * (2 * p * x - x**2),
            m / (1 - p)**2 * ((1 - 2 * p) + 2 * p * x - x**2),
        )
        return yc if p > 1e-6 else np.zeros_like(x)

    def camber_slope(x):
        dy = np.where(
            x < p,
            2 * m / p**2 * (p - x),
            2 * m / (1 - p)**2 * (p - x),
        )
        return dy if p > 1e-6 else np.zeros_like(x)

    yt = thickness(x_upper)
    yc = camber(x_upper)
    theta = np.arctan(camber_slope(x_upper))

    xu = (x_upper - yt * np.sin(theta)) * chord
    yu = (yc     + yt * np.cos(theta)) * chord

    x_lower = x_upper[::-1]
    yt_l = thickness(x_lower)
    yc_l = camber(x_lower)
    theta_l = np.arctan(camber_slope(x_lower))

    xl = (x_lower + yt_l * np.sin(theta_l)) * chord
    yl = (yc_l   - yt_l * np.cos(theta_l)) * chord

    # Concatenate: upper (TE→LE) + lower (LE→TE)
    px = np.concatenate([xu[::-1], xl[1:]])   # TE first
    py = np.concatenate([yu[::-1], yl[1:]])

    return px.astype(np.float64), py.astype(np.float64)
