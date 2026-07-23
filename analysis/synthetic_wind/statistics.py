"""Turbulence statistics from RANS fields on the regular grid."""
from __future__ import annotations

import numpy as np

from .config import C_MU


def velocity_gradients(
    U: np.ndarray,
    coords_x: np.ndarray,
    coords_y: np.ndarray,
    coords_z: np.ndarray,
) -> np.ndarray:
    """
    Compute dU_i/dx_j for U shape (3, nx, ny, nz).
    Returns grad[i, j, ...] = dU_i/dx_j with j=0,1,2 for x,y,z.
    """
    grad = np.zeros((3, 3) + U.shape[1:], dtype=np.float64)
    spacing = [
        np.diff(coords_x) if len(coords_x) > 1 else np.array([1.0]),
        np.diff(coords_y) if len(coords_y) > 1 else np.array([1.0]),
        np.diff(coords_z) if len(coords_z) > 1 else np.array([1.0]),
    ]
    for i in range(3):
        g = np.gradient(U[i], coords_x, coords_y, coords_z, axis=(0, 1, 2))
        for j in range(3):
            grad[i, j] = g[j]
    return grad


def reynolds_stress_tensor(
    U: np.ndarray,
    k: np.ndarray,
    nut: np.ndarray,
    coords_x: np.ndarray,
    coords_y: np.ndarray,
    coords_z: np.ndarray,
    k_floor: float = 1e-6,
    nut_floor: float = 0.0,
) -> np.ndarray:
    """
    Boussinesq R_ij = 2/3 k delta_ij - nut (dU_i/dx_j + dU_j/dx_i).
    Returns R shape (3, 3, nx, ny, nz).
    """
    k_safe = np.maximum(k.squeeze() if k.ndim == 4 else k, k_floor)
    nut_safe = np.maximum(nut.squeeze() if nut.ndim == 4 else nut, nut_floor)
    grad = velocity_gradients(U, coords_x, coords_y, coords_z)

    nx, ny, nz = U.shape[1:]
    R = np.zeros((3, 3, nx, ny, nz), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            strain = grad[i, j] + grad[j, i]
            R[i, j] = -nut_safe * strain
            if i == j:
                R[i, j] += (2.0 / 3.0) * k_safe
    return R


def integral_scales(
    k: np.ndarray,
    epsilon: np.ndarray,
    eps_floor: float = 1e-8,
    k_floor: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (L, T) with L = C_mu^0.75 k^1.5/eps and T = k/eps."""
    k2 = k.squeeze() if k.ndim == 4 else k
    e2 = epsilon.squeeze() if epsilon.ndim == 4 else epsilon
    k_safe = np.maximum(k2, k_floor)
    e_safe = np.maximum(e2, eps_floor)
    L = (C_MU ** 0.75) * (k_safe ** 1.5) / e_safe
    T = k_safe / e_safe
    return L, T


def cholesky_3x3_single(R: np.ndarray) -> np.ndarray:
    """Robust Cholesky for one 3x3 Reynolds-stress tensor."""
    M = 0.5 * (R + R.T)
    w, v = np.linalg.eigh(M)
    w = np.maximum(w, 1e-10)
    M = (v * w) @ v.T
    return np.linalg.cholesky(M)


def cholesky_3x3(R: np.ndarray) -> np.ndarray:
    """
    Vectorized Cholesky for R stored as (3,3,nx,ny,nz).
    Uses batched 3x3 analytic factorization after PSD regularization.
    """
    shape = R.shape[2:]
    n = int(np.prod(shape))
    A = R.reshape(3, 3, n).transpose(2, 0, 1).astype(np.float64, copy=True)
    A = 0.5 * (A + A.transpose(0, 2, 1))
    # Clip diagonal for numerical stability
    for i in range(3):
        A[:, i, i] = np.maximum(A[:, i, i], 1e-10)

    L = np.zeros_like(A)
    L[:, 0, 0] = np.sqrt(A[:, 0, 0])
    L[:, 1, 0] = A[:, 1, 0] / L[:, 0, 0]
    L[:, 1, 1] = np.sqrt(np.maximum(A[:, 1, 1] - L[:, 1, 0] ** 2, 1e-12))
    L[:, 2, 0] = A[:, 2, 0] / L[:, 0, 0]
    L[:, 2, 1] = (A[:, 2, 1] - L[:, 2, 0] * L[:, 1, 0]) / L[:, 1, 1]
    L[:, 2, 2] = np.sqrt(
        np.maximum(A[:, 2, 2] - L[:, 2, 0] ** 2 - L[:, 2, 1] ** 2, 1e-12)
    )
    return L.transpose(1, 2, 0).reshape(3, 3, *shape)


def turbulence_intensity(R: np.ndarray, U: np.ndarray) -> np.ndarray:
    """TI = sigma_u / |U_h| using R_00 as sigma_u^2."""
    sigma_u = np.sqrt(np.maximum(R[0, 0], 0.0))
    U_h = np.sqrt(U[0] ** 2 + U[1] ** 2 + U[2] ** 2)
    U_h = np.maximum(U_h, 0.5)
    return sigma_u / U_h


def summarize_case_stats(
    k: np.ndarray,
    epsilon: np.ndarray,
    U: np.ndarray,
    coords_z: np.ndarray,
) -> dict:
    """Domain / aloft summary for stability classification."""
    k2 = k.squeeze() if k.ndim == 4 else k
    z = coords_z
    k_aloft = float(np.nanmean(k2[:, :, z > 500.0])) if np.any(z > 500.0) else float(np.nanmean(k2))
    k_max = float(np.nanmax(k2))
    U_h = np.sqrt(U[0] ** 2 + U[1] ** 2)
    # LLJ proxy on column-averaged profile
    prof_ws = []
    prof_z = []
    for iz, zz in enumerate(z):
        if 50.0 < zz < 600.0:
            prof_ws.append(float(np.nanmean(U_h[:, :, iz])))
            prof_z.append(zz)
    if prof_ws:
        u_max = max(prof_ws)
        high_z = z[z > min(1500.0, float(z.max()))]
        if high_z.size > 0:
            u_top = float(np.nanmean(U_h[:, :, z > min(1500.0, float(z.max()))]))
        else:
            u_top = float(np.nanmean(U_h[:, :, -3:]))
    else:
        u_max = float(np.nanmax(U_h))
        u_top = float(np.nanmean(U_h[:, :, -1]))
    return {
        "k_aloft": k_aloft,
        "k_max": k_max,
        "u_max_low": u_max,
        "u_top": u_top,
        "llj": bool(u_max > 1.1 * u_top and u_max > 3.0),
    }


def classify_stability(summary: dict) -> str:
    """Three-class stability from docs/methodology/abl_stability_and_llj_detection.md."""
    if summary["k_aloft"] < 0.05 and summary["k_max"] < 0.6:
        return "strongly_stable"
    if summary["k_aloft"] > 0.2 or summary["k_max"] > 1.5:
        return "unstable"
    return "neutral"
