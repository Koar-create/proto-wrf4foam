"""
Random Flow Generation (RFG) — Smirnov & Celik (2001) style synthetic turbulence.

Practical grid-based implementation:
  1. Gaussian-filtered white noise per component (von Kármán-like spatial correlation)
  2. Cholesky transform from local Reynolds-stress tensor R_ij
  3. AR(1) temporal evolution with integral time scale T = k/epsilon
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from .statistics import cholesky_3x3, cholesky_3x3_single


def von_karman_spectrum_kappa(kappa: np.ndarray, L: float) -> np.ndarray:
    """Isotropic von Kármán energy spectrum E(kappa) ~ kappa^4 / (1 + kappa^2)^(17/6)."""
    k = np.maximum(kappa * L, 1e-12)
    return (k ** 4) / ((1.0 + k ** 2) ** (17.0 / 6.0))


def _gaussian_sigma_cells(L: float, dx: float) -> float:
    """Map integral length scale L to Gaussian filter sigma (cells)."""
    return max(L / (dx * np.sqrt(np.pi)), 0.5)


class SyntheticTurbulenceGenerator:
    """Generate u'(x,t) on a regular grid with target R_ij, L, T."""

    def __init__(
        self,
        R: np.ndarray,
        L: np.ndarray,
        T: np.ndarray,
        coords_x: np.ndarray,
        coords_y: np.ndarray,
        coords_z: np.ndarray,
        seed: int = 42,
    ):
        self.R = R
        self.L = L
        self.T = T
        self.coords_x = coords_x
        self.coords_y = coords_y
        self.coords_z = coords_z
        self.dx = float(coords_x[1] - coords_x[0]) if len(coords_x) > 1 else 40.0
        self.dy = float(coords_y[1] - coords_y[0]) if len(coords_y) > 1 else self.dx
        self.dz = np.diff(coords_z) if len(coords_z) > 1 else np.array([10.0])
        self.shape = R.shape[2:]
        self.L_chol = cholesky_3x3(R)
        self.rng = np.random.default_rng(seed)
        self._state: np.ndarray | None = None

    def _spatial_filter(self, noise: np.ndarray) -> np.ndarray:
        """Component-wise anisotropic Gaussian smoothing using local L."""
        L_mean = float(np.nanmedian(self.L))
        sig_x = _gaussian_sigma_cells(L_mean, self.dx)
        sig_y = _gaussian_sigma_cells(L_mean, self.dy)
        sig_z = _gaussian_sigma_cells(L_mean, float(np.nanmedian(np.diff(self.coords_z))))
        out = np.zeros_like(noise)
        for c in range(3):
            out[c] = gaussian_filter(
                noise[c], sigma=(sig_x, sig_y, sig_z), mode="nearest"
            )
        return out

    def _apply_anisotropy(self, filtered: np.ndarray) -> np.ndarray:
        """u'_i = L_ij g_j — fully vectorized."""
        nx, ny, nz = self.shape
        n = nx * ny * nz
        g = filtered.reshape(3, n).T
        L = self.L_chol.reshape(3, 3, n).transpose(2, 0, 1)
        u = np.einsum("nij,nj->ni", L, g)
        return u.T.reshape(3, nx, ny, nz)

    def _normalize_variance(self, u_prime: np.ndarray) -> np.ndarray:
        """Pointwise scaling so RMS(u'_i) ~ sqrt(R_ii) across the domain."""
        out = u_prime.copy()
        for i in range(3):
            rms_map = np.sqrt(np.maximum(self.R[i, i], 1e-12))
            rms_out = np.sqrt(np.mean(out[i] ** 2))
            if rms_out > 1e-12:
                out[i] *= rms_map / rms_out
        return out

    def generate_innovation(self) -> np.ndarray:
        """One spatially correlated, anisotropic fluctuation field."""
        noise = self.rng.standard_normal((3,) + self.shape)
        filtered = self._spatial_filter(noise)
        u_prime = self._apply_anisotropy(filtered)
        return self._normalize_variance(u_prime)

    def step(self, dt: float) -> np.ndarray:
        """AR(1) step: u'(t+dt) = a u'(t) + b * innovation."""
        T_med = float(np.nanmedian(self.T))
        T_med = max(T_med, dt)
        a = np.exp(-dt / T_med)
        b = np.sqrt(max(1.0 - a * a, 0.0))
        innov = self.generate_innovation()
        if self._state is None:
            self._state = innov
        else:
            self._state = a * self._state + b * innov
        return self._state.copy()

    def generate_series(
        self,
        n_steps: int,
        dt: float,
        mean_u: np.ndarray | None = None,
        burn_in: int = 50,
    ) -> np.ndarray:
        """
        Return U(t) shape (n_steps, 3, nx, ny, nz).
        If mean_u provided, add to fluctuations (frozen mean within window).
        """
        series = np.zeros((n_steps, 3) + self.shape, dtype=np.float32)
        self._state = None
        for _ in range(burn_in):
            self.step(dt)
        for t in range(n_steps):
            up = self.step(dt)
            if mean_u is not None:
                series[t] = (mean_u + up).astype(np.float32)
            else:
                series[t] = up.astype(np.float32)

        # Calibrate temporal variance to match R_ii (ergodic adjustment)
        if n_steps > 10:
            fluc = series - np.mean(series, axis=0, keepdims=True)
            for i in range(3):
                var_t = np.var(fluc[:, i], axis=0)
                target = np.maximum(self.R[i, i], 1e-12)
                scale = np.sqrt(target / np.maximum(var_t, 1e-12))
                series[:, i] = (mean_u[i] if mean_u is not None else 0.0) + fluc[:, i] * scale
        return series

    def _ar1_probe(
        self,
        R_loc: np.ndarray,
        T_loc: float,
        dt: float,
        n_steps: int,
        burn_in: int = 50,
    ) -> np.ndarray:
        """Single-probe AR(1) fluctuation series, shape (n_steps, 3)."""
        T_loc = max(T_loc, dt)
        a = np.exp(-dt / T_loc)
        b = np.sqrt(max(1.0 - a * a, 0.0))
        L_loc = cholesky_3x3_single(R_loc)
        series = np.zeros((n_steps, 3), dtype=np.float64)
        state = np.zeros(3, dtype=np.float64)
        for _ in range(burn_in):
            state = a * state + b * (L_loc @ self.rng.standard_normal(3))
        for t in range(n_steps):
            state = a * state + b * (L_loc @ self.rng.standard_normal(3))
            series[t] = state
        return series

    def generate_probe_series(
        self,
        indices: list[tuple[int, int, int]],
        probe_meta: list[dict],
        n_steps: int,
        dt: float,
        mean_u: np.ndarray | None = None,
        coh_length_m: float = 100.0,
    ) -> np.ndarray:
        """
        Fast probe AR(1) with Davenport-style vertical coherence within each site.
        """
        from collections import defaultdict

        n_probes = len(indices)
        out = np.zeros((n_steps, n_probes, 3), dtype=np.float64)

        site_groups: dict[str, list[tuple[int, float, tuple[int, int, int]]]] = defaultdict(list)
        for p, meta in enumerate(probe_meta):
            site = meta["name"].split("_z")[0]
            site_groups[site].append((p, float(meta["z"]), indices[p]))

        for _site, members in site_groups.items():
            members = sorted(members, key=lambda x: x[1])
            ref_p, ref_z, (ix0, iy0, iz0) = members[0]
            R0 = self.R[:, :, ix0, iy0, iz0]
            T0 = float(self.T[ix0, iy0, iz0])
            fluc = self._ar1_probe(R0, T0, dt, n_steps)
            if mean_u is not None:
                out[:, ref_p, :] = fluc + mean_u[:, ix0, iy0, iz0]
            else:
                out[:, ref_p, :] = fluc
            self._calibrate_probe(out, ref_p, R0, mean_u, ix0, iy0, iz0)

            prev_p = ref_p
            prev_ix, prev_iy, prev_iz = ix0, iy0, iz0
            prev_z = ref_z
            for p, z, (ix, iy, iz) in members[1:]:
                dz = abs(z - prev_z)
                rho = float(np.exp(-dz / max(coh_length_m, 1.0)))
                R_loc = self.R[:, :, ix, iy, iz]
                T_loc = float(self.T[ix, iy, iz])
                indep = self._ar1_probe(R_loc, T_loc, dt, n_steps)
                if mean_u is not None:
                    fluc_prev = out[:, prev_p, :] - mean_u[:, prev_ix, prev_iy, prev_iz]
                    mu_p = mean_u[:, ix, iy, iz]
                else:
                    fluc_prev = out[:, prev_p, :]
                    mu_p = 0.0
                fluc_p = rho * fluc_prev + np.sqrt(1.0 - rho * rho) * indep
                out[:, p, :] = fluc_p + (mu_p if mean_u is not None else 0.0)
                self._calibrate_probe(out, p, R_loc, mean_u, ix, iy, iz)
                prev_p, prev_z = p, z
                prev_ix, prev_iy, prev_iz = ix, iy, iz

        return out.astype(np.float32)

    @staticmethod
    def _calibrate_probe(
        out: np.ndarray,
        p: int,
        R_loc: np.ndarray,
        mean_u: np.ndarray | None,
        ix: int,
        iy: int,
        iz: int,
    ) -> None:
        for comp in range(3):
            if mean_u is not None:
                fluc = out[:, p, comp] - mean_u[comp, ix, iy, iz]
            else:
                fluc = out[:, p, comp]
            target = max(float(R_loc[comp, comp]), 1e-12)
            var_t = max(float(np.var(fluc)), 1e-12)
            sc = np.sqrt(target / var_t)
            if mean_u is not None:
                out[:, p, comp] = mean_u[comp, ix, iy, iz] + fluc * sc
            else:
                out[:, p, comp] *= sc
