"""Wake frame, Fackrell cavity length, and the sample grids used for X_R.

Coordinates: origin at the ground-footprint centroid. x points downstream
(along the local approach-flow direction theta), y points to the left of the
flow, z is height above ground.

A uniform 4 m sample of the full box in the plan (x in [-H, 10H], y in
±3 max(W, H), z up to 1.5H) is 10^7–10^8 points on the tallest towers, which
this mesh cannot be probed at. The grids below keep 4 m spacing where X_R is
actually measured (the center band) and coarsen only the auxiliary 3D cavity
and the plotting plane. See WAKE_XR_METHODS_RESULTS.md.
"""
from __future__ import annotations

import numpy as np

DX_CENTER = 4.0
L_OVER_H_CLIP = (0.3, 3.0)


def wrap_pi(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def basis(theta: float) -> tuple[np.ndarray, np.ndarray]:
    ex = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
    ey = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float64)
    return ex, ey


def project(xy: np.ndarray, origin: np.ndarray, ex: np.ndarray, ey: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rel = np.asarray(xy, dtype=np.float64) - np.asarray(origin, dtype=np.float64)
    return rel @ ex, rel @ ey


def frame_from_hull(
    high_xy: np.ndarray,
    origin: np.ndarray,
    theta: float,
) -> dict:
    """W, L and the leeward-face x from the z >= 0.5H footprint."""
    ex, ey = basis(theta)
    xs, ys = project(high_xy, origin, ex, ey)
    if len(xs) == 0:
        raise ValueError("empty high footprint")
    length = float(xs.max() - xs.min())
    width = float(ys.max() - ys.min())
    # A roof that collapses to a line still needs a finite width for the band.
    length = max(length, 8.0)
    width = max(width, 8.0)
    return {
        "theta": float(theta),
        "ex": ex,
        "ey": ey,
        "L": length,
        "W": width,
        "x_lee": float(xs.max()),
        "x_wind": float(xs.min()),
        "y_min": float(ys.min()),
        "y_max": float(ys.max()),
    }


def fackrell_lr(width: float, height: float, length: float) -> float:
    """Recirculation-cavity length from the lee face, Fackrell (1984).

    L/H is clipped to [0.3, 3] as in AERMOD-PRIME when the building is outside
    the range the fit was built on.
    """
    wh = width / height
    lh = float(np.clip(length / height, L_OVER_H_CLIP[0], L_OVER_H_CLIP[1]))
    return float(height * 1.8 * wh / (lh**0.3 * (1.0 + 0.24 * wh)))


def approach_ring(cx: float, cy: float, height: float) -> np.ndarray:
    """Points on the 2.2H–3.8H ring at z = H/2. Used to estimate theta."""
    radii = (2.2 * height, 3.0 * height, 3.8 * height)
    az = np.linspace(0.0, 2.0 * np.pi, 72, endpoint=False)
    pts = []
    z = 0.5 * height
    for r in radii:
        for a in az:
            pts.append((cx + r * np.cos(a), cy + r * np.sin(a), z))
    return np.asarray(pts, dtype=np.float64)


def theta_from_ring(
    xyz: np.ndarray,
    uxy: np.ndarray,
    valid: np.ndarray,
    cx: float,
    cy: float,
) -> tuple[float, float, int]:
    """Approach direction from the upstream part of the ring.

    Returns theta (atan2(v, u), the direction the wind blows toward),
    the mean horizontal speed on that sector, and the number of points used.
    """
    valid = np.asarray(valid, dtype=bool) & np.isfinite(uxy).all(axis=1)
    if int(valid.sum()) < 5:
        return np.nan, np.nan, int(valid.sum())
    mean0 = uxy[valid].mean(axis=0)
    theta0 = float(np.arctan2(mean0[1], mean0[0]))
    pos = np.arctan2(xyz[:, 1] - cy, xyz[:, 0] - cx)
    upstream = valid & (np.abs(wrap_pi(pos - (theta0 + np.pi))) <= np.deg2rad(70.0))
    if int(upstream.sum()) < 8:
        upstream = valid
    mean = uxy[upstream].mean(axis=0)
    theta = float(np.arctan2(mean[1], mean[0]))
    speed = float(np.hypot(mean[0], mean[1]))
    return theta, speed, int(upstream.sum())


def _axis(a: float, b: float, step: float) -> np.ndarray:
    if b < a:
        a, b = b, a
    n = int(np.floor((b - a) / step + 1e-6)) + 1
    return a + step * np.arange(n, dtype=np.float64)


def sample_grids(height: float, width: float, x_lee: float, x_wind: float) -> dict[str, np.ndarray]:
    """Local-frame axes. See module docstring for why these are not a full box."""
    h = float(height)
    w = float(width)
    x_center = _axis(-1.0 * h, 10.0 * h, DX_CENTER)
    half_band = max(w / 4.0, DX_CENTER)
    y_center = _axis(-half_band, half_band, DX_CENTER)
    z_center = np.unique(np.r_[3.0, 0.10 * h, 0.25 * h, 0.50 * h, 1.00 * h])
    z_center = z_center[(z_center >= 2.5) & (z_center <= 1.05 * h)]

    y_lat_max = 3.0 * max(w, h)
    dy_lat = DX_CENTER if y_lat_max <= 400.0 else max(DX_CENTER, (2.0 * y_lat_max) / 160.0)
    y_lat = _axis(-y_lat_max, y_lat_max, dy_lat)
    ks = np.array([0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0])
    x_lat = x_lee + ks * h
    x_lat = x_lat[(x_lat > x_lee) & (x_lat <= 10.0 * h)]
    z_lat = np.unique(np.r_[3.0, 0.25 * h, 0.50 * h])
    z_lat = z_lat[(z_lat >= 2.5) & (z_lat <= h)]

    dx_cav = DX_CENTER if h <= 220.0 else 8.0
    x_cav = _axis(x_lee - 0.25 * h, min(x_lee + 4.0 * h, 10.0 * h), dx_cav)
    y_cav_max = 1.5 * max(w, 16.0)
    y_cav = _axis(-y_cav_max, y_cav_max, dx_cav)
    dz_cav = max(DX_CENTER, h / 30.0)
    z_cav = _axis(3.0, h, dz_cav)

    dx_pl = max(DX_CENTER, h / 60.0)
    x_pl = _axis(-0.5 * h, min(8.0 * h, 10.0 * h), dx_pl)
    y_pl_max = max(2.0 * w, 0.8 * h)
    y_pl = _axis(-y_pl_max, y_pl_max, dx_pl)
    z_pl = np.array([0.25 * h], dtype=np.float64)

    z_amb = _axis(3.0, 1.5 * h, max(DX_CENTER, h / 24.0))
    y_amb = _axis(-0.5 * w, 0.5 * w, max(w / 4.0, DX_CENTER))
    x_amb = x_wind - np.array([2.2, 3.0, 3.8]) * h

    return {
        "x_center": x_center,
        "y_center": y_center,
        "z_center": z_center,
        "x_lat": x_lat,
        "y_lat": y_lat,
        "z_lat": z_lat,
        "x_cav": x_cav,
        "y_cav": y_cav,
        "z_cav": z_cav,
        "x_plane": x_pl,
        "y_plane": y_pl,
        "z_plane": z_pl,
        "x_amb": x_amb,
        "y_amb": y_amb,
        "z_amb": z_amb,
    }


def local_to_global(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    origin: np.ndarray,
    ex: np.ndarray,
    ey: np.ndarray,
) -> np.ndarray:
    """Meshgrid local axes into an (N, 3) global array. x varies fastest, then y, then z."""
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    xy = np.stack([xx, yy], axis=-1)
    gxy = origin + xy[..., 0:1] * ex + xy[..., 1:2] * ey
    out = np.empty(xx.shape + (3,), dtype=np.float64)
    out[..., 0] = gxy[..., 0]
    out[..., 1] = gxy[..., 1]
    out[..., 2] = zz
    return out.reshape(-1, 3)


def count_points(grids: dict[str, np.ndarray]) -> dict[str, int]:
    def n(a, b, c):
        return int(len(grids[a]) * len(grids[b]) * len(grids[c]))

    parts = {
        "center": n("x_center", "y_center", "z_center"),
        "lat": n("x_lat", "y_lat", "z_lat"),
        "cav": n("x_cav", "y_cav", "z_cav"),
        "plane": n("x_plane", "y_plane", "z_plane"),
        "amb": n("x_amb", "y_amb", "z_amb"),
    }
    parts["total"] = sum(parts.values())
    return parts
