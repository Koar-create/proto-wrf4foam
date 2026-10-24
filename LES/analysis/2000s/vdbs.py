"""M4 virtual DBS: four-beam retrieval on a horizontal slice."""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from lesio import read_vtp_points_U, vtp_path
from paths import STATIONS, TIMES

HALF_ANGLES = {
    15.0: "elev 75°",
    18.62: "GAW105 ref., elev 71.38°",
    30.0: "elev 60°",
}


def beam_offset(z, half_angle_deg) -> float:
    return float(z) * np.tan(np.radians(half_angle_deg))


def retrieve_uv(vr: dict, half_angle_deg: float):
    """Standard DBS: u = (vE − vW)/(2 sin θ), v = (vN − vS)/(2 sin θ)."""
    th = np.radians(half_angle_deg)
    s = np.sin(th)
    u = (vr["E"] - vr["W"]) / (2.0 * s)
    v = (vr["N"] - vr["S"]) / (2.0 * s)
    return u, v


def radial_speed(u, v, w, az_deg, half_angle_deg):
    """az_deg: 0 = North, 90 = East. θ_c = cone half-angle from zenith."""
    th = np.radians(half_angle_deg)
    az = np.radians(az_deg)
    # plan: v_r = (u sinφ + v cosφ) sin θ + w cos θ  with N=(0,1), E=(1,0)
    # so φ_N = 0 (v), φ_E = 90 (u)
    return (u * np.sin(az) + v * np.cos(az)) * np.sin(th) + w * np.cos(th)


def dbs_from_grid(u, v, w, dx, half_angle_deg, z):
    """Shifted four-beam retrieval on a regular 2-D array (ny, nx)."""
    th = np.radians(half_angle_deg)
    k = int(round(beam_offset(z, half_angle_deg) / dx))

    def sh(a, dx_, dy):
        out = np.full_like(a, np.nan)
        ny, nx = a.shape
        ys = slice(max(0, dy), ny + min(0, dy))
        yd = slice(max(0, -dy), ny + min(0, -dy))
        xs = slice(max(0, dx_), nx + min(0, dx_))
        xd = slice(max(0, -dx_), nx + min(0, -dx_))
        out[yd, xd] = a[ys, xs]
        return out

    vr = {}
    for name, (ax_, ay_) in {"N": (0, 1), "S": (0, -1), "E": (1, 0), "W": (-1, 0)}.items():
        uu, vv, ww = sh(u, ax_ * k, ay_ * k), sh(v, ax_ * k, ay_ * k), sh(w, ax_ * k, ay_ * k)
        vr[name] = (uu * ax_ + vv * ay_) * np.sin(th) + ww * np.cos(th)
    ud, vd = retrieve_uv(vr, half_angle_deg)
    true = np.hypot(u, v)
    err = (np.hypot(ud, vd) - true) / np.maximum(true, 0.3)
    return err, ud, vd, beam_offset(z, half_angle_deg)


def query_uv_at(pts, vec, xy, k=4):
    """IDW of k nearest VTP points. xy: (n,2)."""
    tree = cKDTree(pts[:, :2])
    dist, idx = tree.query(xy, k=k)
    dist = np.maximum(dist, 1e-3)
    w = 1.0 / dist ** 2
    w = w / w.sum(axis=-1, keepdims=True)
    if k == 1:
        return vec[idx]
    return np.einsum("nk,nkc->nc", w, vec[idx])


def station_dbs_error_snapshot(pts, vec, z, half_angle_deg, t, k=4):
    names = list(STATIONS)
    xy0 = np.array([STATIONS[s]["xy"] for s in names], float)
    r = beam_offset(z, half_angle_deg)
    beams = {
        "N": xy0 + np.array([0.0, r]),
        "S": xy0 + np.array([0.0, -r]),
        "E": xy0 + np.array([r, 0.0]),
        "W": xy0 + np.array([-r, 0.0]),
    }
    true = query_uv_at(pts, vec, xy0, k=k)
    vr = {}
    for name, xy in beams.items():
        uvw = query_uv_at(pts, vec, xy, k=k)
        az = {"N": 0.0, "E": 90.0, "S": 180.0, "W": 270.0}[name]
        vr[name] = radial_speed(uvw[:, 0], uvw[:, 1], uvw[:, 2], az, half_angle_deg)
    ud, vd = retrieve_uv(vr, half_angle_deg)
    rows = []
    for i, s in enumerate(names):
        true_s = np.hypot(true[i, 0], true[i, 1])
        dbs_s = np.hypot(ud[i], vd[i])
        rows.append(dict(
            t=t, station=s, z=z, theta=half_angle_deg,
            U_true=true_s, U_dbs=dbs_s,
            dU=dbs_s - true_s,
            rel=(dbs_s - true_s) / max(true_s, 0.3),
            u_true=true[i, 0], v_true=true[i, 1],
            u_dbs=ud[i], v_dbs=vd[i],
        ))
    return rows


def station_dbs_error(z: int, half_angle_deg: float, times=None, k=4):
    """3 stations × n snapshots: DBS vs true point value from the VTP cloud."""
    times = times if times is not None else [t for t in TIMES if t >= 1000]
    rows = []
    for t in times:
        pts, vec = read_vtp_points_U(vtp_path(z, t))
        rows.extend(station_dbs_error_snapshot(pts, vec, z, half_angle_deg, t, k=k))
    return rows


def variance_ratio(u_true_series, u_dbs_series):
    s0 = np.nanstd(u_true_series)
    s1 = np.nanstd(u_dbs_series)
    if not s0:
        return np.nan
    return float(s1 / s0)
