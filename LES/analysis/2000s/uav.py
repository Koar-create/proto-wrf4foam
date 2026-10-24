"""M5 UAV exposure: transient Λ, route envelopes, encounter gusts."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt

from paths import TIMES


def lambda_field(les_speed, rans_speed, umin=0.2):
    """Λ_LES = |U_LES| / |U_RANS|; drop cells with RANS < umin (manuscript §2.6)."""
    den = np.asarray(rans_speed, float)
    num = np.asarray(les_speed, float)
    den, num = np.broadcast_arrays(den, num)
    out = np.full(num.shape, np.nan, dtype=float)
    ok = np.isfinite(den) & np.isfinite(num) & (den >= umin)
    out[ok] = num[ok] / den[ok]
    return out


def snapshot_percentiles(lam, qs=(50.0, 95.0)):
    """lam: (nt, ny, nx) → dict of (ny,nx) maps."""
    return {q: np.nanpercentile(lam, q, axis=0) for q in qs}


def distance_to_solid(solid, dx):
    """Euclidean distance (m) from each fluid cell to the nearest building."""
    return distance_transform_edt(~np.asarray(solid, bool)) * float(dx)


def bin_by_distance(values, dist, solid, bins=(0, 25, 50, 100, 200, 400, 1e9)):
    rows = []
    v = np.asarray(values, float)
    d = np.asarray(dist, float)
    mfluid = ~np.asarray(solid, bool)
    for a, b in zip(bins[:-1], bins[1:]):
        m = mfluid & (d >= a) & (d < b) & np.isfinite(v)
        rows.append(dict(
            d0=a, d1=b, n=int(m.sum()),
            median=float(np.nanmedian(v[m])) if m.any() else np.nan,
            p95=float(np.nanpercentile(v[m], 95)) if m.any() else np.nan,
        ))
    return rows


def route_hovmoller(U_tnc):
    """|U_h| from route array (nt, n, 3)."""
    return np.hypot(U_tnc[..., 0], U_tnc[..., 1])


def route_envelope(speed_tn, t, t0=1000.0):
    sel = np.asarray(t) >= t0
    s = speed_tn[sel]
    return dict(
        p05=np.nanpercentile(s, 5, axis=0),
        p50=np.nanpercentile(s, 50, axis=0),
        p95=np.nanpercentile(s, 95, axis=0),
        mean=np.nanmean(s, axis=0),
        mx=np.nanmax(s, axis=0),
    )


def along_track_delta(speed_n, r_m, dx=20.0):
    """ΔU(r) = |U(s+r)| − |U(s)| on one snapshot. r in metres."""
    k = int(round(r_m / dx))
    if k <= 0 or k >= speed_n.size:
        return np.array([])
    return speed_n[k:] - speed_n[:-k]


def spatial_structure_function(speed_n, edges_m, dx=20.0, min_n=20):
    s = np.asarray(speed_n, float)
    n = s.size
    i, j = np.triu_indices(n, k=1)
    lag = np.abs(i - j) * dx
    d2 = (s[j] - s[i]) ** 2
    ok = np.isfinite(s[i]) & np.isfinite(s[j])
    idx = np.digitize(lag[ok], edges_m) - 1
    good = (idx >= 0) & (idx < len(edges_m) - 1)
    cnt = np.bincount(idx[good], minlength=len(edges_m) - 1)
    acc = np.bincount(idx[good], weights=d2[ok][good], minlength=len(edges_m) - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        D = acc / cnt
    D[cnt < min_n] = np.nan
    return D, cnt


def resolved_gust_factor(t, ws, T_avg=60.0, T_mean=600.0, t1=None):
    """G = max(T_avg moving mean) / T_mean mean. Resolved gust factor (lower bound)."""
    t = np.asarray(t, float)
    ws = np.asarray(ws, float)
    if t1 is None:
        t1 = t[-1]
    t0 = t1 - T_mean
    sel = (t >= t0) & (t <= t1)
    tt, ww = t[sel], ws[sel]
    if ww.size < 10:
        return np.nan
    dt = float(np.median(np.diff(tt)))
    k = max(int(round(T_avg / dt)), 1)
    ker = np.ones(k) / k
    mv = np.convolve(ww, ker, mode="valid")
    denom = float(np.mean(ww))
    if denom <= 1e-6:
        return np.nan
    return float(np.max(mv) / denom)


def exceedance_fraction(x, thresh):
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    if not ok.any():
        return np.nan
    return float(np.mean(np.abs(x[ok]) > thresh))
