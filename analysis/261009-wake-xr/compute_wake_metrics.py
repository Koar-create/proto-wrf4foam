"""X_R, recirculation-cavity size, and main-wake decay from the probed boxes.

Run after pv_extract_wake.py, from the repo root:

  python3 analysis/261009-wake-xr/compute_wake_metrics.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage, optimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wake_geom import fackrell_lr

REPO = Path("/hpc2hdd/home/zyang248/WRF-OpenFOAM-Coupling")
RAW = REPO / "data" / "261009" / "raw" / "wake_boxes"
OUT = REPO / "data" / "261009" / "processed" / "wake_metrics.csv"

# A NaN gap longer than this, after the flow has already reversed, means the
# center band has run into a downstream building rather than reattached.
GAP_CELLS = 3


def streamwise(U: np.ndarray, ex: np.ndarray) -> np.ndarray:
    return U[..., 0] * ex[0] + U[..., 1] * ex[1]


def band_mean(U: np.ndarray, valid: np.ndarray, ex: np.ndarray) -> np.ndarray:
    """Mean streamwise velocity over y. Shape (nz, nx). NaN if the whole band is invalid."""
    s = streamwise(U, ex)
    m = valid > 0
    s = np.where(m, s, np.nan)
    count = m.sum(axis=1)
    prof = np.full((s.shape[0], s.shape[2]), np.nan, dtype=np.float64)
    ok = count > 0
    if np.any(ok):
        # Mean over the cross-wind samples that landed in the fluid.
        summed = np.nansum(np.where(m, s, 0.0), axis=1)
        prof[ok] = summed[ok] / count[ok]
    return prof


def classify_xr(x: np.ndarray, u: np.ndarray, x_lee: float, height: float) -> tuple[str, float]:
    """Return (status, X_R measured from the lee face).

    status is one of reattached, no_recirculation, truncated, no_reattachment_in_window.
    X_R is finite only for a clean negative-to-positive crossing.
    """
    downstream = np.flatnonzero(x >= x_lee - 1e-6)
    if len(downstream) == 0:
        return "no_window", np.nan
    seen_neg = False
    i_neg = -1
    near = x_lee + 0.35 * height
    for i in downstream:
        ui = u[i]
        if not np.isfinite(ui):
            if not seen_neg:
                continue
            # Look past this gap. A short hole is a missing sample; a long one is a building.
            j = i
            while j < len(u) and not np.isfinite(u[j]):
                j += 1
            gap = j - i
            if j >= len(u):
                return "truncated", np.nan
            if gap >= GAP_CELLS:
                return "truncated", np.nan
            continue
        if ui < 0.0:
            seen_neg = True
            i_neg = i
            continue
        # ui >= 0
        if not seen_neg:
            if x[i] > near:
                return "no_recirculation", np.nan
            continue
        if i_neg < 0 or (i - i_neg) > GAP_CELLS:
            return "truncated", np.nan
        u0, u1 = float(u[i_neg]), float(ui)
        x0, x1 = float(x[i_neg]), float(x[i])
        if u1 == u0:
            xc = x1
        else:
            xc = x0 - u0 * (x1 - x0) / (u1 - u0)
        return "reattached", float(xc - x_lee)
    if seen_neg:
        return "no_reattachment_in_window", np.nan
    return "no_recirculation", np.nan


def ambient_profile(U: np.ndarray, valid: np.ndarray, z: np.ndarray, ex: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    s = np.where(valid > 0, streamwise(U, ex), np.nan)
    with np.errstate(all="ignore"):
        prof = np.nanmean(s, axis=(1, 2))
    return z, prof


def interp_amb(z_amb: np.ndarray, u_amb: np.ndarray, z: float) -> float:
    ok = np.isfinite(u_amb)
    if ok.sum() < 2:
        return np.nan
    return float(np.interp(z, z_amb[ok], u_amb[ok]))


def fit_decay(xi: np.ndarray, deficit: np.ndarray) -> tuple[float, float, float, float, int]:
    """Fit deficit = a + b * xi**(-p). Returns a, b, p, r2, n."""
    m = np.isfinite(xi) & np.isfinite(deficit) & (xi > 0.15) & (deficit > 0.02)
    xi, deficit = xi[m], deficit[m]
    if len(xi) < 8:
        return (np.nan, np.nan, np.nan, np.nan, int(len(xi)))

    def model(x, a, b, p):
        return a + b * np.power(x, -p)

    b0 = float(np.median(deficit) * (np.median(xi) ** 1.0))
    try:
        popt, _ = optimize.curve_fit(
            model,
            xi,
            deficit,
            p0=(0.0, max(b0, 0.05), 1.0),
            bounds=([-2.0, 0.0, 0.05], [2.0, 40.0, 4.0]),
            maxfev=4000,
        )
    except Exception:
        return (np.nan, np.nan, np.nan, np.nan, int(len(xi)))
    pred = model(xi, *popt)
    ss_res = float(np.sum((deficit - pred) ** 2))
    ss_tot = float(np.sum((deficit - deficit.mean()) ** 2))
    r2 = np.nan if ss_tot <= 0 else 1.0 - ss_res / ss_tot
    return (float(popt[0]), float(popt[1]), float(popt[2]), float(r2), int(len(xi)))


def half_width(y: np.ndarray, deficit: np.ndarray) -> float:
    """Full width where the deficit falls to half its centerline value. NaN if it does not."""
    m = np.isfinite(deficit)
    if m.sum() < 5:
        return np.nan
    y, deficit = y[m], deficit[m]
    i0 = int(np.argmin(np.abs(y)))
    d0 = float(deficit[i0])
    if d0 <= 0.05:
        return np.nan
    half = 0.5 * d0

    def side(sl):
        yy, dd = y[sl], deficit[sl]
        # moving away from the center, first drop through `half`
        if len(yy) < 2:
            return np.nan
        for i in range(len(yy) - 1):
            if dd[i] >= half and dd[i + 1] < half:
                if dd[i] == dd[i + 1]:
                    return float(yy[i + 1])
                w = (dd[i] - half) / (dd[i] - dd[i + 1])
                return float(yy[i] + w * (yy[i + 1] - yy[i]))
        return np.nan

    left = side(slice(i0, None, -1))
    right = side(slice(i0, None))
    # side() walks toward decreasing index on the left, so the returned y is the location.
    if not np.isfinite(left) or not np.isfinite(right):
        return np.nan
    return float(abs(right - left))


def cavity_metrics(blob, ex: np.ndarray, x_lee: float) -> dict:
    x = blob["x_cav"]
    y = blob["y_cav"]
    z = blob["z_cav"]
    s = streamwise(blob["U_cav"], ex)
    fluid = blob["valid_cav"] > 0
    neg = fluid & (s < 0.0)
    if not np.any(neg):
        return {"cav_length_m": 0.0, "cav_height_m": 0.0, "cav_width_m": 0.0, "cav_volume_m3": 0.0}
    struct = ndimage.generate_binary_structure(3, 1)
    labels, _ = ndimage.label(neg, structure=struct)
    dx = float(np.median(np.diff(x))) if len(x) > 1 else 4.0
    # Seed on the first fluid column at or downstream of the lee face, within 0.5H.
    height = float(blob["H"])
    seed_x = np.flatnonzero((x >= x_lee - dx) & (x <= x_lee + 0.5 * height))
    seed_labels = set()
    for ix in seed_x:
        if np.any(fluid[:, :, ix]):
            seed_labels.update(int(v) for v in np.unique(labels[:, :, ix]) if v != 0)
            if seed_labels:
                break
    if not seed_labels:
        return {"cav_length_m": np.nan, "cav_height_m": np.nan, "cav_width_m": np.nan, "cav_volume_m3": np.nan}
    mask = np.isin(labels, list(seed_labels))
    xs = x[np.any(mask, axis=(0, 1))]
    ys = y[np.any(mask, axis=(0, 2))]
    zs = z[np.any(mask, axis=(1, 2))]
    dy = float(np.median(np.diff(y))) if len(y) > 1 else dx
    dz = float(np.median(np.diff(z))) if len(z) > 1 else dx
    return {
        "cav_length_m": float(xs.max() - x_lee),
        "cav_height_m": float(zs.max()),
        "cav_width_m": float(ys.max() - ys.min()),
        "cav_volume_m3": float(mask.sum() * dx * dy * dz),
    }


def z_tags(z: np.ndarray, height: float) -> list[tuple[str, int]]:
    targets = {
        "z3m": 3.0,
        "z0.10H": 0.10 * height,
        "z0.25H": 0.25 * height,
        "z0.50H": 0.50 * height,
        "z1.00H": 1.00 * height,
    }
    out = []
    used = set()
    for tag, zt in targets.items():
        iz = int(np.argmin(np.abs(z - zt)))
        if iz in used:
            continue
        if abs(float(z[iz]) - zt) > max(2.0, 0.05 * height):
            continue
        used.add(iz)
        out.append((tag, iz))
    return out


def one_file(path: Path, case: str) -> list[dict]:
    blob = np.load(path)
    ex = blob["ex"]
    height = float(blob["H"])
    x_lee = float(blob["x_lee"])
    width = float(blob["W"])
    length = float(blob["L"])
    lr = fackrell_lr(width, height, length)
    prof = band_mean(blob["U_center"], blob["valid_center"], ex)
    z_amb, u_amb = ambient_profile(blob["U_amb"], blob["valid_amb"], blob["z_amb"], ex)
    cav = cavity_metrics(blob, ex, x_lee)
    x = blob["x_center"]
    rows = []
    for tag, iz in z_tags(blob["z_center"], height):
        z = float(blob["z_center"][iz])
        status, xr = classify_xr(x, prof[iz], x_lee, height)
        u_a = interp_amb(z_amb, u_amb, z)
        if status == "reattached" and np.isfinite(xr):
            x_start = x_lee + xr
        else:
            x_start = x_lee + 0.5 * height
        xi = (x - x_lee) / height
        use = (x >= x_start) & (x <= x_lee + 10.0 * height)
        deficit = u_a - prof[iz]
        a, b, p, r2, nfit = fit_decay(xi[use], deficit[use])
        hw = np.nan
        # Lateral half-width at this height, on the station just downstream of X_R (or at 2H).
        z_lat = blob["z_lat"]
        il = int(np.argmin(np.abs(z_lat - z)))
        if abs(float(z_lat[il]) - z) <= max(2.0, 0.05 * height) and len(blob["x_lat"]):
            target_x = x_lee + (xr if np.isfinite(xr) else 2.0 * height)
            ix = int(np.argmin(np.abs(blob["x_lat"] - target_x)))
            s_lat = streamwise(blob["U_lat"][il, :, ix, :], ex)
            s_lat = np.where(blob["valid_lat"][il, :, ix] > 0, s_lat, np.nan)
            hw = half_width(blob["y_lat"], u_a - s_lat)
        rows.append(
            {
                "case_id": case,
                "bid": int(blob["bid"]),
                "H_m": height,
                "W_m": width,
                "L_m": length,
                "theta_deg": float(np.degrees(float(blob["theta"]))),
                "U_approach_ms": float(blob["U_approach"]),
                "x_lee_m": x_lee,
                "z_tag": tag,
                "z_m": z,
                "z_over_H": z / height,
                "status": status,
                "XR_m": xr,
                "XR_over_H": xr / height if np.isfinite(xr) else np.nan,
                "LR_fackrell_m": lr,
                "LR_fackrell_over_H": lr / height,
                "r_fackrell": xr / lr if np.isfinite(xr) and lr > 0 else np.nan,
                "p_fit": p,
                "p_a": a,
                "p_b": b,
                "p_r2": r2,
                "p_n": nfit,
                "halfwidth_m": hw,
                **cav,
            }
        )
    return rows


def main() -> None:
    rows = []
    for case_dir in sorted(p for p in RAW.iterdir() if p.is_dir() and not p.name.startswith("_")):
        for path in sorted(case_dir.glob("*.npz")):
            rows.extend(one_file(path, case_dir.name))
    if not rows:
        raise SystemExit(f"no npz under {RAW}")
    df = pd.DataFrame(rows)
    tall = pd.read_csv(REPO / "data" / "261009" / "processed" / "tall_buildings.csv")
    df = df.merge(tall[["bid", "I", "H_surround", "n_neigh_2H", "cx", "cy"]], on="bid", how="left")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    ground = df[df["z_tag"] == "z3m"]
    print(f"wrote {OUT}  rows {len(df)}  buildings {df.groupby(['case_id','bid']).ngroups}")
    print(ground.groupby(["case_id", "status"]).size().to_string())
    ok = ground[ground["status"] == "reattached"]
    print("reattached r median", ok["r_fackrell"].median(), "XR/H median", ok["XR_over_H"].median())


if __name__ == "__main__":
    main()
