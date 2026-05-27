#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Perturb WRF-to-CFD inlet fields on the cartesian NetCDF products.

This script is designed to sit between:
  util/convert_wrf_coord_and_do_3D_interp.py  ->  *_cartesian.nc
and:
  util/construct_OF_boundary_arrays.py        ->  boundaryData/

It writes a new NetCDF with physically interpretable perturbations applied to:
  - (A) near-surface wind re-inflation via log-law ratio (U,V)
  - (B) TKE floor from minimum turbulence intensity (TKE_PBL)
  - (C) epsilon override using Blackadar-type mixing length (eps_override)
  - (D) optional LLJ nose boost (U,V)

Downstream compatibility:
  - The perturbed TKE is written back to variable name "TKE_PBL" (and the original
    is preserved as "TKE_PBL_raw" when keep_raw is enabled).
  - If eps_override is present, construct_OF_boundary_arrays.py can be patched to
    consume it; otherwise it will compute epsilon internally as before.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Dict, Tuple

import numpy as np
import xarray as xr


KAPPA = 0.41
CMU = 0.09


@dataclass(frozen=True)
class PerturbConfig:
    # A: log-law re-inflation
    enable_A: bool = True
    z0_wrf: float = 0.8
    z0_eff: float = 0.2
    reinflate_zmax: float = 500.0
    reinflate_gain_cap: float = 1.3

    # B: TKE floor
    enable_B: bool = True
    Imin_unstable: float = 0.10
    Imin_neutral: float = 0.08
    Imin_stable: float = 0.05

    # C: epsilon override (Blackadar mixing length)
    enable_C: bool = True
    write_eps_override: bool = True
    lambda_unstable: float = 80.0
    lambda_neutral: float = 50.0
    lambda_stable: float = 25.0
    L_min: float = 10.0
    L_max: float = 100.0

    # D: LLJ boost (optional)
    enable_D: bool = False
    llj_window_zmin: float = 200.0
    llj_window_zmax: float = 600.0
    llj_nose_pad: float = 100.0
    llj_gain: float = 1.07
    llj_drop_frac: float = 0.25

    # stability diagnosis
    stability: str = "auto"  # auto|unstable|neutral|stable
    stability_zmin: float = 50.0
    stability_zmax: float = 200.0
    stability_thresh: float = 0.005  # K/m; +/- threshold for neutral

    # bookkeeping
    keep_raw: bool = True


def _infer_out_path(in_path: str, out_path: str | None) -> str:
    if out_path:
        return os.path.abspath(os.path.normpath(out_path))
    base, ext = os.path.splitext(in_path)
    if ext.lower() not in [".nc", ".netcdf"]:
        # keep original suffix, just append
        return base + "_perturbed"
    return base + "_perturbed" + ext


def _require_vars(ds: xr.Dataset, vars_required: Tuple[str, ...]) -> None:
    missing = [v for v in vars_required if v not in ds]
    if missing:
        raise KeyError(f"Dataset missing required variables: {missing}")
    if "z" not in ds.coords and "z" not in ds:
        raise KeyError("Dataset missing required coordinate/variable: 'z'")


def _stability_from_theta(ds: xr.Dataset, zmin: float, zmax: float, thresh: float) -> str:
    """
    Diagnose stability using mean dTheta/dz in [zmin, zmax].
    - unstable: mean grad < -thresh
    - stable:   mean grad > +thresh
    - neutral:  otherwise
    """
    if "Theta" not in ds:
        return "neutral"

    z = ds["z"].values.astype(float)
    mask = (z >= zmin) & (z <= zmax)
    if mask.sum() < 3:
        return "neutral"

    theta = ds["Theta"].values
    # theta dims are (z, y, x) in this pipeline; reduce to 1D profile
    theta_prof = np.nanmean(theta, axis=(1, 2))
    z_sel = z[mask]
    th_sel = theta_prof[mask]

    # Robust linear fit for gradient; ignore NaNs
    valid = np.isfinite(z_sel) & np.isfinite(th_sel)
    if valid.sum() < 3:
        return "neutral"

    coef = np.polyfit(z_sel[valid], th_sel[valid], 1)
    dthdz = float(coef[0])
    if dthdz < -thresh:
        return "unstable"
    if dthdz > thresh:
        return "stable"
    return "neutral"


def _select_by_stability(stab: str, unstable_v: float, neutral_v: float, stable_v: float) -> float:
    if stab == "unstable":
        return float(unstable_v)
    if stab == "stable":
        return float(stable_v)
    return float(neutral_v)


def reinflate_wind(ds: xr.Dataset, cfg: PerturbConfig) -> xr.Dataset:
    z = ds["z"].values.astype(float)
    z3 = np.broadcast_to(z[:, None, None], ds["U"].shape)

    z0_wrf = max(float(cfg.z0_wrf), 1e-6)
    z0_eff = max(float(cfg.z0_eff), 1e-6)
    zmax = float(cfg.reinflate_zmax)
    gain_cap = float(cfg.reinflate_gain_cap)

    # ratio of log-law multipliers; clamp for safety
    num = np.log((z3 + z0_eff) / z0_eff)
    den = np.log((z3 + z0_wrf) / z0_wrf)
    with np.errstate(divide="ignore", invalid="ignore"):
        G = num / den
    G = np.where(np.isfinite(G), G, 1.0)
    G = np.clip(G, 0.0, gain_cap)
    G = np.where(z3 < zmax, G, 1.0)

    out = ds.copy()
    out["U"] = (ds["U"].dims, ds["U"].values * G)
    out["V"] = (ds["V"].dims, ds["V"].values * G)
    # WS is optional; if present update for convenience
    if "WS" in out:
        out["WS"] = (ds["WS"].dims, np.sqrt(out["U"].values ** 2 + out["V"].values ** 2))
    return out


def floor_TKE(ds: xr.Dataset, I_min: float) -> xr.Dataset:
    out = ds.copy()
    Uh = np.sqrt(out["U"].values ** 2 + out["V"].values ** 2)
    k_floor = 1.5 * (float(I_min) * Uh) ** 2
    k_raw = out["TKE_PBL"].values
    k_new = np.maximum(np.maximum(k_raw, 1e-6), k_floor)
    out["TKE_PBL"] = (out["TKE_PBL"].dims, k_new)
    return out


def _blackadar_mixing_length(z3: np.ndarray, lam: float, L_min: float, L_max: float) -> np.ndarray:
    lam = max(float(lam), 1e-6)
    L = (KAPPA * z3) / (1.0 + (KAPPA * z3 / lam))
    return np.clip(L, float(L_min), float(L_max))


def recompute_epsilon_override(ds: xr.Dataset, lam: float, L_min: float, L_max: float) -> xr.Dataset:
    z = ds["z"].values.astype(float)
    z3 = np.broadcast_to(z[:, None, None], ds["TKE_PBL"].shape)

    k = np.maximum(ds["TKE_PBL"].values, 1e-6)
    L = _blackadar_mixing_length(z3, lam=lam, L_min=L_min, L_max=L_max)
    eps = (CMU ** 0.75) * (k ** 1.5) / L
    eps = np.maximum(eps, 1e-8)

    out = ds.copy()
    out["eps_override"] = (ds["TKE_PBL"].dims, eps)
    out["mixingLength_override"] = (ds["TKE_PBL"].dims, L)
    return out


def _detect_llj_from_profile(z: np.ndarray, ws_prof: np.ndarray, zmin: float, zmax: float, drop_frac: float) -> Tuple[bool, float]:
    """Return (is_llj, z_nose)."""
    mask = (z >= zmin) & (z <= zmax) & np.isfinite(ws_prof)
    if mask.sum() < 5:
        return False, float("nan")

    z_w = z[mask]
    ws_w = ws_prof[mask]
    idx = int(np.argmax(ws_w))
    ws_peak = float(ws_w[idx])
    z_peak = float(z_w[idx])

    # Need a meaningful drop above the nose inside the window.
    if idx >= len(ws_w) - 2:
        return False, z_peak
    ws_above_min = float(np.nanmin(ws_w[idx + 1 :]))
    if not np.isfinite(ws_above_min) or ws_peak <= 0:
        return False, z_peak
    drop = (ws_peak - ws_above_min) / ws_peak
    return drop >= float(drop_frac), z_peak


def llj_boost(ds: xr.Dataset, cfg: PerturbConfig) -> xr.Dataset:
    if "WS" in ds:
        ws = ds["WS"].values
    else:
        ws = np.sqrt(ds["U"].values ** 2 + ds["V"].values ** 2)

    z = ds["z"].values.astype(float)
    ws_prof = np.nanmean(ws, axis=(1, 2))
    is_llj, z_nose = _detect_llj_from_profile(
        z=z,
        ws_prof=ws_prof,
        zmin=float(cfg.llj_window_zmin),
        zmax=float(cfg.llj_window_zmax),
        drop_frac=float(cfg.llj_drop_frac),
    )
    if (not is_llj) or (not np.isfinite(z_nose)):
        return ds

    z3 = np.broadcast_to(z[:, None, None], ds["U"].shape)
    z0 = float(z_nose)
    pad = float(cfg.llj_nose_pad)
    gain = float(cfg.llj_gain)

    G = np.where((z3 >= (z0 - pad)) & (z3 <= (z0 + pad)), gain, 1.0)
    out = ds.copy()
    out["U"] = (ds["U"].dims, ds["U"].values * G)
    out["V"] = (ds["V"].dims, ds["V"].values * G)
    if "WS" in out:
        out["WS"] = (out["WS"].dims, np.sqrt(out["U"].values ** 2 + out["V"].values ** 2))
    out.attrs["llj_detected"] = "true"
    out.attrs["llj_nose_z_m"] = f"{z_nose:.3f}"
    return out


def _attach_raw(ds: xr.Dataset, keep_raw: bool) -> xr.Dataset:
    if not keep_raw:
        return ds
    out = ds.copy()
    for name in ["U", "V", "WS", "TKE_PBL"]:
        if name in out and f"{name}_raw" not in out:
            out[f"{name}_raw"] = out[name].copy(deep=True)
    return out


def _finalize_attrs(ds: xr.Dataset, cfg: PerturbConfig, stability_used: str, in_path: str) -> xr.Dataset:
    out = ds.copy()
    meta: Dict[str, object] = {
        "generated_by": "util/perturb_OF_inlet_data.py",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_file": os.path.abspath(in_path),
        "stability_used": stability_used,
        "config": asdict(cfg),
    }
    out.attrs["inlet_perturbation_meta"] = json.dumps(meta, ensure_ascii=False)
    return out


def main() -> None:
    p = argparse.ArgumentParser(
        description="Perturb WRF cartesian inlet NetCDF for OpenFOAM boundaryData generation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input_nc", help="Path to *_cartesian.nc (output of convert_wrf_coord_and_do_3D_interp.py)")
    p.add_argument("--output", default=None, help="Output NetCDF path. Default: append _perturbed before .nc")

    # A
    p.add_argument("--z0-wrf", type=float, default=0.8)
    p.add_argument("--z0-eff", type=float, default=0.2)
    p.add_argument("--reinflate-zmax", type=float, default=500.0)
    p.add_argument("--reinflate-gain-cap", type=float, default=1.3)

    # B
    p.add_argument("--Imin-unstable", type=float, default=0.10)
    p.add_argument("--Imin-neutral", type=float, default=0.08)
    p.add_argument("--Imin-stable", type=float, default=0.05)

    # C
    p.add_argument("--lambda-unstable", type=float, default=80.0)
    p.add_argument("--lambda-neutral", type=float, default=50.0)
    p.add_argument("--lambda-stable", type=float, default=25.0)
    p.add_argument("--write-eps-override", action="store_true", default=True)
    p.add_argument("--no-write-eps-override", action="store_false", dest="write_eps_override")
    p.add_argument("--L-min", type=float, default=10.0)
    p.add_argument("--L-max", type=float, default=100.0)

    # D
    p.add_argument("--llj-boost", action="store_true", default=False)
    p.add_argument("--llj-window", nargs=2, type=float, default=[200.0, 600.0], metavar=("ZMIN", "ZMAX"))
    p.add_argument("--llj-nose-pad", type=float, default=100.0)
    p.add_argument("--llj-gain", type=float, default=1.07)
    p.add_argument("--llj-drop-frac", type=float, default=0.25)

    # stability
    p.add_argument("--stability", choices=["auto", "unstable", "neutral", "stable"], default="auto")
    p.add_argument("--stability-z", nargs=2, type=float, default=[50.0, 200.0], metavar=("ZMIN", "ZMAX"))
    p.add_argument("--stability-thresh", type=float, default=0.005)

    # toggles / bookkeeping
    p.add_argument("--disable-A", action="store_true", default=False)
    p.add_argument("--disable-B", action="store_true", default=False)
    p.add_argument("--disable-C", action="store_true", default=False)
    p.add_argument("--keep-raw", action="store_true", default=True)
    p.add_argument("--no-keep-raw", action="store_false", dest="keep_raw")

    args = p.parse_args()

    in_path = os.path.abspath(os.path.normpath(args.input_nc))
    if not os.path.isfile(in_path):
        raise FileNotFoundError(f"Input file not found: {in_path}")
    out_path = _infer_out_path(in_path, args.output)

    cfg = PerturbConfig(
        enable_A=(not args.disable_A),
        z0_wrf=float(args.z0_wrf),
        z0_eff=float(args.z0_eff),
        reinflate_zmax=float(args.reinflate_zmax),
        reinflate_gain_cap=float(args.reinflate_gain_cap),
        enable_B=(not args.disable_B),
        Imin_unstable=float(args.Imin_unstable),
        Imin_neutral=float(args.Imin_neutral),
        Imin_stable=float(args.Imin_stable),
        enable_C=(not args.disable_C),
        write_eps_override=bool(args.write_eps_override),
        lambda_unstable=float(args.lambda_unstable),
        lambda_neutral=float(args.lambda_neutral),
        lambda_stable=float(args.lambda_stable),
        L_min=float(args.L_min),
        L_max=float(args.L_max),
        enable_D=bool(args.llj_boost),
        llj_window_zmin=float(args.llj_window[0]),
        llj_window_zmax=float(args.llj_window[1]),
        llj_nose_pad=float(args.llj_nose_pad),
        llj_gain=float(args.llj_gain),
        llj_drop_frac=float(args.llj_drop_frac),
        stability=str(args.stability),
        stability_zmin=float(args.stability_z[0]),
        stability_zmax=float(args.stability_z[1]),
        stability_thresh=float(args.stability_thresh),
        keep_raw=bool(args.keep_raw),
    )

    ds0 = xr.open_dataset(in_path)
    _require_vars(ds0, ("U", "V", "W", "TKE_PBL"))
    if "WS" not in ds0:
        ds0 = ds0.assign(WS=np.sqrt(ds0["U"] ** 2 + ds0["V"] ** 2))

    ds = _attach_raw(ds0, keep_raw=cfg.keep_raw)

    if cfg.stability == "auto":
        stability_used = _stability_from_theta(
            ds,
            zmin=cfg.stability_zmin,
            zmax=cfg.stability_zmax,
            thresh=cfg.stability_thresh,
        )
    else:
        stability_used = cfg.stability

    # A
    if cfg.enable_A:
        ds = reinflate_wind(ds, cfg)

    # B (depends on (possibly perturbed) wind magnitude)
    if cfg.enable_B:
        Imin = _select_by_stability(stability_used, cfg.Imin_unstable, cfg.Imin_neutral, cfg.Imin_stable)
        ds = floor_TKE(ds, I_min=Imin)

    # C: write epsilon override if requested
    if cfg.enable_C and cfg.write_eps_override:
        lam = _select_by_stability(stability_used, cfg.lambda_unstable, cfg.lambda_neutral, cfg.lambda_stable)
        ds = recompute_epsilon_override(ds, lam=lam, L_min=cfg.L_min, L_max=cfg.L_max)

    # D: optional LLJ boost last (shape tweak)
    if cfg.enable_D:
        ds = llj_boost(ds, cfg)

    ds = _finalize_attrs(ds, cfg, stability_used=stability_used, in_path=in_path)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    ds.to_netcdf(out_path)
    print(f"[OK] wrote: {out_path}")
    print(f"[INFO] stability_used={stability_used}")
    print(f"[INFO] keep_raw={cfg.keep_raw} enable_A={cfg.enable_A} enable_B={cfg.enable_B} enable_C={cfg.enable_C} enable_D={cfg.enable_D}")


if __name__ == "__main__":
    main()

