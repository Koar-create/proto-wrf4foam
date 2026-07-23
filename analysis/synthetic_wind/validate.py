#!/usr/bin/env python3
"""Validation of synthetic wind fields against RANS statistics and LiDAR."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from analysis.synthetic_wind.config import (  # noqa: E402
    LIDAR_CSV,
    LIDAR_PROBES,
    OUTPUT_ROOT,
    SYNTH_FIELD_SUBDIR,
    SYNTH_PROBE_SUBDIR,
    VALIDATION_SUBDIR,
    dump_json,
)
from analysis.synthetic_wind.grid import load_enhanced_hdf5  # noqa: E402
from analysis.synthetic_wind.statistics import turbulence_intensity  # noqa: E402


def _load_synthetic_field(case_id: str, stride: int = 5) -> dict:
    path = OUTPUT_ROOT / SYNTH_FIELD_SUBDIR / f"synthetic_{case_id}.h5"
    with h5py.File(path, "r") as f:
        U = f["U"][::stride]
        return {
            "U": U,
            "time_s": f["time_s"][::stride],
            "coords_x": f["coords_x"][:],
            "coords_y": f["coords_y"][:],
            "coords_z": f["coords_z"][:],
        }


def validate_mean_and_variance(case_id: str) -> dict:
    rans = load_enhanced_hdf5(case_id)
    syn = _load_synthetic_field(case_id)
    U_syn = syn["U"]
    U_rans = rans["U"]
    U_mean_syn = np.mean(U_syn, axis=0)
    mean_bias = float(np.nanmean(np.abs(U_mean_syn - U_rans)))

    U_fluc = U_syn - U_mean_syn[np.newaxis, ...]
    var_syn = np.var(U_fluc, axis=0)
    var_target = np.stack([rans["R"][i, i] for i in range(3)], axis=0)
    mask = var_target > 1e-4
    if mask.any():
        var_ratio = float(np.nanmean(var_syn[mask] / var_target[mask]))
    else:
        var_ratio = float("nan")
    return {
        "mean_abs_bias_m_s": mean_bias,
        "variance_ratio_mean": var_ratio,
        "pass_mean": mean_bias < 0.15,
        "pass_variance": 0.5 < var_ratio < 2.0 if np.isfinite(var_ratio) else False,
    }


def validate_spectrum(case_id: str, probe_name: str = "GAW103_z100") -> dict:
    probe_path = OUTPUT_ROOT / SYNTH_PROBE_SUBDIR / f"probes_{case_id}.h5"
    with h5py.File(probe_path, "r") as f:
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in f["probe_names"][:]]
        idx = names.index(probe_name)
        u = f["U"][:, idx, 0]
        dt = float(f.attrs["meta"] and json.loads(f.attrs["meta"])["dt_s"])
    u = u - np.mean(u)
    freqs, psd = signal.welch(u, fs=1.0 / dt, nperseg=min(256, len(u) // 4))
    # inertial subrange slope between 0.1-1 Hz (if in range)
    mask = (freqs > 0.05) & (freqs < 2.0) & (psd > 0)
    slope = np.nan
    if mask.sum() > 3:
        slope = float(np.polyfit(np.log10(freqs[mask]), np.log10(psd[mask]), 1)[0])
    return {"spectral_slope": slope, "pass_spectrum": slope < -0.5 if np.isfinite(slope) else False}


def validate_ti_vs_lidar(case_id: str) -> dict:
    """Compare TI at LiDAR sites (100 m): RANS, synthetic probes, and LiDAR."""
    rans = load_enhanced_hdf5(case_id)
    TI_rans = turbulence_intensity(rans["R"], rans["U"])
    coords_x, coords_y, coords_z = rans["coords_x"], rans["coords_y"], rans["coords_z"]
    iz = int(np.argmin(np.abs(coords_z - 100.0)))

    probe_path = OUTPUT_ROOT / SYNTH_PROBE_SUBDIR / f"probes_{case_id}.h5"
    ti_synth = {}
    if probe_path.is_file():
        with h5py.File(probe_path, "r") as f:
            names = [n.decode() if isinstance(n, bytes) else str(n) for n in f["probe_names"][:]]
            for name in names:
                if name.endswith("_z100"):
                    site = name.split("_z")[0]
                    idx = names.index(name)
                    u = f["U"][:, idx, 0]
                    ti_synth[site] = float(np.std(u) / max(np.mean(u), 0.5))

    if not LIDAR_CSV.is_file():
        return {"lidar_available": False}

    dt_str = case_id[:8]
    hh, mm = case_id[9:11], case_id[11:13]
    ts = f"{dt_str[:4]}-{dt_str[4:6]}-{dt_str[6:8]} {hh}:{mm}:00"
    df = pd.read_csv(LIDAR_CSV)
    df = df[df["datetime"] == ts]
    rows = []
    for name, xy in LIDAR_PROBES.items():
        ix = int(np.argmin(np.abs(coords_x - xy["x"])))
        iy = int(np.argmin(np.abs(coords_y - xy["y"])))
        ti_model = float(TI_rans[ix, iy, iz])
        obs = df[(df["obtid"] == name) & (np.abs(df["Height"] - 100) < 10)]
        if len(obs) >= 3:
            ti_obs = float(obs["ws_obs"].std() / max(obs["ws_obs"].mean(), 0.5))
        elif len(obs) == 1:
            ti_obs = float("nan")
        else:
            ti_obs = float("nan")
        rows.append({
            "site": name,
            "ti_rans": ti_model,
            "ti_synthetic": ti_synth.get(name, float("nan")),
            "ti_lidar": ti_obs,
        })
    return {"lidar_available": True, "sites": rows}


def validate_coherence(case_id: str) -> dict:
    """Vertical coherence between 80 m and 150 m at GAW103."""
    probe_path = OUTPUT_ROOT / SYNTH_PROBE_SUBDIR / f"probes_{case_id}.h5"
    with h5py.File(probe_path, "r") as f:
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in f["probe_names"][:]]
        i80 = names.index("GAW103_z80")
        i150 = names.index("GAW103_z150")
        u80 = f["U"][:, i80, 0]
        u150 = f["U"][:, i150, 0]
        dt = float(json.loads(f.attrs["meta"])["dt_s"])
    freqs, cxy = signal.coherence(u80, u150, fs=1.0 / dt, nperseg=min(256, len(u80) // 4))
    coh_low = float(np.nanmean(cxy[freqs < 0.5]))
    return {"coherence_80_150_below_0p5Hz": coh_low, "pass_coherence": coh_low > 0.3}


def plot_validation(case_id: str, out_dir: Path) -> None:
    probe_path = OUTPUT_ROOT / SYNTH_PROBE_SUBDIR / f"probes_{case_id}.h5"
    with h5py.File(probe_path, "r") as f:
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in f["probe_names"][:]]
        idx = names.index("GAW103_z100")
        u = f["U"][:, idx, 0]
        dt = float(json.loads(f.attrs["meta"])["dt_s"])
    freqs, psd = signal.welch(u - np.mean(u), fs=1.0 / dt, nperseg=min(256, len(u) // 4))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(np.arange(len(u)) * dt, u)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("u (m/s)")
    axes[0].set_title(f"{case_id} GAW103 z=100m")
    axes[1].loglog(freqs[1:], psd[1:])
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("PSD")
    axes[1].set_title("Power spectrum")
    fig.tight_layout()
    fig.savefig(out_dir / f"validate_{case_id}.png", dpi=150)
    plt.close(fig)


def validate_integral_time(case_id: str, probe_name: str = "GAW103_z100") -> dict:
    """Compare empirical integral time scale from ACF with RANS T = k/epsilon."""
    rans = load_enhanced_hdf5(case_id)
    probe_path = OUTPUT_ROOT / SYNTH_PROBE_SUBDIR / f"probes_{case_id}.h5"
    with h5py.File(probe_path, "r") as f:
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in f["probe_names"][:]]
        idx = names.index(probe_name)
        u = f["U"][:, idx, 0]
        dt = float(json.loads(f.attrs["meta"])["dt_s"])
    u = u - np.mean(u)
    acf = np.correlate(u, u, mode="full")[len(u) - 1:]
    acf /= acf[0] if acf[0] > 0 else 1.0
    below = np.where(acf < 1.0 / np.e)[0]
    t_emp = float(below[0] * dt) if len(below) else float("nan")
    ix, iy, iz = [
        int(np.argmin(np.abs(rans["coords_x"] - LIDAR_PROBES["GAW103"]["x"]))),
        int(np.argmin(np.abs(rans["coords_y"] - LIDAR_PROBES["GAW103"]["y"]))),
        int(np.argmin(np.abs(rans["coords_z"] - 100.0))),
    ]
    t_rans = float(rans["T"][ix, iy, iz])
    ratio = t_emp / t_rans if t_rans > 0 and np.isfinite(t_emp) else float("nan")
    return {
        "T_empirical_s": t_emp,
        "T_rans_s": t_rans,
        "T_ratio": ratio,
        "pass_integral_time": 0.05 < ratio < 5.0 if np.isfinite(ratio) else False,
    }


def validate_case(case_id: str) -> dict:
    out_dir = OUTPUT_ROOT / VALIDATION_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"case_id": case_id}
    report.update(validate_mean_and_variance(case_id))
    report.update(validate_spectrum(case_id))
    report.update(validate_ti_vs_lidar(case_id))
    report.update(validate_coherence(case_id))
    report.update(validate_integral_time(case_id))
    try:
        plot_validation(case_id, out_dir)
    except Exception as exc:
        report["plot_error"] = str(exc)
    dump_json(out_dir / f"report_{case_id}.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    args = parser.parse_args()
    if args.case == "representative":
        rep = json.loads(
            (OUTPUT_ROOT / "statistics" / "representative_cases.json").read_text(encoding="utf-8")
        )
        cases = rep["representative_cases"]
    else:
        cases = [args.case]
    for cid in cases:
        field_h5 = OUTPUT_ROOT / SYNTH_FIELD_SUBDIR / f"synthetic_{cid}.h5"
        if not field_h5.is_file():
            print(f"  [{cid}] SKIP (no synthetic field yet)", flush=True)
            continue
        try:
            r = validate_case(cid)
            print(f"  [{cid}] mean_bias={r.get('mean_abs_bias_m_s', '?'):.4f} "
                  f"var_ratio={r.get('variance_ratio_mean', '?'):.3f}", flush=True)
        except Exception as exc:
            print(f"  [{cid}] FAILED: {exc}", flush=True)


if __name__ == "__main__":
    main()
