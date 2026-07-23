#!/usr/bin/env python3
"""Generate simple presentation figures for the synthetic-wind pipeline."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch
from scipy import signal

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from analysis.synthetic_wind.config import (  # noqa: E402
    LIDAR_PROBES,
    OUTPUT_ROOT,
    REPO_ROOT,
    SYNTH_FIELD_SUBDIR,
    SYNTH_PROBE_SUBDIR,
    VALIDATION_SUBDIR,
)
from analysis.synthetic_wind.grid import load_enhanced_hdf5  # noqa: E402
from analysis.synthetic_wind.mean_field import case_id_to_datetime  # noqa: E402

PRESENTATION_DIR = REPO_ROOT / "results" / "synthetic_wind" / "presentation"
DEFAULT_CASE_NIGHT = "20250903_1400_two_boundaries_as_outlet"
DEFAULT_CASE_DAY = "20250901_1200_two_boundaries_as_outlet"
CROP_HALF_M = 1500.0
DPI = 150


def _synthetic_field_path(case_id: str) -> Path:
    return OUTPUT_ROOT / SYNTH_FIELD_SUBDIR / f"synthetic_{case_id}.h5"


def _resolve_field_case(case_id: str) -> tuple[str, bool]:
    """Return a case_id with a synthetic field HDF5, optionally falling back."""
    if _synthetic_field_path(case_id).is_file():
        return case_id, False
    prefix = case_id[:13]  # YYYYMMDD_HHMM
    for p in sorted(_synthetic_field_path("x").parent.glob(f"synthetic_{prefix[:8]}*_two_boundaries_as_outlet.h5")):
        cid = p.name.removeprefix("synthetic_").removesuffix(".h5")
        return cid, True
    for p in sorted(_synthetic_field_path("x").parent.glob("synthetic_*_two_boundaries_as_outlet.h5")):
        cid = p.name.removeprefix("synthetic_").removesuffix(".h5")
        return cid, True
    raise FileNotFoundError(f"No synthetic field HDF5 found for {case_id}")


def _ensure_out() -> Path:
    PRESENTATION_DIR.mkdir(parents=True, exist_ok=True)
    return PRESENTATION_DIR


def _case_title(case_id: str) -> str:
    dt = case_id_to_datetime(case_id)
    return dt.strftime("%Y-%m-%d %H:%M UTC")


def _crop_slice(
    field_2d: np.ndarray,
    coords_x: np.ndarray,
    coords_y: np.ndarray,
    half_m: float = CROP_HALF_M,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mask_x = (coords_x >= -half_m) & (coords_x <= half_m)
    mask_y = (coords_y >= -half_m) & (coords_y <= half_m)
    return (
        field_2d[np.ix_(mask_x, mask_y)],
        coords_x[mask_x],
        coords_y[mask_y],
    )


def _overlay_probes(ax: plt.Axes) -> None:
    for name, xy in LIDAR_PROBES.items():
        if abs(xy["x"]) <= CROP_HALF_M and abs(xy["y"]) <= CROP_HALF_M:
            ax.plot(xy["x"], xy["y"], "wx", ms=7, mew=1.5)
            ax.text(xy["x"], xy["y"] + 80, name, color="white", ha="center", fontsize=7)


def plot_gust_animation(
    case_id: str,
    out_dir: Path,
    stride: int = 1,
    fps: int = 8,
    max_duration_s: float = 60.0,
) -> Path:
    """Animated horizontal slice of synthetic |U| at z=100 m (right panel of mean_vs_gust)."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    field_case, fallback = _resolve_field_case(case_id)
    rans = load_enhanced_hdf5(case_id)
    coords_x, coords_y, coords_z = rans["coords_x"], rans["coords_y"], rans["coords_z"]
    iz = int(np.argmin(np.abs(coords_z - 100.0)))

    u_mean = np.linalg.norm(rans["U"], axis=0)[:, :, iz]
    u_mean_c, cx, cy = _crop_slice(u_mean, coords_x, coords_y)
    extent = [cx[0], cx[-1], cy[0], cy[-1]]

    field_path = _synthetic_field_path(field_case)
    frames_data: list[np.ndarray] = []
    time_labels: list[float] = []
    vmin = float(np.nanmin(u_mean_c))
    vmax = float(np.nanmax(u_mean_c))

    with h5py.File(field_path, "r") as f:
        time_s = f["time_s"][:]
        indices = [i for i in range(0, len(time_s), stride) if time_s[i] <= max_duration_s]
        print(f"  Loading {len(indices)} frames (stride={stride}) from {field_path.name} ...", flush=True)
        for i in indices:
            u_c, _, _ = _crop_slice(np.linalg.norm(f["U"][i], axis=0)[:, :, iz], coords_x, coords_y)
            frames_data.append(u_c.T)
            time_labels.append(float(time_s[i]))
            vmin = min(vmin, float(np.nanmin(u_c)))
            vmax = max(vmax, float(np.nanmax(u_c)))

    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    im = ax.imshow(
        frames_data[0], origin="lower", extent=extent, aspect="equal",
        vmin=vmin, vmax=vmax, cmap="viridis",
    )
    _overlay_probes(ax)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="|U| (m/s)")
    title = f"Synthetic |U| at z=100 m — {_case_title(case_id)}"
    if fallback:
        title += f"\n(frames from {_case_title(field_case)})"
    suptitle = fig.suptitle(title, fontsize=10)
    time_text = ax.set_title(f"t = {time_labels[0]:.0f} s")

    def _update(k: int):
        im.set_data(frames_data[k])
        time_text.set_text(f"t = {time_labels[k]:.0f} s")
        return im, time_text, suptitle

    anim = FuncAnimation(fig, _update, frames=len(frames_data), interval=1000 / fps, blit=False)
    out = out_dir / "gust_animation_z100m.gif"
    print(f"  Writing {out.name} ({len(frames_data)} frames @ {fps} fps) ...", flush=True)
    anim.save(out, writer=PillowWriter(fps=fps), dpi=120)
    plt.close(fig)
    return out


def plot_pipeline(out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(10, 2.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    ax.axis("off")

    steps = [
        (0.3, "1. Hourly RANS\n(simpleFoam)"),
        (2.8, "2. Extract U, k, ε, ν_t\n→ R, L, T on 40 m grid"),
        (5.3, "3. RFG synthesis\n2 Hz field, 20 Hz probes"),
        (7.8, "4. Validate +\nexport LUT"),
    ]
    box_w, box_h = 2.1, 1.35
    for x, label in steps:
        patch = FancyBboxPatch(
            (x, 0.85), box_w, box_h,
            boxstyle="round,pad=0.05,rounding_size=0.08",
            linewidth=1.0, edgecolor="black", facecolor="white",
        )
        ax.add_patch(patch)
        ax.text(x + box_w / 2, 1.55, label, ha="center", va="center", fontsize=9)

    for x0 in (2.4, 4.9, 7.4):
        ax.annotate(
            "", xy=(x0 + 0.35, 1.55), xytext=(x0, 1.55),
            arrowprops=dict(arrowstyle="->", lw=1.2),
        )

    ax.set_title("Synthetic wind pipeline: RANS statistics → second-scale gusts", fontsize=11, pad=12)
    out = out_dir / "pipeline.png"
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_mean_vs_gust(case_id: str, out_dir: Path, frame_t_s: float = 120.0) -> Path:
    field_case, fallback = _resolve_field_case(case_id)
    rans = load_enhanced_hdf5(case_id)
    coords_x, coords_y, coords_z = rans["coords_x"], rans["coords_y"], rans["coords_z"]
    iz = int(np.argmin(np.abs(coords_z - 100.0)))

    u_mean = np.linalg.norm(rans["U"], axis=0)[:, :, iz]

    field_path = _synthetic_field_path(field_case)
    with h5py.File(field_path, "r") as f:
        time_s = f["time_s"][:]
        iframe = int(np.argmin(np.abs(time_s - frame_t_s)))
        u_inst = np.linalg.norm(f["U"][iframe], axis=0)[:, :, iz]

    u_mean_c, cx, cy = _crop_slice(u_mean, coords_x, coords_y)
    u_inst_c, _, _ = _crop_slice(u_inst, coords_x, coords_y)
    vmin = float(np.nanmin([u_mean_c, u_inst_c]))
    vmax = float(np.nanmax([u_mean_c, u_inst_c]))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    extent = [cx[0], cx[-1], cy[0], cy[-1]]
    for ax, data, title in zip(
        axes,
        [u_mean_c, u_inst_c],
        ["RANS mean |U|", f"Synthetic |U| at t={time_s[iframe]:.0f} s"],
    ):
        im = ax.imshow(
            data.T, origin="lower", extent=extent, aspect="equal",
            vmin=vmin, vmax=vmax, cmap="viridis",
        )
        _overlay_probes(ax)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="|U| (m/s)")

    title = f"Horizontal slice at z=100 m — {_case_title(case_id)}"
    if fallback:
        title += f"\n(synthetic frame from {_case_title(field_case)})"
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out = out_dir / "mean_vs_gust_z100m.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def plot_probe_ts_psd(case_id: str, out_dir: Path) -> Path:
    probe_path = OUTPUT_ROOT / SYNTH_PROBE_SUBDIR / f"probes_{case_id}.h5"
    with h5py.File(probe_path, "r") as f:
        names = [n.decode() if isinstance(n, bytes) else str(n) for n in f["probe_names"][:]]
        idx = names.index("GAW103_z100")
        u = f["U"][:, idx, 0]
        dt = float(json.loads(f.attrs["meta"])["dt_s"])

    freqs, psd = signal.welch(u - np.mean(u), fs=1.0 / dt, nperseg=min(256, len(u) // 4))
    t = np.arange(len(u)) * dt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(t, u, lw=0.6)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("u (m/s)")
    axes[0].set_title("Along-wind component")
    axes[0].grid(True, alpha=0.3)

    axes[1].loglog(freqs[1:], psd[1:])
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("PSD")
    axes[1].set_title("Power spectrum")
    axes[1].grid(True, alpha=0.3, which="both")

    fig.suptitle(f"Synthetic wind at GAW103, z=100 m — {_case_title(case_id)}", fontsize=11)
    fig.tight_layout()
    out = out_dir / "probe_ts_psd.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def _load_reports() -> list[dict]:
    reports = []
    val_dir = OUTPUT_ROOT / VALIDATION_SUBDIR
    for p in sorted(val_dir.glob("report_*.json")):
        reports.append(json.loads(p.read_text(encoding="utf-8")))
    return reports


def plot_validation_summary(
    out_dir: Path,
    case_day: str = DEFAULT_CASE_DAY,
    case_night: str = DEFAULT_CASE_NIGHT,
) -> Path:
    reports = _load_reports()
    by_id = {r["case_id"]: r for r in reports}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax_bar = axes[0]

    sites = ["GAW103", "GAW104", "GAW111"]
    width = 0.35
    clusters = [("day", case_day, 0.0), ("night", case_night, 3.5)]
    for tag, cid, x0 in clusters:
        rep = by_id.get(cid)
        if not rep:
            continue
        site_map = {s["site"]: s for s in rep.get("sites", [])}
        x = x0 + np.arange(len(sites))
        ti_rans = [site_map[s]["ti_rans"] for s in sites]
        ti_syn = [site_map[s]["ti_synthetic"] for s in sites]
        ax_bar.bar(x - width / 2, ti_rans, width, color="C0")
        ax_bar.bar(x + width / 2, ti_syn, width, color="C1")
        mid = x0 + 1.0
        ax_bar.text(mid, -0.08, tag, ha="center", fontsize=8, style="italic", transform=ax_bar.get_xaxis_transform())

    ax_bar.set_xticks(list(np.arange(3)) + list(3.5 + np.arange(3)))
    ax_bar.set_xticklabels(sites + sites)
    ax_bar.set_ylabel("TI at z=100 m (-)")
    ax_bar.set_title("Turbulence intensity: RANS vs synthetic")
    ax_bar.legend(
        handles=[
            Patch(facecolor="C0", edgecolor="C0", label="RANS TI"),
            Patch(facecolor="C1", edgecolor="C1", label="Synthetic TI"),
        ],
        fontsize=7,
    )
    ax_bar.grid(True, axis="y", alpha=0.3)

    ax_sc = axes[1]
    bias = [r["mean_abs_bias_m_s"] for r in reports]
    vratio = [r["variance_ratio_mean"] for r in reports]
    passed = [
        r.get("pass_mean") and r.get("pass_variance")
        for r in reports
    ]
    colors_sc = ["C2" if p else "C3" for p in passed]
    ax_sc.scatter(bias, vratio, c=colors_sc, alpha=0.75, edgecolors="k", linewidths=0.4, s=40)
    ax_sc.axvline(0.15, color="gray", ls="--", lw=0.8)
    ax_sc.axhline(0.5, color="gray", ls=":", lw=0.8)
    ax_sc.axhline(2.0, color="gray", ls=":", lw=0.8)
    ax_sc.set_xlabel("Mean |U| bias (m/s)")
    ax_sc.set_ylabel("Variance ratio (synthetic / R_ii)")
    ax_sc.set_title(f"Validation metrics ({len(reports)} representative cases)")
    ax_sc.legend(
        handles=[
            Line2D([0], [0], color="gray", ls="--", lw=1.2, label="mean bias limit"),
            Line2D([0], [0], color="gray", ls=":", lw=1.2, label="variance ratio band"),
        ],
        fontsize=7,
        loc="upper right",
    )
    ax_sc.grid(True, alpha=0.3)

    fig.tight_layout()
    out = out_dir / "validation_summary.png"
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def copy_diurnal(out_dir: Path) -> Path:
    src = OUTPUT_ROOT / "statistics" / "daily_turbulence_stats.png"
    dst = out_dir / "diurnal_turbulence.png"
    if not src.is_file():
        raise FileNotFoundError(
            f"{src} not found; run: python analysis/synthetic_wind/summarize_daily_stats.py --plot"
        )
    shutil.copy2(src, dst)
    return dst


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot synthetic-wind presentation figures.")
    parser.add_argument("--all", action="store_true", help="Generate all figures")
    parser.add_argument(
        "--fig",
        choices=[
            "pipeline", "mean_vs_gust", "gust_animation", "probe_ts_psd",
            "validation_summary", "diurnal",
        ],
    )
    parser.add_argument("--case", default=DEFAULT_CASE_NIGHT)
    parser.add_argument("--case-day", default=DEFAULT_CASE_DAY)
    parser.add_argument("--case-night", default=DEFAULT_CASE_NIGHT)
    parser.add_argument("--stride", type=int, default=1, help="Frame stride for gust animation")
    parser.add_argument("--fps", type=int, default=8, help="GIF frame rate")
    parser.add_argument("--max-duration-s", type=float, default=60.0, help="Max sim time in animation")
    args = parser.parse_args()

    if not args.all and not args.fig:
        parser.error("Specify --all or --fig")

    out_dir = _ensure_out()
    jobs = (
        [
            "pipeline", "mean_vs_gust", "gust_animation", "diurnal",
            "probe_ts_psd", "validation_summary",
        ]
        if args.all
        else [args.fig]
    )

    for job in jobs:
        if job == "pipeline":
            path = plot_pipeline(out_dir)
        elif job == "mean_vs_gust":
            path = plot_mean_vs_gust(args.case, out_dir)
        elif job == "gust_animation":
            path = plot_gust_animation(
                args.case, out_dir,
                stride=args.stride, fps=args.fps, max_duration_s=args.max_duration_s,
            )
        elif job == "probe_ts_psd":
            path = plot_probe_ts_psd(args.case_night if args.all else args.case, out_dir)
        elif job == "validation_summary":
            path = plot_validation_summary(out_dir, args.case_day, args.case_night)
        elif job == "diurnal":
            path = copy_diurnal(out_dir)
        else:
            continue
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
