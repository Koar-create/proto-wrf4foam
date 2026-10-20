"""Figures for the pilot wake diagnosis.

  python3 analysis/261009-wake-xr/plot_wake.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_wake_metrics import ambient_profile, band_mean, interp_amb

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "data" / "261009" / "raw" / "wake_boxes"
MET = REPO / "data" / "261009" / "processed" / "wake_metrics.csv"
FIG = REPO / "results" / "wake_xr" / "261009"
KE_LO, KE_HI = 2.1 / 1.45, 2.6 / 1.42


def _us(U, ex):
    return U[..., 0] * ex[0] + U[..., 1] * ex[1]


def plane_mosaic(case: str, metrics: pd.DataFrame) -> None:
    case_dir = RAW / case
    paths = sorted(case_dir.glob("*.npz"))
    if not paths:
        print(f"no planes for {case}")
        return
    ncol = 7
    nrow = int(np.ceil(len(paths) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.15, nrow * 2.05), squeeze=False)
    meta = metrics[(metrics["case_id"] == case) & (metrics["z_tag"] == "z0.25H")].set_index("bid")
    for ax, path in zip(axes.ravel(), paths):
        blob = np.load(path)
        bid = int(blob["bid"])
        ex = blob["ex"]
        H = float(blob["H"])
        x_lee = float(blob["x_lee"])
        x = blob["x_plane"]
        y = blob["y_plane"]
        us = _us(blob["U_plane"][0], ex)
        us = np.where(blob["valid_plane"][0] > 0, us, np.nan)
        extent = ((x[0] - x_lee) / H, (x[-1] - x_lee) / H, y[0] / H, y[-1] / H)
        vmax = np.nanpercentile(np.abs(us), 98) if np.isfinite(us).any() else 1.0
        vmax = max(float(vmax), 0.2)
        ax.imshow(
            us,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.axvline(0.0, color="k", lw=0.6)
        xr = np.nan
        iso = np.nan
        if bid in meta.index:
            xr = float(meta.loc[bid, "XR_over_H"])
            iso = float(meta.loc[bid, "I"])
            if np.isfinite(xr):
                ax.axvline(xr, color="#b00020", lw=0.7)
        ax.set_xlim(-0.3, 4.0)
        ax.set_title(f"{bid}  I={iso:.1f}" + (f"  XR/H={xr:.2f}" if np.isfinite(xr) else "  no XR"), fontsize=7)
        ax.tick_params(labelsize=6)
    for ax in axes.ravel()[len(paths) :]:
        ax.axis("off")
    fig.suptitle(f"{case}   streamwise velocity at z/H = 0.25    red = X_R", fontsize=11)
    fig.supxlabel(r"$(x - x_{lee})/H$")
    fig.supylabel(r"$y/H$")
    fig.tight_layout()
    out = FIG / f"plane_us_z025H_{case}.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"wrote {out}")


def scatter_xr(metrics: pd.DataFrame) -> None:
    ground = metrics[(metrics["z_tag"] == "z3m") & (metrics["status"] == "reattached")].copy()
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.4))
    ax = axes[0]
    cases = list(ground["case_id"].unique())
    markers = ["o", "s"]
    short = {
        "20250901_1000_two_boundaries_as_outlet": "2025-09-01 10:00 UTC",
        "20250908_0000_two_boundaries_as_outlet": "2025-09-08 00:00 UTC",
    }
    for case, marker in zip(cases, markers):
        g = ground[ground["case_id"] == case]
        sc = ax.scatter(
            g["LR_fackrell_m"],
            g["XR_m"],
            c=np.clip(g["I"], 2, 8),
            cmap="viridis",
            vmin=2,
            vmax=8,
            marker=marker,
            s=28,
            edgecolors="k",
            linewidths=0.3,
            label=short.get(case, case),
        )
    xx = np.linspace(0, max(ground["LR_fackrell_m"].max(), 1) * 1.05, 50)
    ax.plot(xx, xx, color="0.4", lw=0.8, label="r = 1")
    ax.plot(xx, KE_LO * xx, color="#b00020", lw=0.8, ls="--", label=f"Yu k-e/exp {KE_LO:.2f}")
    ax.plot(xx, KE_HI * xx, color="#b00020", lw=0.8, ls=":", label=f"Yu k-e/exp {KE_HI:.2f}")
    ax.set_xlabel("Fackrell L_R (m)")
    ax.set_ylabel("RANS X_R at z = 3 m (m)")
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.legend(fontsize=7, loc="upper left")
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("isolation I (clipped 2–8)")

    ax = axes[1]
    a = metrics[(metrics["case_id"] == cases[0]) & (metrics["z_tag"] == "z3m")]
    if len(cases) > 1:
        b = metrics[(metrics["case_id"] == cases[1]) & (metrics["z_tag"] == "z3m")]
        m = a.merge(b, on="bid", suffixes=("_a", "_b"))
        both = m[(m["status_a"] == "reattached") & (m["status_b"] == "reattached")]
        ax.scatter(both["XR_over_H_a"], both["XR_over_H_b"], c=np.clip(both["I_a"], 2, 8), cmap="viridis", vmin=2, vmax=8, s=28, edgecolors="k", linewidths=0.3)
        lim = max(both["XR_over_H_a"].max(), both["XR_over_H_b"].max(), 0.1) * 1.1
        ax.plot([0, lim], [0, lim], color="0.4", lw=0.8)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_xlabel(r"$X_R/H$  20250901 10:00")
        ax.set_ylabel(r"$X_R/H$  20250908 00:00")
        ax.set_title("same building, two inflows")
        ax.set_aspect("equal")
    else:
        ax.set_axis_off()
        ax.text(0.1, 0.5, "second case not ready")
    fig.tight_layout()
    out = FIG / "xr_vs_fackrell.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"wrote {out}")


def decay_figure(metrics: pd.DataFrame) -> None:
    """Log-log deficit for the most isolated buildings that reattach at z/H = 0.25."""
    sub = metrics[(metrics["z_tag"] == "z0.25H") & (metrics["status"] == "reattached")].copy()
    if sub.empty:
        print("no decay curves")
        return
    # One panel per building, the eight highest isolation indices.
    bids = (
        sub.sort_values("I", ascending=False)["bid"].drop_duplicates().head(8).tolist()
    )
    fig, axes = plt.subplots(2, 4, figsize=(11.5, 5.6), squeeze=False)
    for ax, bid in zip(axes.ravel(), bids):
        for case, color in zip(sorted(sub["case_id"].unique()), ("#1f4e79", "#c47b2b")):
            path = RAW / case / f"{bid}.npz"
            row = sub[(sub["bid"] == bid) & (sub["case_id"] == case)]
            if not path.is_file() or row.empty:
                continue
            blob = np.load(path)
            ex = blob["ex"]
            prof = band_mean(blob["U_center"], blob["valid_center"], ex)
            iz = int(np.argmin(np.abs(blob["z_center"] - 0.25 * float(blob["H"]))))
            z_amb, u_amb = ambient_profile(blob["U_amb"], blob["valid_amb"], blob["z_amb"], ex)
            ua = interp_amb(z_amb, u_amb, float(blob["z_center"][iz]))
            x = blob["x_center"]
            x_lee = float(blob["x_lee"])
            H = float(blob["H"])
            xr = float(row["XR_m"].iloc[0])
            xi = (x - x_lee) / H
            deficit = ua - prof[iz]
            m = (x >= x_lee + xr) & (xi <= 10) & np.isfinite(deficit) & (deficit > 0.02)
            if m.sum() < 3:
                continue
            ax.loglog(xi[m], deficit[m], color=color, lw=1.0)
            p = float(row["p_fit"].iloc[0])
            r2 = float(row["p_r2"].iloc[0])
            # The 10H window usually runs into the next building, so a power law
            # with r2 below 0.3 is not drawn.
            if np.isfinite(p) and np.isfinite(r2) and r2 >= 0.3:
                xx = xi[m]
                aa, bb = float(row["p_a"].iloc[0]), float(row["p_b"].iloc[0])
                ax.loglog(xx, aa + bb * xx ** (-p), color=color, lw=0.6, ls="--")
            if np.isfinite(p):
                ax.text(
                    0.04,
                    0.08 if color == "#1f4e79" else 0.22,
                    f"p={p:.2f} r2={r2:.2f}",
                    transform=ax.transAxes,
                    color=color,
                    fontsize=6,
                )
        I = float(sub[sub["bid"] == bid]["I"].iloc[0])
        ax.set_title(f"bid {bid}  I={I:.1f}", fontsize=8)
        ax.tick_params(labelsize=6)
        ax.set_xlim(0.05, 12)
    for ax in axes.ravel()[len(bids) :]:
        ax.axis("off")
    fig.supxlabel(r"$(x-x_{lee})/H$")
    fig.supylabel(r"$\Delta U$ (m/s)")
    fig.suptitle("Main-wake deficit at z/H = 0.25. Solid: samples. Dashed: a + b (x/H)^(-p).", fontsize=10)
    fig.tight_layout()
    out = FIG / "deficit_decay.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(MET)
    for case_dir in sorted(p for p in RAW.iterdir() if p.is_dir() and not p.name.startswith("_")):
        plane_mosaic(case_dir.name, metrics)
    scatter_xr(metrics)
    decay_figure(metrics)


if __name__ == "__main__":
    main()
