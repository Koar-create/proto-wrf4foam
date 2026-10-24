#!/usr/bin/env python3
"""P1: M1 admissibility diagnostics + M2 turbulence-front Hovmöller."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lesio import load_les_probes, read_vtp_points_U, vtp_path
from paths import CACHE, FIGS, LES_WIN_A, LES_WIN_B, LES_WIN_FULL, OUT, PROBE_Z, STATIONS, TIMES
from stats import (
    ar1_sigma_rel_err,
    batch_mean_sigma_ci,
    block_series,
    haar_mrd,
    integral_time_scale,
    lenschow_var_rel_err,
    mann_kendall,
    reverse_arrangement_pvalue,
    runs_test_pvalue,
)
from style import C_LES, C_URANS, INK, INK2, SEQ, apply, panel_label, savefig

apply()


WINDOWS = {
    "1000-1400": LES_WIN_B,
    "1400-2000": LES_WIN_A,
    "1000-2000": LES_WIN_FULL,
}


def _series(U, t, j, t0, t1):
    sel = (t >= t0) & (t <= t1)
    u = U[sel, j]
    tt = t[sel]
    uh = np.hypot(u[:, 0], u[:, 1])
    return tt, u[:, 0], u[:, 1], u[:, 2], uh


def m1_table(les) -> pd.DataFrame:
    t, U = les["t"], les["U"]
    dt = float(np.median(np.diff(t)))
    rows = []
    for name, meta in STATIONS.items():
        for iz, z in enumerate(PROBE_Z):
            j = meta["i0"] + iz
            for wname, (t0, t1) in WINDOWS.items():
                tt, uu, vv, ww, uh = _series(U, t, j, t0, t1)
                tau = integral_time_scale(uh, dt)
                bc, bmean, bsig, _ = block_series(tt, uh, t0, t1, 50.0)
                ra_m = reverse_arrangement_pvalue(bmean)
                ra_s = reverse_arrangement_pvalue(bsig)
                ru_m = runs_test_pvalue(bmean)
                ru_s = runs_test_pvalue(bsig)
                mk = mann_kendall(bmean)
                bm = batch_mean_sigma_ci(uh, dt, tau)
                ar = ar1_sigma_rel_err(uh, dt)
                T = t1 - t0
                rel_L = lenschow_var_rel_err(tau, T)
                # grade: developing if p<0.01 on block-mean reverse-arrangement
                if ra_m["p"] < 0.01 or ra_s["p"] < 0.01:
                    grade = "B-developing"
                elif ra_m["p"] >= 0.05 and ra_s["p"] >= 0.05:
                    grade = "B-quieter"
                else:
                    grade = "B"
                rows.append(dict(
                    station=name, z=z, window=wname,
                    n=int(uh.size),
                    ws_mean=float(np.mean(uh)),
                    sigma_uh=float(np.std(uh, ddof=1)),
                    sigma_u=float(np.std(uu, ddof=1)),
                    sigma_w=float(np.std(ww, ddof=1)),
                    tau_int_s=tau,
                    n_eff=T / tau if tau and np.isfinite(tau) and tau > 0 else np.nan,
                    ra_mean_p=ra_m["p"], ra_sig_p=ra_s["p"],
                    runs_mean_p=ru_m["p"], runs_sig_p=ru_s["p"],
                    mk_slope=mk["slope"], mk_p=mk["p"],
                    sigma_ci_half=bm["half"], sigma_ci_nbat=bm["nbat"],
                    ar1_rel=ar["rel"], lenschow_rel=rel_L,
                    grade=grade,
                ))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "m1_stationarity.csv", index=False)
    return df


def write_admissibility_report(df: pd.DataFrame):
    sub = df[df.window == "1400-2000"].copy()
    lines = [
        "# 可采性报告（M1）",
        "",
        "本报告依据 50 s 分块均值与分块 σ 的逆序排列检验、游程检验以及 Mann–Kendall 斜率，对 33 个探针在三个时间窗内的非平稳性给出定量诊断。",
        "正文结论仍只建立在 A 级量（逐快照、短滞后）之上；此处的窗口统计一律按 B 级处理，并附 95% 置信区间。",
        "",
        "## 判定规则",
        "",
        "- **B-quieter**：1400–2000 s 内块均值与块 σ 的逆序排列检验均满足 p ≥ 0.05，趋势不显著。",
        "- **B**：至少一项 0.01 ≤ p < 0.05。",
        "- **B-developing**：至少一项 p < 0.01，表明该高度仍在发展，窗口统计仅作佐证。",
        "- 不给出 C 级“已收敛 TI”或系综平均。",
        "",
        "## 1400–2000 s 逐高度判定",
        "",
    ]
    for name in STATIONS:
        g = sub[sub.station == name]
        lines.append(f"### {name}")
        lines.append("")
        lines.append("| z (m) | Ws (m/s) | σ_Uh (m/s) | τ_int (s) | n_eff | RA mean p | RA σ p | 判定 |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---|")
        for _, r in g.iterrows():
            lines.append(
                f"| {r.z:.0f} | {r.ws_mean:.3f} | {r.sigma_uh:.3f} | {r.tau_int_s:.1f} | "
                f"{r.n_eff:.1f} | {r.ra_mean_p:.3g} | {r.ra_sig_p:.3g} | {r.grade} |"
            )
        lines.append("")
    lines += [
        "## 续算优先级（供内部决策，本方案不据此提出续算）",
        "",
        "若未来延长积分，应优先选择 1400–2000 s 仍为 B-developing、且 UAV 航线经过的 30–120 m 高度，",
        "以及 GAW103 尾流柱；400–600 m 若已落入 B-quieter，则延长积分的边际收益较小。",
        "",
        f"全表见 `m1_stationarity.csv`（{len(df)} 行）。",
    ]
    (OUT / "admissibility_report.md").write_text("\n".join(lines), encoding="utf-8")


def fig_m1_heatmaps(df: pd.DataFrame):
    from matplotlib.patches import Patch
    grade_color = {"B-quieter": C_LES, "B": C_URANS, "B-developing": SEQ[6]}
    sub = df[df.window == "1400-2000"]
    fig, axs = plt.subplots(1, 3, figsize=(7.1, 3.4), sharey=True)
    z = np.asarray(PROBE_Z, float)
    for ax, name in zip(axs, STATIONS):
        g = sub[sub.station == name].set_index("z").loc[z]
        cols = [grade_color.get(gr, C_LES) for gr in g.grade]
        ax.barh(z, g.sigma_uh, color=cols, height=12, alpha=0.9)
        for zz, pp, gr in zip(z, g.ra_mean_p, g.grade):
            ax.text(0.02, zz, f"p={pp:.2g}", va="center", fontsize=5.6, color=INK2,
                    transform=ax.get_yaxis_transform())
        ax.set_title(name, loc="left")
        ax.set_xlabel(r"$\sigma_{U_h}$ (m s$^{-1}$)")
        ax.set_ylim(40, 640)
    axs[0].set_ylabel("z AGL (m)")
    fig.suptitle("1400–2000 s block-mean reverse-arrangement p on σ(U_h)", x=0.01, ha="left", fontsize=8, color=INK)
    fig.legend(
        [Patch(color=grade_color[k]) for k in ("B-quieter", "B", "B-developing")],
        ["B-quieter (p ≥ 0.05)", "B (0.01 ≤ p < 0.05)", "B-developing (p < 0.01)"],
        loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.02), fontsize=6.5,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    savefig(fig, FIGS / "m1_sigma_grade.png")
    plt.close(fig)


def fig_f01_real(les, df, lidar_mod=None):
    """F01 with real LES panels b–d. Panel a still needs lidar timestamps (caller)."""
    from sampling import load_lidar, window as lidar_window

    t, U = les["t"], les["U"]
    dt = float(np.median(np.diff(t)))
    j = STATIONS["GAW103"]["i0"] + 2
    uh = np.hypot(U[:, j, 0], U[:, j, 1])
    bc, _, bsig, _ = block_series(t, uh, 150.0, 2000.0, 50.0)

    fig = plt.figure(figsize=(7.1, 4.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.8, 1.15], hspace=0.55, wspace=0.42)

    ax = fig.add_subplot(gs[0, :])
    d = lidar_window(load_lidar())
    for k, s in enumerate(STATIONS):
        g = d[(d.obtid == s) & (np.isclose(d.Height, 104))]
        g = g[(g.datetime >= "2025-09-01 10:00") & (g.datetime < "2025-09-01 10:10")]
        tt = (g.datetime - pd.Timestamp("2025-09-01 10:00")).dt.total_seconds()
        ax.vlines(tt, k - 0.32, k + 0.32, color=INK, lw=0.6)
        dtg = np.diff(np.sort(tt.to_numpy())) if len(tt) > 1 else np.array([np.nan])
        ax.text(606, k, f"{s}  median Δt {np.nanmedian(dtg):.0f} s · {len(tt)} profiles",
                va="center", fontsize=6.6, color=INK2)
    ax.set_yticks([])
    ax.set_xlim(0, 600)
    ax.set_ylim(-0.6, 2.6)
    ax.set_xlabel("time after 10:00 UTC (s)")
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.set_title("Real DBS sampling, 104 m gate", loc="left")
    panel_label(ax, "a")

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(bc, bsig, color=C_LES, lw=1.6, marker="o", ms=2.6)
    ax.axvspan(1400, 2000, color="#e6e5e1", alpha=0.8, lw=0)
    g = df[(df.station == "GAW103") & (df.z == 100) & (df.window == "1400-2000")].iloc[0]
    ax.text(0.03, 0.97, f"1400–2000 s\nRA p(mean)={g.ra_mean_p:.2g}\nRA p(σ)={g.ra_sig_p:.2g}\n{g.grade}",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.0, color=INK2)
    ax.set_xlabel("t (s)")
    ax.set_ylabel(r"50 s block $\sigma_{U_h}$ (m s$^{-1}$)")
    ax.set_title("GAW103 z=100 m", loc="left")
    panel_label(ax, "b")

    ax = fig.add_subplot(gs[1, 1])
    for (a, b), col, lab in [((1000, 1400), INK2, "1000–1400 s"), ((1400, 2000), C_LES, "1400–2000 s")]:
        sel = (t >= a) & (t < b)
        sc, v = haar_mrd(uh[sel], dt)
        ax.semilogx(sc, v, color=col, lw=1.4, marker="o", ms=2.4, label=lab)
    ax.axvspan(0.1, 90, color="#e6e5e1", lw=0, alpha=0.8)
    ax.set_xlim(0.3, 1500)
    ax.set_xlabel("averaging scale (s)")
    ax.set_ylabel(r"MRD variance (m$^2$ s$^{-2}$)")
    ax.set_title("Where the variance sits", loc="left")
    ax.legend(loc="upper left", fontsize=6.2)
    panel_label(ax, "c")

    ax = fig.add_subplot(gs[1, 2])
    Ts = np.array([100, 200, 300, 400, 600])
    for z, col, lab in [(100, C_LES, "z=100 m"), (400, INK2, "z=400 m")]:
        jz = STATIONS["GAW103"]["i0"] + list(PROBE_Z).index(z)
        sel = (t >= 1400) & (t <= 2000)
        xx = np.hypot(U[sel, jz, 0], U[sel, jz, 1])
        tau = integral_time_scale(xx, dt)
        rels = []
        for T in Ts:
            n = min(int(round(T / dt)), xx.size)
            bm = batch_mean_sigma_ci(xx[:n], dt, tau)
            rels.append(100 * bm["half"] / bm["sigma"] if bm["sigma"] and np.isfinite(bm["half"]) else np.nan)
        ax.plot(Ts, rels, color=col, lw=1.6, marker="o", ms=2.6, label=fr"{lab} ($\tau\approx${tau:.0f} s)")
    ax.axhline(20, color="#8a8985", lw=0.8, ls=":", label="±20 % target")
    ax.set_xlim(80, 820)
    ax.set_xlabel("window length T (s)")
    ax.set_ylabel(r"95 % CI of $\sigma$ (±%)")
    ax.set_title("Sampling uncertainty", loc="left")
    ax.legend(loc="upper right", fontsize=6.0)
    panel_label(ax, "d")
    savefig(fig, FIGS / "F01_sampling_admissibility.png")
    plt.close(fig)


def m2_front(z=100, tile=200.0, x0=975.0, y0=-5000.0, y1=-320.0):
    """200 m tiles of spatial σ(|Uh|) along x=x0.

    Outer 100 m cells yield only a few cutting-plane points per 200 m tile,
    so the x-corridor is 400 m wide and the minimum count is 3.
    """
    dest = CACHE / f"m2_front_x{int(x0)}_z{z}.npz"
    ys = np.arange(y0 + tile / 2, y1 + tile / 2, tile)
    sig = np.full((len(TIMES), len(ys)), np.nan, np.float32)
    npts = np.zeros((len(TIMES), len(ys)), np.int16)
    x_half = 400.0
    for it, t in enumerate(TIMES):
        pts, vec = read_vtp_points_U(vtp_path(z, t))
        uh = np.hypot(vec[:, 0], vec[:, 1])
        x, y = pts[:, 0], pts[:, 1]
        inx = np.abs(x - x0) <= x_half
        for iy, yc in enumerate(ys):
            m = inx & (np.abs(y - yc) <= tile / 2)
            npts[it, iy] = int(m.sum())
            if m.sum() >= 3:
                sig[it, iy] = np.std(uh[m], ddof=1)
        if it % 8 == 0:
            print(f"  M2 front t={t}", flush=True)
    np.savez_compressed(
        dest, t=np.asarray(TIMES, float), y=ys, sig=sig, npts=npts,
        x0=x0, z=z, tile=tile, x_half=x_half,
    )
    return dest


def fig_f07(front_npz, target=1.13):
    z = np.load(front_npz)
    t, y, sig = z["t"], z["y"], z["sig"]
    dist = y - y.min()  # from first tile centre; south inlet is y=-5000
    dist_from_inlet = y - (-5000.0)
    fig, axs = plt.subplots(1, 2, figsize=(7.1, 2.9), gridspec_kw=dict(width_ratios=[1.35, 1]))
    ax = axs[0]
    im = ax.pcolormesh(dist_from_inlet / 1000.0, t, sig / target, cmap="Blues", vmin=0, vmax=1.2, shading="nearest")
    ax.axvline((-320.0 + 5000) / 1000.0, color=INK2, lw=0.8, label="GAW103")
    ax.legend(loc="lower right", fontsize=6.2)
    ax.set_xlabel("distance from south inlet along x = 975 m (km)")
    ax.set_ylabel("LES time t (s)")
    ax.set_title(r"Turbulence front z=100 m: tile $\sigma_{U_h}$ / 1.13", loc="left")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02).set_label("normalised σ")
    panel_label(ax, "a")
    ax = axs[1]
    for (t0, t1), col, lab in [((150, 1000), INK2, "150–1000 s"), ((1050, 2000), C_LES, "1050–2000 s")]:
        sel = (t >= t0) & (t <= t1)
        with np.errstate(all="ignore"):
            prof = np.nanmean(sig[sel] / target, axis=0)
        ax.plot(dist_from_inlet / 1000.0, prof, color=col, lw=1.5, label=lab)
    ax.axhline(1, color=INK2, lw=0.6, label="inlet target")
    ax.set_xlabel("distance from south inlet (km)")
    ax.set_ylabel("σ / target")
    ax.set_title("Inflow-turbulence recovery", loc="left")
    ax.legend(loc="upper left", fontsize=6.2)
    panel_label(ax, "b")
    fig.tight_layout()
    savefig(fig, FIGS / "F07_turbulence_front.png")
    plt.close(fig)


def main():
    t0 = time.time()
    FIGS.mkdir(parents=True, exist_ok=True)
    les = load_les_probes()
    print("[P1] M1 stationarity …", flush=True)
    df = m1_table(les)
    write_admissibility_report(df)
    fig_m1_heatmaps(df)
    print("[P1] F01 …", flush=True)
    fig_f01_real(les, df)
    print("[P1] M2 front (38 VTP) …", flush=True)
    nz = m2_front()
    fig_f07(nz)
    print(f"[P1] done in {time.time()-t0:.1f}s", flush=True)
    print(f"  {OUT / 'admissibility_report.md'}")
    print(f"  {FIGS / 'F01_sampling_admissibility.png'}")
    print(f"  {FIGS / 'F07_turbulence_front.png'}")


if __name__ == "__main__":
    main()
