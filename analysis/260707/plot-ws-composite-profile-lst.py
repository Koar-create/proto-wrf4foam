"""
plot-ws-composite-profile-lst.py — 风速复合廓线，按 UTC 日输出 1×4 子图。

每张图对应一个 UTC 日，四个子图分别为当天 00 / 06 / 12 / 18:00 UTC 的
ws composite（LiDAR / WRF / OpenFOAM）。共享 y 轴（高度）。

数据筛选按 UTC（与 CSV 中 datetime 列一致）；子图标题可按 --tz 显示 LST 或 UTC。
实验/观测 CSV 覆盖 2025-09-06 00:00 .. 2025-09-13 18:00 UTC。
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
import argparse

import campaign_config as cfg

# ─── 路径与阈值配置（避免 Hardcoding）────────────────────────────────────────
REPO_ROOT = cfg.REPO_ROOT
DATA_PATH = cfg.DATA_PATH
OUTPUT_DIR = REPO_ROOT / "results/ws_composite_profile" / cfg.RESULTS_TAG

WS_MAX_OBS  = 30.0   # m/s: 观测物理上限（仪器异常阈值）
WS_MAX_CFD  = 20.0   # m/s: CFD 上限（>20 视为数值发散）

HEIGHT_BINS = [0, 300, 1000, 2100]                          # m：分层边界
LAYER_NAMES_EN = ["Low (52–300 m)",
                  "Mid (300–1000 m)",
                  "High (1000–2000 m)"]
HEIGHT_BIN_WIDTH = 50.0  # m：廓线聚合分箱宽度

METRIC_DATETIMES = cfg.METRIC_DATETIMES
TIME_LABELS = cfg.TIME_LABELS
SYNOPTIC_UTC_HOURS = cfg.SYNOPTIC_UTC_HOURS


def metric_utc_days() -> list[int]:
    return cfg.metric_utc_days()


# ─── 色盲友好配色（IBM Color Blind Safe Palette 变体）────────────────────────
COLOR_OBS = "#1a1a2e"   # 深蓝黑 – LiDAR 观测
COLOR_WRF = "#e07b39"   # 橙色   – WRF 中尺度
COLOR_CFD = "#2196a5"   # 青蓝   – OpenFOAM CFD

# ─── 全局 Matplotlib 学术样式设置 ───────────────────────────────────────────
def configure_matplotlib_style() -> None:
    plt.rcParams.update({
        'font.family':       'DejaVu Serif',   # 学术字体，类 LaTeX 效果
        'font.size':         10,
        'axes.labelsize':    11,
        'axes.titlesize':    12,
        'axes.titleweight':  'bold',
        'axes.linewidth':    0.8,
        'axes.grid':         True,
        'grid.alpha':        0.25,
        'grid.linestyle':    '--',
        'xtick.direction':   'in',
        'ytick.direction':   'in',
        'xtick.top':         True,
        'ytick.right':       True,
        'legend.framealpha': 0.9,
        'legend.edgecolor':  '0.8',
        'figure.dpi':        120,     # 屏幕显示
        'savefig.dpi':       300,     # 论文保存（GMD 要求 ≥300 DPI）
        'savefig.bbox':      'tight',
    })

def load_and_preprocess(path: str | Path) -> pd.DataFrame:
    """
        加载已对齐的三源数据 CSV，派生所有需要的物理量。

    派生量：
      - ws_cfd  : CFD 水平风速 = sqrt(u²+v²)
      - wd_obs/wrf/cfd : 气象风向（北=0°, 顺时针, 风从何方来）
      - layer   : 高度分层标签（低/中/高）
      - time_label : 人类可读时次标签
    """
    df = pd.read_csv(str(path), parse_dates=['datetime'])

    # 1. 派生 CFD 水平风速（原始只有 u_cfd, v_cfd）
    df['ws_cfd'] = np.sqrt(df['u_cfd']**2 + df['v_cfd']**2)

    # 2. 气象风向（arctan2 参数顺序：取反后为"从何方来"）
    df['wd_obs'] = np.degrees(np.arctan2(-df['u_obs'], -df['v_obs'])) % 360
    df['wd_wrf'] = np.degrees(np.arctan2(-df['u_wrf'], -df['v_wrf'])) % 360
    df['wd_cfd'] = np.degrees(np.arctan2(-df['u_cfd'], -df['v_cfd'])) % 360

    # 3. 高度分层
    df['layer'] = pd.cut(df['Height'], bins=HEIGHT_BINS, labels=LAYER_NAMES_EN)

    # 4. 可读时次标签
    df['time_label'] = df['datetime'].astype(str).map(TIME_LABELS)

    return df


def quality_control(df: pd.DataFrame,
                   ws_max_obs: float = WS_MAX_OBS,
                   ws_max_cfd: float = WS_MAX_CFD) -> pd.DataFrame:
    """
    数据质量控制（QC）。

    规则（按优先级）：
    ① obs NaN → 保留行但排除于统计（LiDAR 信噪比不足的缺测层，属有效缺测）
    ② ws_obs > ws_max_obs → 仪器异常，标记为无效
    ③ ws_cfd > ws_max_cfd → OpenFOAM 数值发散（RANS 在偏斜网格局部不收敛），标记整行无效

    注意：不对 WRF 执行额外 QC，WRF 数据本身经过诊断，物理范围合理。

    返回：原始 df 加 3 个布尔掩码列（qc_obs_ok, qc_cfd_ok, qc_ok）。
    """
    # CFD 发散标记
    cfd_ok = df['ws_cfd'] <= ws_max_cfd
    # 观测物理上限（NaN 不触发该规则）
    obs_ok = (df['ws_obs'] <= ws_max_obs) | df['ws_obs'].isna()

    df = df.copy()
    df['qc_obs_ok'] = obs_ok
    df['qc_cfd_ok'] = cfd_ok
    df['qc_ok'] = obs_ok & cfd_ok   # 行级别总掩码

    return df


# ─── 时区显示工具 ───────────────────────────────────────────────────────────
def _format_display_label(utc_ts: pd.Timestamp, tz: str) -> str:
    """根据 tz 选择子图标题: 'mm-dd HH:MM LST' 或 'mm-dd HH:MM UTC'。"""
    if tz == "utc":
        return utc_ts.strftime("%m-%d %H:%M UTC")
    lst_ts = utc_ts + pd.Timedelta(hours=8)
    return lst_ts.strftime("%m-%d %H:%M LST")


def _utc_slots_for_day(day: int) -> list[pd.Timestamp]:
    """返回某 UTC 日四个 synoptic 时次：00 / 06 / 12 / 18。"""
    return [
        pd.Timestamp(f"2025-09-{day:02d} {h:02d}:00:00")
        for h in SYNOPTIC_UTC_HOURS
    ]


def _height_bins(zmax: float) -> np.ndarray:
    """按 zmax 生成高度分箱（选值范围，而非仅作 ymax 显示）。"""
    return np.arange(0, float(zmax) + HEIGHT_BIN_WIDTH, HEIGHT_BIN_WIDTH)


def _ws_xlim(agg_obs: pd.DataFrame, agg_model: pd.DataFrame,
             pad: float = 1.25) -> tuple[float, float]:
    """按本子图数据（含 obs ± std）确定风速 x 轴上限，避免裁切或右侧空白过大。"""
    xmax = 1.0
    if not agg_obs.empty:
        xmax = max(
            xmax,
            float((agg_obs['mean_ws'] + agg_obs['std_ws'].fillna(0)).max()),
        )
    if not agg_model.empty:
        xmax = max(
            xmax,
            float(agg_model['mean_wrf'].max()),
            float(agg_model['mean_cfd'].max()),
        )
    return 0.0, xmax * pad


# ─── 主绘图函数 ─────────────────────────────────────────────────────────────
def plot_profiles_with_errorbars(df: pd.DataFrame,
                                 out_dir: Path = OUTPUT_DIR,
                                 tz: str = "lst",
                                 zmax: float = 1000.0) -> None:
    """
    按 UTC 日绘制风速复合廓线（每组 4 子图 1×4：00/06/12/18 UTC）。
    数据以 UTC 时刻过滤 (与 CSV 列对齐)；子图标题按 tz 显示 LST 或 UTC。
    zmax 作为选值上限：先筛 Height<=zmax，再按对应 bins 聚合。
    共享 y 轴；x 轴（风速）按各时次数据独立定标。
    """
    tz = tz.lower()
    if tz not in {"utc", "lst"}:
        raise ValueError("tz must be one of: utc, lst")

    height_bins = _height_bins(zmax)
    legend_handles = [
        Line2D([0], [0], marker='o', ms=5, color=COLOR_OBS, linestyle='none', label='LiDAR (obs)'),
        Line2D([0], [0], color=COLOR_WRF, lw=2, ls='--', label='WRF'),
        Line2D([0], [0], color=COLOR_CFD, lw=2, ls='-', label='OpenFOAM'),
    ]

    for day in metric_utc_days():
        utc_slots = _utc_slots_for_day(day)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4.5),
                                 sharex=False, sharey=True,
                                 constrained_layout=True)

        for ax, utc_ts in zip(axes, utc_slots):
            t_raw = utc_ts.strftime("%Y-%m-%d %H:%M:%S")
            tl_utc = TIME_LABELS.get(t_raw)

            if tl_utc is None:
                ax.axis('off')
                continue

            tl_disp = _format_display_label(utc_ts, tz=tz)

            # zmax 作为选值上限：先筛 Height，再按对应 bins 聚合
            sub = df[
                (df['time_label'] == tl_utc)
                & df['qc_ok']
                & (df['Height'] <= zmax)
            ].copy()

            if sub.empty:
                ax.set_title(tl_disp, fontweight='bold', pad=8)
                ax.set_xlim(0, 1.0)
                ax.set_ylim(0, zmax)
                continue

            sub['H_bin'] = pd.cut(sub['Height'], bins=height_bins)

            agg_obs = pd.DataFrame()
            sub_obs = sub.dropna(subset=['ws_obs']).copy()
            if not sub_obs.empty:
                agg_obs = sub_obs.groupby('H_bin', observed=True).agg(
                    mean_ws=('ws_obs', 'mean'),
                    std_ws=('ws_obs', 'std'),
                    mean_h=('Height', 'mean'),
                ).dropna(subset=['mean_h']).reset_index()
                ax.errorbar(agg_obs['mean_ws'], agg_obs['mean_h'],
                            xerr=agg_obs['std_ws'].fillna(0),
                            fmt='o', ms=3, color=COLOR_OBS, alpha=0.85,
                            elinewidth=0.8, capsize=2, zorder=5)

            agg_model = sub.groupby('H_bin', observed=True).agg(
                mean_h=('Height', 'mean'),
                mean_wrf=('ws_wrf', 'mean'),
                mean_cfd=('ws_cfd', 'mean'),
            ).dropna(subset=['mean_h']).reset_index()

            ax.plot(agg_model['mean_wrf'], agg_model['mean_h'],
                    color=COLOR_WRF, lw=2.0, ls='--')
            ax.plot(agg_model['mean_cfd'], agg_model['mean_h'],
                    color=COLOR_CFD, lw=2.0, ls='-')

            for h_line in (300, 1000):
                if h_line <= zmax:
                    ax.axhline(h_line, color='0.6', lw=0.8, ls=':', zorder=0)

            ax.set_title(tl_disp, fontweight='bold', pad=8)
            ax.set_xlim(*_ws_xlim(agg_obs, agg_model))
            ax.set_ylim(0, zmax)
            ax.set_xlabel(r'Wind Speed (m s$^{-1}$)', fontsize=11)

        axes[0].set_ylabel('Height (m a.g.l.)', fontsize=11)
        fig.legend(handles=legend_handles, fontsize=9,
                   loc='upper center', ncol=3, framealpha=0.9,
                   bbox_to_anchor=(0.5, 0.02))

        fig.suptitle(
            f'Wind Speed Composite Profiles — 2025-09-{day:02d} '
            f'(00/06/12/18 UTC)',
            fontsize=14, fontweight='bold')

        save_path = out_dir / (
            f"fig4_ws_composite_09{day:02d}_z{int(zmax)}m_tz-{tz}.png"
        )
        fig.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"Saved: {save_path}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate Fig4 wind speed composite profiles: 1×4 panels "
                    "per UTC day (00/06/12/18 UTC), shared y-axis.")
    p.add_argument("--tz", choices=["utc", "lst"], default="lst",
                   help="Subplot title time zone only: 'lst' (default) or 'utc'. "
                        "X-axis label is wind speed only (no LST/UTC suffix).")
    p.add_argument("--zmax", type=float, default=1000.0,
                   help="Height selection cutoff (m): keep Height<=zmax for "
                        "aggregation, then set ylim. Default: 1000.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    configure_matplotlib_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = quality_control(load_and_preprocess(DATA_PATH))
    plot_profiles_with_errorbars(df, OUTPUT_DIR, tz=args.tz, zmax=args.zmax)


if __name__ == "__main__":
    main()
