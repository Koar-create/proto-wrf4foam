"""Wind-speed / wind-direction metric functions, QC, and table builders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from metrics.layers import (
    HEIGHT_BINS,
    LAYER_NAMES_EN,
    WS_MAX_CFD,
    WS_MAX_OBS,
    summary_layers,
)


def mean_bias_error(model: np.ndarray, obs: np.ndarray) -> float:
    return float(np.nanmean(model - obs))


def rmse(model: np.ndarray, obs: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean((model - obs) ** 2)))


def index_of_agreement(model: np.ndarray, obs: np.ndarray) -> float:
    obs_mean = np.nanmean(obs)
    numerator = np.nansum((model - obs) ** 2)
    denominator = np.nansum((np.abs(model - obs_mean) + np.abs(obs - obs_mean)) ** 2)
    if denominator == 0:
        return np.nan
    return float(1 - numerator / denominator)


def fractional_bias(model: np.ndarray, obs: np.ndarray) -> float:
    m_bar = np.nanmean(model)
    o_bar = np.nanmean(obs)
    denom = m_bar + o_bar
    return float(2 * (m_bar - o_bar) / denom) if denom != 0 else np.nan


def skill_score(model: np.ndarray, obs: np.ndarray, baseline: np.ndarray) -> float:
    mse_model = np.nanmean((model - obs) ** 2)
    mse_baseline = np.nanmean((baseline - obs) ** 2)
    return float(1 - mse_model / mse_baseline) if mse_baseline != 0 else np.nan


def circular_error_stats(wd_model: np.ndarray, wd_obs: np.ndarray) -> tuple[float, float]:
    wd_model = np.asarray(wd_model, dtype=float)
    wd_obs = np.asarray(wd_obs, dtype=float)
    valid = ~(np.isnan(wd_model) | np.isnan(wd_obs))
    if valid.sum() < 3:
        return np.nan, np.nan
    delta = wd_model[valid] - wd_obs[valid]
    delta = (delta + 180) % 360 - 180
    circ_mean_err = float(
        np.degrees(np.arctan2(np.mean(np.sin(np.radians(delta))), np.mean(np.cos(np.radians(delta)))))
    )
    circ_rmse = float(np.sqrt(np.mean(delta**2)))
    return circ_mean_err, circ_rmse


def load_and_preprocess(
    path: str | Path,
    *,
    time_labels: dict[str, str] | None = None,
) -> pd.DataFrame:
    df = pd.read_csv(str(path), parse_dates=["datetime"])
    df["ws_cfd"] = np.sqrt(df["u_cfd"] ** 2 + df["v_cfd"] ** 2)
    df["wd_obs"] = np.degrees(np.arctan2(-df["u_obs"], -df["v_obs"])) % 360
    df["wd_wrf"] = np.degrees(np.arctan2(-df["u_wrf"], -df["v_wrf"])) % 360
    df["wd_cfd"] = np.degrees(np.arctan2(-df["u_cfd"], -df["v_cfd"])) % 360
    df["layer"] = pd.cut(df["Height"], bins=HEIGHT_BINS, labels=LAYER_NAMES_EN)
    if time_labels is not None:
        df["time_label"] = df["datetime"].astype(str).map(time_labels)
    return df


def quality_control(
    df: pd.DataFrame,
    ws_max_obs: float = WS_MAX_OBS,
    ws_max_cfd: float = WS_MAX_CFD,
) -> pd.DataFrame:
    cfd_ok = df["ws_cfd"] <= ws_max_cfd
    obs_ok = (df["ws_obs"] <= ws_max_obs) | df["ws_obs"].isna()
    out = df.copy()
    out["qc_obs_ok"] = obs_ok
    out["qc_cfd_ok"] = cfd_ok
    out["qc_ok"] = obs_ok & cfd_ok
    return out


def compute_metrics_table(df: pd.DataFrame) -> pd.DataFrame:
    subset = df[df["qc_ok"] & df["ws_obs"].notna()].copy()
    records = []
    for (site, time, layer), grp in subset.groupby(["obtid", "time_label", "layer"], observed=True):
        if len(grp) < 5:
            continue
        obs = grp["ws_obs"].values
        wrf = grp["ws_wrf"].values
        cfd = grp["ws_cfd"].values
        wrf_cme, wrf_crmse = circular_error_stats(grp["wd_wrf"].values, grp["wd_obs"].values)
        cfd_cme, cfd_crmse = circular_error_stats(grp["wd_cfd"].values, grp["wd_obs"].values)
        records.append(
            {
                "Site": site,
                "Time": time,
                "Layer": layer,
                "N": len(grp),
                "WRF_MBE": mean_bias_error(wrf, obs),
                "WRF_RMSE": rmse(wrf, obs),
                "WRF_IoA": index_of_agreement(wrf, obs),
                "WRF_FB": fractional_bias(wrf, obs),
                "CFD_MBE": mean_bias_error(cfd, obs),
                "CFD_RMSE": rmse(cfd, obs),
                "CFD_IoA": index_of_agreement(cfd, obs),
                "CFD_FB": fractional_bias(cfd, obs),
                "CFD_SS_vs_WRF": skill_score(cfd, obs, wrf),
                "WRF_WD_CME_circ": wrf_cme,
                "WRF_WD_CRMSE_circ": wrf_crmse,
                "CFD_WD_CME_circ": cfd_cme,
                "CFD_WD_CRMSE_circ": cfd_crmse,
            }
        )
    result = pd.DataFrame(records)
    num_cols = result.select_dtypes(include="number").columns
    result[num_cols] = result[num_cols].round(3)
    return result


def compute_layer_summary(df: pd.DataFrame, *, skip_high: bool = False) -> pd.DataFrame:
    sub_df = df[df["qc_ok"] & df["ws_obs"].notna()].copy()
    rows: list[dict] = []
    for layer in summary_layers(skip_high=skip_high):
        g = sub_df[sub_df["layer"] == layer]
        if len(g) < 5:
            continue
        o, w, c = g["ws_obs"].values, g["ws_wrf"].values, g["ws_cfd"].values
        rows.append(
            {
                "Layer": layer,
                "N": len(g),
                "WRF_MBE": round(mean_bias_error(w, o), 3),
                "WRF_RMSE": round(rmse(w, o), 3),
                "WRF_IoA": round(index_of_agreement(w, o), 3),
                "CFD_MBE": round(mean_bias_error(c, o), 3),
                "CFD_RMSE": round(rmse(c, o), 3),
                "CFD_IoA": round(index_of_agreement(c, o), 3),
                "SS": round(skill_score(c, o, w), 3),
            }
        )
    return pd.DataFrame(rows)
