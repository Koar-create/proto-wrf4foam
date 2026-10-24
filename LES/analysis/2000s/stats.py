"""Stationarity tests, sampling uncertainty, MRD, nested-block variance, D(τ)."""
from __future__ import annotations

import numpy as np
from scipy import stats as spstats
from scipy.signal import lombscargle


# ---------------------------------------------------------------------------
# Time scales
# ---------------------------------------------------------------------------
def autocorr(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    x = x - np.nanmean(x)
    x = np.where(np.isfinite(x), x, 0.0)
    n = x.size
    if n < 4:
        return np.array([np.nan])
    nfft = 1 << (2 * n - 1).bit_length()
    fx = np.fft.rfft(x, n=nfft)
    ac = np.fft.irfft(fx * np.conj(fx))[:n]
    if ac[0] > 0:
        ac = ac / ac[0]
    return ac


def integral_time_scale(x: np.ndarray, dt: float) -> float:
    """Integrate autocorrelation to the first zero crossing (seconds)."""
    ac = autocorr(x)
    if ac.size < 3 or not np.isfinite(ac[0]):
        return float("nan")
    z = np.where(ac[1:] <= 0.0)[0]
    if z.size == 0:
        return float((ac.size - 1) * dt)
    i = int(z[0]) + 1
    return float(np.trapz(ac[: i + 1], dx=dt))


def efold_time_scale(x: np.ndarray, dt: float) -> float:
    ac = autocorr(x)
    if ac.size < 3 or not np.isfinite(ac[0]):
        return float("nan")
    idx = np.where(ac < np.exp(-1.0))[0]
    if idx.size == 0:
        return float((ac.size - 1) * dt)
    return float(idx[0] * dt)


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------
def block_series(t, x, t0, t1, block=50.0):
    t = np.asarray(t, float)
    x = np.asarray(x, float)
    edges = np.arange(t0, t1 + 0.5 * block, block)
    centres, means, sigs, ns = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (t > a) & (t <= b)
        n = int(sel.sum())
        if n < 5:
            continue
        xx = x[sel]
        centres.append(0.5 * (a + b))
        means.append(float(np.nanmean(xx)))
        sigs.append(float(np.nanstd(xx, ddof=1)))
        ns.append(n)
    return np.asarray(centres), np.asarray(means), np.asarray(sigs), np.asarray(ns)


# ---------------------------------------------------------------------------
# Reverse-arrangement, runs, Mann–Kendall
# ---------------------------------------------------------------------------
def reverse_arrangement_pvalue(x: np.ndarray) -> dict:
    """Bendat & Piersol reverse-arrangement test against a trend (two-sided)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 8:
        return dict(A=np.nan, z=np.nan, p=np.nan, n=n)
    # A = number of pairs i<j with x_i > x_j
    A = 0
    for i in range(n - 1):
        A += int(np.sum(x[i] > x[i + 1 :]))
    EA = n * (n - 1) / 4.0
    VA = n * (2 * n + 5) * (n - 1) / 72.0
    z = (A - EA) / np.sqrt(VA) if VA > 0 else 0.0
    p = 2.0 * spstats.norm.sf(abs(z))
    return dict(A=float(A), z=float(z), p=float(p), n=int(n))


def runs_test_pvalue(x: np.ndarray) -> dict:
    """Runs test about the median (Bendat & Piersol)."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 8:
        return dict(runs=np.nan, z=np.nan, p=np.nan, n=n)
    med = np.median(x)
    s = np.sign(x - med)
    s = s[s != 0]
    n1 = int(np.sum(s > 0))
    n2 = int(np.sum(s < 0))
    if n1 < 2 or n2 < 2:
        return dict(runs=np.nan, z=np.nan, p=np.nan, n=n)
    runs = 1 + int(np.sum(s[1:] != s[:-1]))
    N = n1 + n2
    ER = 1 + 2 * n1 * n2 / N
    VR = 2 * n1 * n2 * (2 * n1 * n2 - N) / (N ** 2 * (N - 1))
    z = (runs - ER) / np.sqrt(VR) if VR > 0 else 0.0
    p = 2.0 * spstats.norm.sf(abs(z))
    return dict(runs=float(runs), z=float(z), p=float(p), n=int(n))


def mann_kendall(x: np.ndarray) -> dict:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 8:
        return dict(tau=np.nan, slope=np.nan, p=np.nan, n=n)
    s = 0
    for i in range(n - 1):
        s += int(np.sum(np.sign(x[i + 1 :] - x[i])))
    # Sen slope: median of pairwise slopes
    slopes = []
    for i in range(n - 1):
        dt = np.arange(1, n - i)
        slopes.append((x[i + 1 :] - x[i]) / dt)
    slope = float(np.median(np.concatenate(slopes)))
    var_s = n * (n - 1) * (2 * n + 5) / 18.0
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0
    p = 2.0 * spstats.norm.sf(abs(z))
    tau = s / (0.5 * n * (n - 1))
    return dict(tau=float(tau), slope=slope, p=float(p), n=int(n), z=float(z))


# ---------------------------------------------------------------------------
# Sampling uncertainty
# ---------------------------------------------------------------------------
def batch_mean_sigma_ci(x: np.ndarray, dt: float, tau_int: float, alpha=0.05) -> dict:
    """95 % CI of σ via batch means; batch length ≥ 2 τ_int."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 20 or not np.isfinite(tau_int) or tau_int <= 0:
        return dict(sigma=float(np.nanstd(x, ddof=1)), half=np.nan, nbat=0, L=np.nan)
    L = max(int(np.ceil(2.0 * tau_int / dt)), 10)
    nbat = n // L
    if nbat < 3:
        L = max(n // 3, 10)
        nbat = n // L
    if nbat < 3:
        sig = float(np.nanstd(x, ddof=1))
        return dict(sigma=sig, half=np.nan, nbat=int(nbat), L=float(L * dt))
    batches = x[: nbat * L].reshape(nbat, L)
    sig_b = batches.std(axis=1, ddof=1)
    sig = float(np.mean(sig_b))
    se = float(sig_b.std(ddof=1) / np.sqrt(nbat))
    tcrit = float(spstats.t.ppf(1 - alpha / 2, nbat - 1))
    return dict(sigma=sig, half=tcrit * se, nbat=int(nbat), L=float(L * dt), se=se)


def ar1_sigma_rel_err(x: np.ndarray, dt: float) -> dict:
    """AR(1) relative error of the sample standard deviation."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    x = x - x.mean()
    if x.size < 20:
        return dict(rho=np.nan, rel=np.nan, n_eff=np.nan)
    rho = float(np.corrcoef(x[:-1], x[1:])[0, 1])
    rho = float(np.clip(rho, -0.99, 0.99))
    n_eff = x.size * (1 - rho) / (1 + rho) if abs(rho) < 0.999 else 1.0
    n_eff = max(n_eff, 1.0)
    rel = 1.0 / np.sqrt(2.0 * n_eff)
    return dict(rho=rho, rel=float(rel), n_eff=float(n_eff))


def lenschow_var_rel_err(tau_int: float, T: float) -> float:
    """Lenschow et al. (1994) relative random error of a second moment."""
    if not np.isfinite(tau_int) or tau_int <= 0 or T <= 0:
        return float("nan")
    return float(np.sqrt(2.0 * tau_int / T))


def ci_width_vs_T(x, t, tau_int, Ts=(100, 200, 300, 400, 600)):
    dt = float(np.median(np.diff(t)))
    rows = []
    t0 = float(t[0])
    for T in Ts:
        sel = (t >= t0) & (t < t0 + T)
        xx = x[sel]
        bm = batch_mean_sigma_ci(xx, dt, tau_int)
        rel = (bm["half"] / bm["sigma"]) if bm["sigma"] and np.isfinite(bm["half"]) else np.nan
        rows.append(dict(T=T, rel=rel, **{k: bm[k] for k in ("sigma", "half", "nbat", "L")}))
    return rows


# ---------------------------------------------------------------------------
# Haar MRD (Howell & Mahrt 1997)
# ---------------------------------------------------------------------------
def haar_mrd(x: np.ndarray, dt: float):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = 2 ** int(np.log2(x.size))
    if n < 8:
        return np.array([]), np.array([])
    r = x[:n] - x[:n].mean()
    scales, var = [], []
    m = int(np.log2(n))
    for lev in range(m, 0, -1):
        w = 2 ** lev
        means = r.reshape(-1, w).mean(1)
        var.append(float(np.mean(means ** 2)))
        scales.append(w * dt)
        r = r - np.repeat(means, w)
    return np.asarray(scales[::-1]), np.asarray(var[::-1])


# ---------------------------------------------------------------------------
# Nested-block variance (strictly additive; irregular samples OK)
# ---------------------------------------------------------------------------
def band_variance(t_s, x, t_lo=120.0, t_hi=600.0):
    """σ² = E[within <t_lo] + E_hi[var of t_lo-block means] + var of t_hi-block means."""
    t_s = np.asarray(t_s, float)
    x = np.asarray(x, float)
    ok = np.isfinite(t_s) & np.isfinite(x)
    t_s, x = t_s[ok], x[ok]
    if x.size < 10:
        return dict(lo=np.nan, mid=np.nan, hi=np.nan, tot=np.nan, n=int(x.size))
    t0 = t_s.min()
    b_lo = np.floor((t_s - t0) / t_lo).astype(int)
    b_hi = np.floor((t_s - t0) / t_hi).astype(int)
    mu = float(x.mean())
    tot = float(((x - mu) ** 2).mean())
    # group means
    m_lo = _group_mean(x, b_lo)
    m_hi = _group_mean(x, b_hi)
    lo = float(((x - m_lo) ** 2).mean())
    mid = float(((m_lo - m_hi) ** 2).mean())
    hi = float(((m_hi - mu) ** 2).mean())
    if abs(lo + mid + hi - tot) > 1e-8 * max(tot, 1e-12):
        raise AssertionError(f"band_variance not additive: {lo+mid+hi} vs {tot}")
    return dict(lo=lo, mid=mid, hi=hi, tot=tot, n=int(x.size))


def _group_mean(x, g):
    # transform group means back to sample length
    gx, ix = np.unique(g, return_inverse=True)
    sums = np.bincount(ix, weights=x)
    cnt = np.bincount(ix)
    means = sums / np.maximum(cnt, 1)
    return means[ix]


# ---------------------------------------------------------------------------
# Irregular-sample structure function
# ---------------------------------------------------------------------------
def structure_function(t_s, x, edges, min_n=30):
    t_s = np.asarray(t_s, float)
    x = np.asarray(x, float)
    ok = np.isfinite(t_s) & np.isfinite(x)
    t_s, x = t_s[ok], x[ok]
    n = t_s.size
    if n < min_n + 2:
        D = np.full(len(edges) - 1, np.nan)
        return D, np.zeros_like(D, dtype=int)
    i, j = np.triu_indices(n, k=1)
    lag = np.abs(t_s[j] - t_s[i])
    dx2 = (x[j] - x[i]) ** 2
    idx = np.digitize(lag, edges) - 1
    good = (idx >= 0) & (idx < len(edges) - 1)
    cnt = np.bincount(idx[good], minlength=len(edges) - 1)
    s = np.bincount(idx[good], weights=dx2[good], minlength=len(edges) - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        D = s / cnt
    D[cnt < min_n] = np.nan
    return D, cnt


def default_lag_edges(tmin=1.5, tmax=3600.0, n=30):
    return np.unique(np.round(np.logspace(np.log10(tmin), np.log10(tmax), n), 1))


# ---------------------------------------------------------------------------
# Lomb–Scargle
# ---------------------------------------------------------------------------
def lomb_scargle_spectrum(t_s, x, fmin=0.005, fmax=0.1, nfreq=200):
    t_s = np.asarray(t_s, float)
    x = np.asarray(x, float)
    ok = np.isfinite(t_s) & np.isfinite(x)
    t_s, x = t_s[ok], x[ok]
    x = x - x.mean()
    freq = np.linspace(fmin, fmax, nfreq)
    ang = 2 * np.pi * freq
    pxx = lombscargle(t_s - t_s[0], x, ang, normalize=True)
    return freq, pxx
