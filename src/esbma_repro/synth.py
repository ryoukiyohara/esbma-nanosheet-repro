"""Synthetic single-variable target (Fig 6) -- code of ~/esbma_synth/synth_solvent_one.py + bma_core.py.

Design (unchanged from the original):
  * descriptors: the 16 real descriptors of the 80 yield samples, z-scored (std==0 guard);
  * target: y = beta * z(x_true) + N(0, sigma^2), sigma = beta*sqrt((1-R^2)/R^2); x_true = x18
    (= paper x8, solvent HSP hydrogen-bonding term), beta = 1, R^2 = 0.95 -> sigma = 0.2294;
  * repetition r uses np.random.default_rng(20260831 + r) for the noise and for Bolasso resampling;
  * Bolasso: 50 bootstrap resamples, LassoCV(cv=5, 40 alphas, max_iter=20000, random_state=0),
    selection frequency = fraction with |coef| > 1e-10;
  * ES-BMA: all 65,535 non-empty subsets; per-subset Bayesian linear regression with
    (sigma_noise, sigma_coef) marginalised on a 30x30 log grid logspace(-2, 1.5, 30) under a 1/sigma
    prior, intercept marginalised analytically; uniform model prior (p = 0.5); y standardised.
    As in the original code, subsets with fewer than two (non-constant) features are skipped
    (`if len(feat) < 2: continue` in bma_core.bma), so the 16 single-feature models carry no weight.
"""
import inspect
import itertools

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LassoCV
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ---------------------------------------------------------------- bma_core.py (verbatim maths)
GRID = 30
_LOG_S = np.logspace(-2, 1.5, GRID)
_SE = np.repeat(_LOG_S, GRID)
_SW = np.tile(_LOG_S, GRID)
_SE2 = _SE ** 2
_LAM = (_SE / _SW) ** 2


def fit_predict(X, y, Xs, sigma_y=1.0):
    n, p = X.shape
    XtX = X.T @ X
    d, V = np.linalg.eigh(XtX)
    d = np.maximum(d, 1e-12)
    b = V.T @ (X.T @ y)
    Z = Xs @ V
    yy = float(y @ y)
    den = d[None, :] + _LAM[:, None]
    quad = (yy - (b[None, :] ** 2 / den).sum(1)) / _SE2
    logdet = (n - p) * np.log(_SE2) + np.log(_SE2[:, None] + (_SW ** 2)[:, None] * d[None, :]).sum(1)
    logp = -0.5 * (n * np.log(2 * np.pi) + logdet + quad)
    logp = logp + 0.5 * (np.log(_SE2) - np.log(n * sigma_y ** 2 + _SE2))
    mx = logp.max()
    wg = np.exp(logp - mx)
    Sw = wg.sum()
    log_ev = mx + np.log(Sw) - np.log(len(logp))
    wg = wg / Sw
    mu = (b[None, :] / den) @ Z.T
    var = _SE2[:, None] + (_SE2[:, None] / den) @ (Z ** 2).T
    mean = wg @ mu
    second = wg @ (var + mu ** 2)
    return mean, np.maximum(second - mean ** 2, 1e-12), float(log_ev)


def bma(combos, M, obs, y, pool_idx, prior_p):
    active = sorted({j for c in combos for j in c})
    freq = {j: min(0.95, max(0.05, float(prior_p[j]))) for j in active}
    Xo, yo, Xp = M[obs], y[obs], M[pool_idx]
    ym, ys = yo.mean(), (yo.std() or 1.0)
    yc = (yo - ym) / ys
    logw, used = [], []
    for c in combos:
        feat = [j for j in c if Xo[:, j].std() > 1e-9]
        if len(feat) < 2:
            continue
        mu_, sd_ = Xo[:, feat].mean(0), Xo[:, feat].std(0)
        sd_[sd_ < 1e-12] = 1.0
        Xs = (Xo[:, feat] - mu_) / sd_
        Xc = (Xp[:, feat] - mu_) / sd_
        _, _, ev = fit_predict(Xs, yc, Xc)
        lp = sum(np.log(freq[j]) for j in feat) + sum(np.log(1 - freq[j]) for j in active if j not in feat)
        logw.append(lp + ev)
        used.append(feat)
    lw = np.array(logw)
    w = np.exp(lw - lw.max())
    return w / w.sum(), used


# ---------------------------------------------------------------- synth_solvent_one.py
def _lasso_kwargs():
    names = inspect.signature(LassoCV.__init__).parameters
    return {"n_alphas": 40} if "n_alphas" in names else {"alphas": 40}


def bolasso_freq(X, y, boot, rng):
    n, p = X.shape
    cnt = np.zeros(p)
    for _ in range(boot):
        idx = rng.integers(0, n, n)
        m = LassoCV(cv=5, max_iter=20000, random_state=0, **_lasso_kwargs()).fit(X[idx], y[idx])
        cnt[np.abs(m.coef_) > 1e-10] += 1
    return cnt / boot


def esbma_pin(X, y, combos):
    p = X.shape[1]
    obs = np.arange(len(y))
    w, used = bma(combos, X, obs, y, obs, prior_p=np.full(p, 0.5))
    pin = np.zeros(p)
    for wt, cols in zip(np.asarray(w, float), used):
        for j in cols:
            pin[j] += wt
    return pin


def zscore(X):
    return (X - X.mean(0)) / np.where(X.std(0) == 0, 1.0, X.std(0))


def run_rep(Z, jt, rep, r2=0.95, beta=1.0, boot=50, max_size=16, do_bolasso=True):
    """One noise realisation. Returns (bolasso_freq[p], esbma_pin[p], sigma)."""
    p = Z.shape[1]
    sigma = beta * np.sqrt((1 - r2) / r2)
    combos = [list(c) for k in range(1, max_size + 1) for c in itertools.combinations(range(p), k)]
    rng = np.random.default_rng(20260831 + rep)
    y = beta * Z[:, jt] + rng.normal(0.0, sigma, len(Z))
    bf = bolasso_freq(Z, y, boot, rng) if do_bolasso else np.full(p, np.nan)
    pi = esbma_pin(Z, y, combos)
    return bf, pi, sigma


# ---------------------------------------------------------------- unified setting (used in the paper)
def run_rep_unified(X, names, rep, r2=0.95, beta=1.0, j_true=7, n_jobs=-1, nb=100, use_exhbma=False):
    """One noise realisation analysed with the Fig 4 setting.

    Target: identical to run_rep (y = beta*z(x_true) + N(0, sigma^2), rng default_rng(20260831 + rep)).
    ES-BMA: exhbma model (fast_evidence.fit_fast by default, identical evidences; use_exhbma=True for the
    original per-model exhbma loop), sigma grid inverse(logspace(-2.5, 2.5, 30)), all 2^16 models (incl. single-variable
    and null models), exhbma standardisation (X z-scored, y centred).
    Bolasso: Fig 4 fixed version (alpha by 5-fold CV, 100 bootstraps, seed 43, std==0 guard).
    Returns (bolasso_rate[p], esbma_posterior[p], y)."""
    from .esbma import fit_exhaustive
    from .fast_evidence import fit_fast
    from .bolasso import bolasso_rate
    Z = zscore(X)
    sigma = beta * np.sqrt((1 - r2) / r2)
    rng = np.random.default_rng(20260831 + rep)
    y = beta * Z[:, j_true] + rng.normal(0.0, sigma, len(Z))
    r = fit_exhaustive(X, y, -2.5, 2.5, 30, n_jobs=n_jobs) if use_exhbma else fit_fast(X, y, -2.5, 2.5, 30)
    rate, _, _ = bolasso_rate(X, y, names, nb=nb, seed=43, guard="zero")
    return rate, r["feature_posteriors"], y
