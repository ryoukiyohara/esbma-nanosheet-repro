"""Bolasso (bootstrap Lasso) on the real yield data -- code of ~/esbma_bolasso_fix/run_fix.py.

(1) `bolasso_rate`: Lasso penalty alpha chosen by 5-fold CV (MSE, KFold(5, shuffle, seed 43)) on a
    grid logspace(-5, 5, 100); then 100 bootstrap resamples (RandomState(43) seeds), Lasso refit on
    each; selection rate = fraction of bootstraps with |coef| > 1e-12.
(2) `bolasso_cveopt`: alpha = the one whose full-data Lasso support minimises the DoS-CVE
    (5-fold OLS RMSE, same folds as run_dos_cve.py); then the same 100 bootstraps.

Standardisation: X -> (X - mean)/std with std computed on the fitting set; y centred.
guard="zero" (fixed version, used in the paper): only std == 0 columns are left unscaled.
guard="1e-12" (original redo_fp_main.py / run_bolasso.py): std < 1e-12 left unscaled. With the real
data this silently disabled scaling of x14 (raw values ~1e-39), so x14 was never selected.
"""
import numpy as np
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import KFold
from sklearn.utils import resample

ALPHAS = np.logspace(-5, 5, 100)


def std_x(a, ref, guard="zero"):
    s = ref.std(axis=0)
    s = np.where(s == 0, 1.0, s) if guard == "zero" else np.where(s < 1e-12, 1.0, s)
    return (a - ref.mean(axis=0)) / s


def cen_y(a, ref):
    return a - ref.mean()


def _bootstrap(X, y, alpha, nb, seed, guard, names):
    rng = np.random.RandomState(seed)
    cnt = np.zeros(X.shape[1])
    rows = []
    for b in range(nb):
        Xr, yr = resample(X, y, replace=True, n_samples=len(X), random_state=int(rng.randint(0, 2**31 - 1)))
        c = Lasso(alpha=alpha, max_iter=10000).fit(std_x(Xr, Xr, guard), cen_y(yr, yr)).coef_
        sel = [i for i in range(X.shape[1]) if abs(c[i]) > 1e-12]
        cnt[sel] += 1
        rows.append(dict(bootstrap=b, mask=int(sum(1 << i for i in sel)), size=len(sel),
                         features="|".join(names[i] for i in sel)))
    return cnt / nb, rows


def bolasso_rate(X, y, names, nb=100, seed=43, guard="zero", alphas=ALPHAS):
    """Returns (selection_rate[p], supports(list of dict), alpha)."""
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    best = (1e18, None)
    for al in alphas:
        r = np.mean([np.mean((cen_y(y[va], y[tr]) - Lasso(alpha=al, max_iter=10000)
                              .fit(std_x(X[tr], X[tr], guard), cen_y(y[tr], y[tr]))
                              .predict(std_x(X[va], X[tr], guard))) ** 2)
                     for tr, va in kf.split(X)])
        if r < best[0]:
            best = (r, al)
    rate, rows = _bootstrap(X, y, best[1], nb, seed, guard, names)
    return rate, rows, best[1]


def dos_folds(X, y, n_splits=5, seed=43):
    folds = []
    for tr, va in KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X):
        m = X[tr].mean(0); s = X[tr].std(0); s = np.where(s == 0, 1.0, s); ym = y[tr].mean()
        folds.append(((X[tr] - m) / s, (X[va] - m) / s, y[tr] - ym, y[va] - ym))
    return folds


def cve_cols(cols, folds):
    if not cols:
        return np.inf
    return float(np.mean([np.sqrt(np.mean((d - LinearRegression(fit_intercept=False).fit(a[:, cols], c)
                                            .predict(b[:, cols])) ** 2)) for a, b, c, d in folds]))


def bolasso_cveopt(X, y, names, nb=100, seed=43, guard="zero", alphas=ALPHAS):
    folds = dos_folds(X, y, seed=seed)
    Xs = std_x(X, X, guard); yc = cen_y(y, y); bc = (np.inf, None)
    for al in alphas:
        cols = [i for i, c in enumerate(Lasso(alpha=al, max_iter=10000).fit(Xs, yc).coef_) if abs(c) > 1e-12]
        v = cve_cols(cols, folds)
        if v < bc[0]:
            bc = (v, al)
    rate, rows = _bootstrap(X, y, bc[1], nb, seed, guard, names)
    return rate, rows, bc[1]
