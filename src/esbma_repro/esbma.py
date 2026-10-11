"""ES-BMA: exhaustive search over all 2^p linear models + Bayesian model averaging.

All numerical work is done by the vendored `exhbma` 0.1.12 package (src/exhbma), exactly as in
the original analysis scripts:

    sx = StandardScaler(n_dim=2); sy = StandardScaler(n_dim=1, scaling=False)
    grid = inverse(np.logspace(lo, hi, n))            # p(sigma) ~ 1/sigma on a log grid
    m = ExhaustiveLinearRegression(sigma_noise_points=grid, sigma_coef_points=grid)
    m.fit(sx.transform(X), sy.transform(y))

`ExhaustiveLinearRegression.fit` is a serial loop over the 2^p indicator vectors (about 1 h for
p = 16 and a 31x31 grid). `fit_exhaustive` below runs the *same per-model call*
(`ExhaustiveLinearRegression._fit_over_sigma_noise_and_coef`) in parallel chunks and then uses
exhbma's own aggregation methods (`_calculate_log_marginal_likelihood`,
`_calculate_feature_posterior`, `_fixed_alpha_prior`, `_generate_indicator`). The only difference
is that the (sigma_noise, sigma_coef) log-likelihood surface is log-sum-exp'ed chunk-wise and then
across chunks, which is mathematically identical and agrees to floating-point rounding.
`fit_exhaustive(..., reference=True)` calls the untouched serial `ExhaustiveLinearRegression.fit`.

Notes on the model space (as implemented by exhbma 0.1.12 with its defaults):
  * exclude_null=False, i.e. the null (intercept-only) model is included -> 2^p models
    (65,536 for p = 16; the 65,535 non-empty models plus the constant model);
  * model prior: independent inclusion with alpha = 0.5 (uniform over models);
  * the feature posterior of x_j is the sum of the posterior probabilities of all models that
    contain x_j.
"""
from types import SimpleNamespace

import numpy as np
from scipy.special import logsumexp

from exhbma import ExhaustiveLinearRegression, StandardScaler, inverse
from exhbma.linear_regression import LinearRegression as _ExLR


def sigma_grid(lo, hi, n):
    """inverse(np.logspace(lo, hi, n)) -- the grid used for both sigma_noise and sigma_coef."""
    return inverse(np.logspace(lo, hi, n))


def standardize(X, y):
    """exhbma scalers: X -> mean 0 / std 1 (ddof=0); y -> centred only."""
    sx = StandardScaler(n_dim=2)
    sy = StandardScaler(n_dim=1, scaling=False)
    sx.fit(X)
    sy.fit(y)
    return sx.transform(X), sy.transform(y)


def _new_model(grid):
    return ExhaustiveLinearRegression(sigma_noise_points=grid, sigma_coef_points=grid)


def _fit_chunk(Xn, yn, grid, indicators):
    m = _new_model(grid)
    ll, lp, coefs, los = [], [], [], []
    for ind in indicators:
        l, l_over_sigma, c = m._fit_over_sigma_noise_and_coef(X=Xn[:, np.array(ind) == 1], y=yn)
        p = m._fixed_alpha_prior(indicator=ind)
        ll.append(l)
        lp.append(p)
        coefs.append(c)
        los.append(np.array(l_over_sigma) + p)
    return ll, lp, coefs, logsumexp(np.array(los), axis=0)


def fit_exhaustive(X, y, grid_lo, grid_hi, grid_n, n_jobs=-1, standardize_data=True,
                   reference=False, chunks=512):
    """Fit ES-BMA on (X, y). Returns a dict with

    indicators (2^p, p) int8 | log_likelihoods (2^p,) marginal log-likelihood of each model
    log_priors | coef (2^p, p) posterior-mean coefficients (0 where a feature is absent)
    log_likelihood (BMA evidence) | feature_posteriors (p,)
    log_likelihood_over_sigma (G, G) | sigma_log_posterior (G, G) as in montage_conv.py
    log10_sigma_noise, log10_sigma_coef (G,)
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    Xn, yn = standardize(X, y) if standardize_data else (X, y)
    grid = sigma_grid(grid_lo, grid_hi, grid_n)
    p = Xn.shape[1]
    m = _new_model(grid)
    if reference:
        m.fit(Xn, yn, verbose=False)
        inds = m.indicators_
        ll, lp = list(m.log_likelihoods_), list(m.log_priors_)
        coefs = [mi.coefficient for mi in m.models_]
        los = np.array(m.log_likelihood_over_sigma_)
        fpost = np.array(m.feature_posteriors_)
        lml = m.log_likelihood_
    else:
        _ExLR.validate_target_centralization(y=yn, tolerance=m._preprocessing_tolerance)
        _ExLR.validate_feature_standardization(X=Xn, tolerance=m._preprocessing_tolerance)
        m.n_features_in_ = p
        inds = m._generate_indicator(n_features=p)
        parts = np.array_split(np.arange(len(inds)), min(chunks, len(inds)))
        jobs = [[inds[i] for i in part] for part in parts if len(part)]
        if n_jobs == 1:
            res = [_fit_chunk(Xn, yn, grid, j) for j in jobs]
        else:
            from joblib import Parallel, delayed
            res = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_fit_chunk)(Xn, yn, grid, j) for j in jobs)
        ll = [v for r in res for v in r[0]]
        lp = [v for r in res for v in r[1]]
        coefs = [v for r in res for v in r[2]]
        los = logsumexp(np.array([r[3] for r in res]), axis=0)
        models = [SimpleNamespace(log_likelihood=v) for v in ll]
        lml = m._calculate_log_marginal_likelihood(log_priors=lp, models=models)
        fpost = np.array(m._calculate_feature_posterior(log_priors=lp, indicators=inds, models=models))
    ind_arr = np.array(inds, dtype=np.int8)
    coef = np.zeros(ind_arr.shape, float)
    for i, c in enumerate(coefs):
        coef[i, ind_arr[i] == 1] = c
    log_prior_grid = (np.log([g.prob for g in grid]).reshape(-1, 1)
                      + np.log([g.prob for g in grid]).reshape(1, -1))
    log10_s = np.log10(np.array([g.position for g in grid]))
    return dict(indicators=ind_arr, log_likelihoods=np.array(ll), log_priors=np.array(lp), coef=coef,
                log_likelihood=float(lml), feature_posteriors=fpost,
                log_likelihood_over_sigma=np.array(los),
                sigma_log_posterior=log_prior_grid + np.array(los) - lml,
                log10_sigma_noise=log10_s, log10_sigma_coef=log10_s)


def indicator_to_mask(indicators):
    """Row of 0/1 (x1..xp) -> integer bit mask with bit i = x_{i+1} (convention of run_dos_cve.py)."""
    ind = np.asarray(indicators)
    return (ind.astype(np.int64) << np.arange(ind.shape[1], dtype=np.int64)).sum(1)
