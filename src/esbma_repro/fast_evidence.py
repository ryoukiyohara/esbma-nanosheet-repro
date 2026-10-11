"""Fast, vectorised evaluation of the exhbma 0.1.12 model evidences (same numbers, ~100x faster).

Where exhbma spends its time (per model S, in ExhaustiveLinearRegression.fit):
  * MarginalLinearRegression._fit_models_over_sigma: one full SVD of X_S (n x n U matrix, full_matrices=True), then
    a Python double loop over the 30 x 30 = 900 (sigma_noise, sigma_coef) grid points, creating 900 LinearRegression
    objects; each one computes the posterior-mean coefficient (a k x k matrix-vector product) and the log likelihood
    (vectors of length n) with ~15 small numpy calls -> ~14,000 tiny numpy calls per model;
  * the 900 log-likelihoods are integrated over the grid (integrate_log_values_in_square: weights, logsumexp), and the
    same integration is repeated once per coefficient (k more times);
  * a pydantic ModelInfo object holding the 900 log-likelihoods is built per model.
  The arithmetic is tiny; the cost is Python/numpy call overhead (≈ 60 ms per model, 65,536 models ≈ 65 CPU-min).

What this module does instead (identical formulas, exhbma conventions):
  * models are grouped by size k; for a batch of models the thin SVDs of all X_S are computed in one batched LAPACK
    call (np.linalg.svd on an (M, n, k) array) - singular values s and U^T y are all that the evidence needs;
  * the log likelihood on all 900 grid points is evaluated by broadcasting, exactly as in
    exhbma.linear_regression.LinearRegression._calculate_log_likelihood:
        log p = -n/2 log 2pi - 1/2 sum_i log(sn^2 + sc^2 s_i^2) - (n - k)/2 log sn^2
                - 1/2 [ sum_i (U^T y)_i^2 / (sn^2 + sc^2 s_i^2) + (y^T y - sum_i (U^T y)_i^2) / sn^2 ]
                + 1/2 [ log sn^2 - log(n var(y) + sn^2) ]
    (the n - k directions orthogonal to X_S have eigenvalue 0, as in exhbma's full SVD); k = 0 gives
    exhbma.constant_regression.ConstantRegression (the intercept-only model);
  * the grid integration uses the same weights as integrate_log_values_in_square (1/sigma prior densities
    p.prob, trapezoid-like cell areas, factor 1/4) and logsumexp;
  * model priors, the BMA evidence and the feature posteriors use the same expressions as
    ExhaustiveLinearRegression (_fixed_alpha_prior with alpha = 0.5, logsumexp with indicator weights).
  Posterior-mean coefficients are not computed (not needed for feature posteriors).
"""
import numpy as np
from scipy.special import logsumexp

from exhbma import inverse


def grid_log_weights(grid):
    """log of (prior density x integration cell area / 4) on the (sigma_noise, sigma_coef) grid, as in exhbma."""
    x = [p.position for p in grid]
    logprior = np.log([p.prob for p in grid]).reshape(-1, 1) + np.log([p.prob for p in grid]).reshape(1, -1)
    ext = np.array([x[0]] + x + [x[-1]])
    area = np.zeros((len(x), len(x)))
    for i in range(2):
        for j in range(2):
            area += np.outer(ext[i + 1:i + 1 + len(x)] - ext[i:i + len(x)], ext[j + 1:j + 1 + len(x)] - ext[j:j + len(x)])
    return logprior - np.log(4) + np.log(area)


def _ll_grid(s2, uty2, n, k, yy, sn2, sc2, log_int):
    """log p(y | X_S, sn, sc) for a batch: s2, uty2 (M, k); sn2, sc2, log_int (G,). Returns (M, G)."""
    if k == 0:
        cov_logdet = np.zeros((s2.shape[0], sn2.size)); quad = np.zeros_like(cov_logdet); proj = np.zeros((s2.shape[0], 1))
    else:
        cov = sn2[None, :, None] + sc2[None, :, None] * s2[:, None, :]          # (M, G, k)
        cov_logdet = np.log(cov).sum(-1)
        quad = (uty2[:, None, :] / cov).sum(-1)
        proj = uty2.sum(-1, keepdims=True)
    return (-n / 2 * np.log(2 * np.pi) - 0.5 * (cov_logdet + (n - k) * np.log(sn2)[None, :])
            - 0.5 * (quad + (yy - proj) / sn2[None, :]) + log_int[None, :])


def model_log_evidence(Xn, yn, grid, batch=256, return_grid=False, sigma_surface=False, alpha=0.5):
    """Marginal log-likelihood of every model in exhbma order (model index i <-> bit j of i = feature j).
    Xn standardised, yn centred (exhbma scalers). Returns ll (2^p,) [and the (2^p, G1, G2) log-likelihood surface]
    [and, with sigma_surface=True, exhbma's log_likelihood_over_sigma_: log sum_models p(model) p(y|model, sn, sc)]."""
    n, p = Xn.shape
    sig = np.array([g.position for g in grid])
    SN2 = np.repeat(sig ** 2, len(sig)); SC2 = np.tile(sig ** 2, len(sig))      # flattened (noise, coef) grid
    W = grid_log_weights(grid).ravel()
    yy = float(yn @ yn); var_y = np.var(yn)
    log_int = (np.log(SN2) - np.log(n * var_y + SN2)) / 2
    masks = np.arange(2 ** p)
    size = np.array([bin(m).count("1") for m in masks])
    ll = np.empty(2 ** p)
    surf = np.empty((2 ** p, len(sig), len(sig))) if return_grid else None
    bits = (masks[:, None] >> np.arange(p)) & 1
    los = np.full(SN2.size, -np.inf)
    for k in range(p + 1):
        log_prior_k = k * np.log(alpha) + (p - k) * np.log(1 - alpha)
        idx = np.where(size == k)[0]
        cols_all = np.array([np.where(bits[i])[0] for i in idx]).reshape(len(idx), k)
        for a in range(0, len(idx), batch):
            ii, cols = idx[a:a + batch], cols_all[a:a + batch]
            if k == 0:
                s2 = np.zeros((len(ii), 0)); uty2 = s2
            else:
                Xb = np.transpose(Xn[:, cols], (1, 0, 2))                         # (M, n, k)
                u, s, _ = np.linalg.svd(Xb, full_matrices=False)
                s2 = s ** 2; uty2 = np.einsum("mnk,n->mk", u, yn) ** 2
            L = _ll_grid(s2, uty2, n, k, yy, SN2, SC2, log_int)                  # (M, G)
            ll[ii] = logsumexp(L + W[None, :], axis=1)
            if sigma_surface:
                los = np.logaddexp(los, logsumexp(L, axis=0) + log_prior_k)
            if return_grid:
                surf[ii] = L.reshape(len(ii), len(sig), len(sig))
    out = (ll,) + ((surf,) if return_grid else ()) + ((los.reshape(len(sig), len(sig)),) if sigma_surface else ())
    return out if len(out) > 1 else ll


def feature_posteriors(ll, p, alpha=0.5):
    """BMA feature posteriors with exhbma's model prior (alpha = 0.5, null model included)."""
    masks = np.arange(2 ** p)
    bits = (masks[:, None] >> np.arange(p)) & 1
    k = bits.sum(1)
    log_prior = k * np.log(alpha) + (p - k) * np.log(1 - alpha)
    lj = log_prior + ll
    lml = logsumexp(lj)
    return np.exp([logsumexp(lj, b=bits[:, j]) - lml for j in range(p)]), float(lml)


def fit_fast(X, y, grid_lo=-2.5, grid_hi=2.5, grid_n=30, standardize_data=True):
    """Drop-in for esbma.fit_exhaustive when only evidences / feature posteriors are needed."""
    from .esbma import standardize
    X = np.asarray(X, float); y = np.asarray(y, float)
    Xn, yn = standardize(X, y) if standardize_data else (X, y)
    ll = model_log_evidence(Xn, yn, inverse(np.logspace(grid_lo, grid_hi, grid_n)))
    fp, lml = feature_posteriors(ll, X.shape[1])
    return dict(log_likelihoods=ll, feature_posteriors=fp, log_likelihood=lml)


def fit_fast_full(X, y, grid_lo=-2.5, grid_hi=2.5, grid_n=30, coef_models=None, standardize_data=True):
    """Fast replacement of esbma.fit_exhaustive for the full-data analyses (Fig 3, 4, 5, 7).

    Evidences, feature posteriors, BMA evidence and the (sigma_noise, sigma_coef) posterior come from the vectorised
    evaluator; posterior-mean coefficients are computed with exhbma itself (the per-model call used by
    ExhaustiveLinearRegression.fit) only for the model indices in coef_models (e.g. the 200 models shown in Fig 3)."""
    from .esbma import standardize, _new_model
    X = np.asarray(X, float); y = np.asarray(y, float); p = X.shape[1]
    Xn, yn = standardize(X, y) if standardize_data else (X, y)
    grid = inverse(np.logspace(grid_lo, grid_hi, grid_n))
    ll, los = model_log_evidence(Xn, yn, grid, sigma_surface=True)
    fp, lml = feature_posteriors(ll, p)
    log_prior_grid = np.log([g.prob for g in grid]).reshape(-1, 1) + np.log([g.prob for g in grid]).reshape(1, -1)
    log10_s = np.log10(np.array([g.position for g in grid]))
    ind = ((np.arange(2 ** p)[:, None] >> np.arange(p)) & 1).astype(np.int8)
    coef = coefficients(X, y, coef_models, grid_lo, grid_hi, grid_n, standardize_data) if coef_models is not None else {}
    return dict(indicators=ind, log_likelihoods=ll, feature_posteriors=fp, log_likelihood=lml,
                log_likelihood_over_sigma=los, sigma_log_posterior=log_prior_grid + los - lml,
                log10_sigma_noise=log10_s, log10_sigma_coef=log10_s, coef_models=coef)


def coefficients(X, y, model_indices, grid_lo=-2.5, grid_hi=2.5, grid_n=30, standardize_data=True):
    """Posterior-mean coefficients (marginalised over the sigma grid) of selected models, computed by exhbma itself.
    Returns {model index: coefficient vector of length p (0 for absent features)}."""
    from .esbma import standardize, _new_model
    X = np.asarray(X, float); y = np.asarray(y, float); p = X.shape[1]
    Xn, yn = standardize(X, y) if standardize_data else (X, y)
    m = _new_model(inverse(np.logspace(grid_lo, grid_hi, grid_n)))
    out = {}
    for i in model_indices:
        sel = ((int(i) >> np.arange(p)) & 1) == 1
        c = np.zeros(p)
        if sel.any():
            c[sel] = m._fit_over_sigma_noise_and_coef(X=Xn[:, sel], y=yn)[2]
        out[int(i)] = c
    return out
