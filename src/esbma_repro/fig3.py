"""Fig 3 helpers -- code of notebook/experiment5/compare_model.py and calc_weightdiagram_*.py.

Models are ranked by their marginal log-likelihood (= minus the Bayesian free energy, BFE) from the
ES-BMA fit of the yield data with grid inverse(logspace(-2.5, 2.5, 20)) (output/exp1/run2/model.pickle).

Single-fold prediction plots (used in an earlier version of Fig 3; kept for reference, `fold_prediction`): for the selected feature subset, X standardised (exhbma scaler, full data) and y
centred; OLS with intercept (sklearn LinearRegression) trained on folds of
KFold(n_splits=5, shuffle=True, random_state=42). compare_model.py loops over the folds and stops
before the 4th (`end_idx = 4`), so the scatter plot and the RMSE values quoted in the paper are those
of the 3rd fold (index 2).
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import KFold

from exhbma import StandardScaler


def fold_prediction(X, y, selected, fold=2, seed=42):
    sx = StandardScaler(n_dim=2); sy = StandardScaler(n_dim=1, scaling=False)
    Xs = X[:, selected]
    sx.fit(Xs); sy.fit(y)
    Xn, yn = sx.transform(Xs), sy.transform(y)
    tr, te = list(KFold(n_splits=5, shuffle=True, random_state=seed).split(X))[fold]
    m = LinearRegression().fit(Xn[tr], yn[tr])
    p_tr, p_te = sy.restore(m.predict(Xn[tr])), sy.restore(m.predict(Xn[te]))
    return dict(y_train=y[tr], pred_train=p_tr, y_test=y[te], pred_test=p_te,
                rmse_train=float(np.sqrt(mse(y[tr], p_tr))), rmse_test=float(np.sqrt(mse(y[te], p_te))))


def cv_prediction(X, y, selected, n_splits=5, seed=43):
    """Out-of-fold (cross-validated) predictions used for Fig 3 (c)-(e), with exactly the DoS protocol.

    The folds and the train-fold standardisation come from dos.build_folds (KFold(5, shuffle=True, random_state=43);
    X standardised and y centred with train-fold statistics); OLS without intercept on the selected columns
    (dos.cve_for_mask). Returns (prediction for every sample in original units, CVE = mean of the 5 fold RMSEs);
    the CVE is therefore identical to the DoS CVE of the same subset."""
    from sklearn.linear_model import LinearRegression as OLS
    from sklearn.model_selection import KFold
    from .dos import build_folds
    X = np.asarray(X, float); y = np.asarray(y, float); cols = np.where(np.asarray(selected, bool))[0]
    folds = build_folds(X, y, n_splits=n_splits, random_state=seed)
    split = list(KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X))
    pred = np.empty_like(y); errs = []
    for (Xtr, Xva, ytr, yva), (tr, va) in zip(folds, split):
        p = OLS(fit_intercept=False).fit(Xtr[:, cols], ytr).predict(Xva[:, cols])
        errs.append(np.sqrt(np.mean((yva - p) ** 2)))
        pred[va] = p + y[tr].mean()
    return pred, float(np.mean(errs))
