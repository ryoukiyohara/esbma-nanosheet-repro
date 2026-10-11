import sys, time, numpy as np
sys.path.insert(0, "/work/src")
from esbma_repro import data, esbma
X, y, names = data.load("yield", "/work/data/raw")
for cols in [list(range(8)), [1, 7, 12, 13, 15, 3, 9]]:
    Xs = X[:, cols]
    t=time.time(); a = esbma.fit_exhaustive(Xs, y, -2, 2, 31, n_jobs=8, chunks=16); t1=time.time()-t
    t=time.time(); b = esbma.fit_exhaustive(Xs, y, -2, 2, 31, reference=True); t2=time.time()-t
    for k in ["log_likelihoods", "log_priors", "coef", "feature_posteriors", "log_likelihood_over_sigma", "indicators"]:
        print(k, float(np.max(np.abs(a[k] - b[k]))))
    print("lml", a["log_likelihood"] - b["log_likelihood"], "time par/ref", round(t1,1), round(t2,1))
