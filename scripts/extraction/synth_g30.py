# Fig 6 with the unified Fig 4 setting: ES-BMA = exhbma, grid inverse(logspace(-2.5,2.5,30)), all 65,536 models
# (incl. single-variable and null models), exhbma standardisation; Bolasso = Fig 4 fixed version (100 boots, seed 43).
import sys, json, time, numpy as np
sys.path.insert(0, "/work/src")
from esbma_repro import data as D, synth, esbma, bolasso
X, y0, c = D.load("yield", "/work/data/raw"); Z = synth.zscore(X)
beta, r2 = 1.0, 0.95; sigma = beta * np.sqrt((1 - r2) / r2)
out = {"setting": "exhbma grid inverse(logspace(-2.5,2.5,30)); Bolasso fixed (guard zero), 100 boots, seed 43",
       "sigma": sigma, "names": c, "j_true": 7, "bolasso_rate": [], "esbma_fp": [], "bolasso_alpha": [], "y": []}
t0 = time.time()
for rep in range(30):
    rng = np.random.default_rng(20260831 + rep)          # same noise as the original synthetic study
    y = beta * Z[:, 7] + rng.normal(0.0, sigma, len(Z))
    r = esbma.fit_exhaustive(X, y, -2.5, 2.5, 30, n_jobs=110)
    rate, rows, a = bolasso.bolasso_rate(X, y, c, nb=100, seed=43, guard="zero")
    out["esbma_fp"].append(r["feature_posteriors"].tolist()); out["bolasso_rate"].append(rate.tolist())
    out["bolasso_alpha"].append(float(a)); out["y"].append(y.tolist())
    print(rep, "%.0fs" % (time.time() - t0), np.round(r["feature_posteriors"], 2).tolist(), rate.tolist(), flush=True)
json.dump(out, open("/work/work/grid30/fig6_synth_x18_r095_30reps_g30.json", "w"))
print("DONE")
