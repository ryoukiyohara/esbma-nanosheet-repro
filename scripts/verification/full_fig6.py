# Full recomputation of the 30 synthetic realisations (Fig 6) in the analysis image (scikit-learn 1.0.2).
import sys, json, time, numpy as np
sys.path.insert(0, "/work/src")
from joblib import Parallel, delayed
from esbma_repro import data as D, synth
X, y, c = D.load("yield", "/work/data/raw"); Z = synth.zscore(X)
S = json.load(open("/work/data/intermediate/fig6_synth_x18_r095_30reps.json"))
t = time.time()
out = Parallel(n_jobs=30)(delayed(synth.run_rep)(Z, 7, r, 0.95, 1.0, 50) for r in range(30))
print("time %.0f s" % (time.time() - t))
bf = np.array([o[0] for o in out]); pi = np.array([o[1] for o in out])
print("max |d esbma|", np.abs(pi - np.array(S["esbma_pin"])).max(), "max |d bolasso|", np.abs(bf - np.array(S["bolasso_freq"])).max(),
      "n reps with any bolasso diff", int((np.abs(bf - np.array(S["bolasso_freq"])).max(1) > 0).sum()))
fpb = sum((bf[r, [i for i in range(16) if i != 7]] >= 0.75).sum() for r in range(30)) / 30
fpe = sum((pi[r, [i for i in range(16) if i != 7]] >= 0.75).sum() for r in range(30)) / 30
print("FP/run bolasso %.2f esbma %.2f" % (fpb, fpe))
json.dump({"bolasso_freq": bf.tolist(), "esbma_pin": pi.tolist()}, open("/work/work/full_fig6_sklearn102.json", "w"))
