# FigS2 / Fig 5 convergence with the Fig 4 grid: montage_conv.py logic (SEED=1 subsets without replacement, N=10..80).
import sys, time, numpy as np
sys.path.insert(0, "/work/src")
from esbma_repro import data as D, esbma
X, y, c = D.load("yield", "/work/data/raw"); NL = [10, 20, 30, 40, 50, 60, 70, 80]; dist = []; me = []; mw = []
for N in NL:
    rng = np.random.default_rng(1)
    idx = np.arange(len(X)) if N >= len(X) else rng.choice(len(X), size=N, replace=False)
    t = time.time(); r = esbma.fit_exhaustive(X[idx], y[idx], -2.5, 2.5, 30, n_jobs=110)
    d = r["sigma_log_posterior"]; i, j = np.unravel_index(d.argmax(), d.shape)
    p = np.exp(d - d.max()); p /= p.sum(); pe, pw = p.sum(1), p.sum(0); le, lw = r["log10_sigma_noise"], r["log10_sigma_coef"]
    sd = lambda w, x: np.sqrt((w*(x-(w*x).sum())**2).sum())
    print("N=%d %.0fs MAP log_eps=%.2f log_w=%.2f sd_eps=%.3f sd_w=%.3f maxmass=%.2f" % (N, time.time()-t, le[i], lw[j], sd(pe, le), sd(pw, lw), p.max()), flush=True)
    dist.append(d.astype(np.float32)); me.append(10**le[i]); mw.append(10**lw[j])
np.savez_compressed("/work/work/grid30/fig5_post_real_g30.npz", N=np.array(NL), log_eps=le, log_w=lw, dist=np.stack(dist), map_eps=np.array(me), map_w=np.array(mw))
