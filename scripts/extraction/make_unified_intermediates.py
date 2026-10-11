# Build the cached intermediates of the unified ES-BMA setting (grid inverse(logspace(-2.5,2.5,30))) from the fits
# made with src/esbma_repro (work/full_g30.npz = esbma.fit_exhaustive(X, y, -2.5, 2.5, 30) on the 80 yield samples).
import sys, numpy as np
r = np.load(sys.argv[1]); ll = r["log_likelihoods"]; o = np.argsort(ll)[::-1]; shown = np.r_[o[:100], o[59950:60050]]
np.savez_compressed(sys.argv[2], loglik=ll, shown_model_index=shown, shown_coef=r["coef"][shown], feature_posteriors=r["feature_posteriors"])
print("written", sys.argv[2])
