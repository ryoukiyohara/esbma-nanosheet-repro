# Extract compact Fig 3 data from the original exhbma model pickle (output/exp1/run2/model.pickle, 265 MB).
import pickle, sys, types, numpy as np, pandas as pd, pickletools
SRC = sys.argv[1]; OUT = sys.argv[2]
class _Any:  # stand-in for the wrapper class defined in the original __main__
    def __setstate__(self, s): self.__dict__.update(s)
class U(pickle.Unpickler):
    def find_class(self, mod, name):
        if mod == "__main__": return _Any
        return super().find_class(mod, name)
w = U(open(SRC, "rb")).load()
print(type(w), list(w.__dict__.keys()))
m = w.model
ind = np.array(m.indicators_, dtype=np.int8); ll = np.array(m.log_likelihoods_, float)
coef = np.zeros(ind.shape, float)
for i, mi in enumerate(m.models_):
    coef[i, ind[i] == 1] = mi.coefficient
print(ind.shape, ll[:3], m.feature_posteriors_)
print("noise grid", [p.position for p in m.sigma_noise_points][:3], len(m.sigma_noise_points))
np.savez_compressed(OUT, indicators=ind, loglik=ll, coef=coef.astype(np.float64),
                    feature_posteriors=np.array(m.feature_posteriors_))
