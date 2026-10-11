# Collect per-trial feature posteriors of the subsampling ("bootstrap ES-BMA") runs into one long CSV.
# Each trial dir holds fp.pickle (list of posteriors), feature_columns.pickle, sample_indexs.pickle.
import os, sys, pickle, glob, numpy as np, pandas as pd
rows = []
for tag, root in [("yield_exp31", "/app/user/namiuchi/output/exp31"),
                  ("yield_katsuta", "/app/user/katsuta/notebook/experiment3/output/yield"),
                  ("size", "/app/user/katsuta/notebook/experiment3/output/size"),
                  ("size_distribution", "/app/user/katsuta/notebook/experiment3/output/size-distribution")]:
    for d in sorted(glob.glob(root + "/*/trial_*")):
        n = int(d.split("/")[-2]); t = int(d.split("_")[-1])
        L = lambda k: pickle.load(open(os.path.join(d, k + ".pickle"), "rb"))
        fp = np.asarray(L("fp"), float).ravel(); cols = list(L("feature_columns")); idx = list(L("sample_indexs"))
        r = dict(target=tag, n_dir=n, trial=t, n_actual=len(idx), sample_indexs=" ".join(map(str, idx)))
        r.update({c: v for c, v in zip(cols, fp)}); rows.append(r)
df = pd.DataFrame(rows); df.to_csv(sys.argv[1], index=False)
print(df.groupby("target").agg(n=("trial", "size"), nmin=("n_dir", "min"), nmax=("n_dir", "max")))
print(df[df.n_dir != df.n_actual][["target", "n_dir", "n_actual"]].drop_duplicates())
