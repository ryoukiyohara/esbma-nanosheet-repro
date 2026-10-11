# Dump the mean_fp_{n}_y{1,2,3}.pickle files used by katsuta_analysis/analyze.py (Fig 9) into one CSV.
import pickle, glob, re, sys, pandas as pd
rows = []
for tag, pat in [("yield", "yield/mean_fp_*_y1.pickle"), ("size", "size/mean_fp_*_y2.pickle"),
                 ("size_distribution", "size_distribution/mean_fp_*_y3.pickle")]:
    for f in glob.glob("/app/user/katsuta/notebook/experiment3/" + pat):
        n = int(re.search(r"mean_fp_(\d+)_", f).group(1)); s = pickle.load(open(f, "rb"))
        r = dict(target=tag, n=n); r.update(dict(s)); rows.append(r)
pd.DataFrame(rows).sort_values(["target", "n"]).to_csv(sys.argv[1], index=False); print(len(rows))
