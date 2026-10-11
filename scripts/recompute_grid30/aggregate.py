"""Collect the per-trial JSON files of the Fig 8 / Fig 9 recomputation, build the intermediate CSVs in the format
used by the notebook, and redraw Fig 8 (make_fig8.make_figure) and Fig 9.

    python scripts/recompute_grid30/aggregate.py [--outdir results/recompute_grid30] [--install]

--install copies the two CSVs to data/intermediate/ (fig8_fig9_subsampling_trials.csv, fig9_mean_fp_pickles.csv),
replacing the legacy 20-point-grid versions, so that the notebook shows the recomputed results. Only use it when all
trials are complete (the script reports missing trials and refuses --install otherwise).

Fig 9 Max / Min descriptor = highest / lowest ES-BMA posterior on the FULL data of each target (no subsampling, unified
grid; esbma_repro.fig9.select_maxmin), as defined in the paper; the selection is asserted to be yield x32/x36,
size x16/x30, size distribution x32/x41.
"""
import argparse, glob, json, os, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np, pandas as pd                      # noqa: E402
import matplotlib; matplotlib.use("Agg")              # noqa: E402
import matplotlib.pyplot as plt                       # noqa: E402

STUDIES = {"fig8_exp31_yield": ("yield", range(30, 85, 5)), "fig9_size": ("size", range(30, 125, 5)),
           "fig9_size_distribution": ("size_distribution", range(30, 95, 5))}
TARGET_ROWS = {"fig8_exp31_yield": ["yield_exp31", "yield_katsuta"], "fig9_size": ["size"],
               "fig9_size_distribution": ["size_distribution"]}

ap = argparse.ArgumentParser(); ap.add_argument("--outdir", default=os.path.join(ROOT, "results", "recompute_grid30"))
ap.add_argument("--install", action="store_true")
a = ap.parse_args()
rows, complete = [], True
for study, (target, ns) in STUDIES.items():
    files = sorted(glob.glob(os.path.join(a.outdir, study, "n*_trial*.json")))
    recs = [json.load(open(f)) for f in files]
    have = {(r["n"], r["trial"]) for r in recs}
    missing = [(n, t) for n in ns for t in range(30) if (n, t) not in have]
    print(f"{study}: {len(recs)} trial files, {len(missing)} of {len(ns) * 30} missing")
    complete &= not missing
    for r in recs:
        for tg in TARGET_ROWS[study]:
            d = dict(target=tg, n_dir=r["n"], trial=r["trial"], n_actual=len(r["sample_indexs"]),
                     sample_indexs=" ".join(map(str, r["sample_indexs"])))
            d.update(dict(zip(r["feature_columns"], r["fp"]))); rows.append(d)
if not rows:
    sys.exit("no trial files found in " + a.outdir)
T = pd.DataFrame(rows).sort_values(["target", "n_dir", "trial"])
T.to_csv(os.path.join(a.outdir, "fig8_fig9_subsampling_trials.csv"), index=False)
M = []
for tg, src in [("yield", "yield_katsuta"), ("size", "size"), ("size_distribution", "size_distribution")]:
    S = T[(T.target == src) & (T.trial < 30)]
    if len(S):
        m = S.groupby("n_dir")[[c for c in S.columns if c.startswith("x") and S[c].notna().any()]].mean()
        m.insert(0, "target", tg); M.append(m.reset_index().rename(columns={"n_dir": "n"}))
M = pd.concat(M); M.to_csv(os.path.join(a.outdir, "fig9_mean_fp_pickles.csv"), index=False)

# Fig 8: (a)-(c) n = 30, 45, 60; (d), (e) x9, x13 (paper numbering) -- needs all 11 x 30 yield trials
from esbma_repro import data as Dd                    # noqa: E402
Y = T[T.target == "yield_exp31"]
if sorted(Y.n_dir.unique()) == list(range(30, 85, 5)) and (Y.groupby("n_dir").size() == 30).all():
    from esbma_repro import make_fig8
    data8 = {n: Y[Y.n_dir == n].sort_values("trial")[Dd.YIELD_COLS].to_numpy() for n in range(30, 85, 5)}
    fig = make_fig8.make_figure(data8); fig.savefig(os.path.join(a.outdir, "Fig8_sample_size.png"), dpi=300, facecolor="white")
    plt.close(fig); print("Fig 8 written")
else:
    print("Fig 8 skipped (yield trials incomplete)")
plt.rcdefaults()

# Fig 9: Max / Min descriptor = highest / lowest ES-BMA posterior on the full data (no subsampling), unified grid
from esbma_repro import fig9                          # noqa: E402
MAXMIN = fig9.select_maxmin(os.path.join(ROOT, "data", "raw"))
means = {}
for tg in fig9.TARGETS:
    m = M[M.target == tg].set_index("n").drop(columns="target").dropna(axis=1, how="all")
    cmax, cmin, fpf = MAXMIN[tg]
    assert (cmax, cmin) == fig9.EXPECTED_MAXMIN[tg], (tg, cmax, cmin)
    if not len(m):
        continue
    means[tg] = m[[cmax, cmin]]
    print(f"{tg}: Max {cmax} (full-data posterior {fpf[cmax]:.3f}) / Min {cmin} ({fpf[cmin]:.3f}); "
          f"n where the Max curve first exceeds 0.75: {fig9.first_crossing(m[cmax])}")
fig = fig9.make_figure(means, MAXMIN)
fig.savefig(os.path.join(a.outdir, "Fig9_mean_posterior_vs_n.png"), dpi=300); plt.close(fig)
print("Fig 9 written; CSVs in", a.outdir)
if a.install:
    if not complete:
        sys.exit("--install refused: some trials are missing")
    for f in ("fig8_fig9_subsampling_trials.csv", "fig9_mean_fp_pickles.csv"):
        shutil.copy(os.path.join(a.outdir, f), os.path.join(ROOT, "data", "intermediate", f))
    print("installed into data/intermediate/")
