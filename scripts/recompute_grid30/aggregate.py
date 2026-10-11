"""Collect the per-trial JSON files of the Fig 8 / Fig 9 recomputation, build the intermediate CSVs in the format
used by the notebook, and redraw Fig 8 (make_fig8.make_figure) and Fig 9.

    python scripts/recompute_grid30/aggregate.py [--outdir results/recompute_grid30] [--install]

--install copies the two CSVs to data/intermediate/ (fig8_fig9_subsampling_trials.csv, fig9_mean_fp_pickles.csv),
replacing the legacy 20-point-grid versions, so that the notebook shows the recomputed results. Only use it when all
trials are complete (the script reports missing trials and refuses --install otherwise).
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
ORIG_MAXMIN = {"yield": ("x34", "x5"), "size": ("x4", "x21"), "size_distribution": ("x32", "x8")}

ap = argparse.ArgumentParser(); ap.add_argument("--outdir", default=os.path.join(ROOT, "results", "recompute_grid30"))
ap.add_argument("--install", action="store_true")
ap.add_argument("--data-driven", action="store_true", help="plot the descriptors with the highest / lowest mean posterior at the "
                "largest n instead of the descriptors of the published Fig 9 (yield x34/x5, size x4/x21, size distr. x32/x8)")
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
plt.rcdefaults(); plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans"]})

# Fig 9: Max / Min descriptor = highest / lowest mean posterior at the largest n (as in the paper)
fig, ax = plt.subplots(figsize=(12, 6))
for tg, cmap in [("yield", "Blues"), ("size", "Greens"), ("size_distribution", "Reds")]:
    m = M[M.target == tg].set_index("n").drop(columns="target").dropna(axis=1, how="all")
    if not len(m):
        continue
    last = m.iloc[-1]; dmax, dmin = last.idxmax(), last.idxmin()
    cmax, cmin = (dmax, dmin) if a.data_driven else ORIG_MAXMIN[tg]
    print(f"{tg}: plotted Max {cmax} / Min {cmin} (highest / lowest at n = {m.index[-1]}: {dmax} {last[dmax]:.3f} / {dmin} {last[dmin]:.3f}); "
          f"n where the plotted Max first exceeds 0.75: {next((n for n, v in m[cmax].items() if v > 0.75), None)}")
    c = plt.get_cmap(cmap); lab = {"yield": "Yield", "size": "Size", "size_distribution": "Size Dist"}[tg]
    ax.plot(m.index.to_numpy(), m[cmax].to_numpy(), "o-", color=c(0.8), label=f"{lab} (Max)")
    ax.plot(m.index.to_numpy(), m[cmin].to_numpy(), "o-", color=c(0.45), label=f"{lab} (Min)")
ax.axhline(0.75, color="red", ls="--"); ax.axhline(0.25, color="red", ls="--"); ax.set_ylim(0, 1)
ax.set_yticks([0, .25, .5, .75, 1]); ax.set_xticks(np.arange(30, 125, 5))
ax.set_xlabel("Number of sample data", fontsize=16); ax.set_ylabel("Probability", fontsize=16); ax.legend(ncol=3, fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(a.outdir, "Fig9_mean_posterior_vs_n.png"), dpi=300); plt.close(fig)
print("Fig 9 written; CSVs in", a.outdir)
if a.install:
    if not complete:
        sys.exit("--install refused: some trials are missing")
    for f in ("fig8_fig9_subsampling_trials.csv", "fig9_mean_fp_pickles.csv"):
        shutil.copy(os.path.join(a.outdir, f), os.path.join(ROOT, "data", "intermediate", f))
    print("installed into data/intermediate/")
