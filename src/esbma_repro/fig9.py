"""Fig 9: mean ES-BMA posterior of the Max / Min descriptor vs number of samples (yield, size, size distribution).

Max / Min definition (as in the paper): the descriptor with the highest / lowest ES-BMA posterior inclusion probability
on the FULL data set of the target (no subsampling), unified setting (exhbma conventions, sigma grid
inverse(logspace(-2.5, 2.5, 30)), X standardised, y centred). The plotted curve is the mean posterior of that descriptor
over subsampling trials 0-29 at each n.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import data as D
from .fast_evidence import fit_fast

TARGETS = ["yield", "size", "size_distribution"]
LABELS = {"yield": "Yield", "size": "Size", "size_distribution": "Size distribution"}
COLORS = {"yield": "#1f77b4", "size": "#2ca02c", "size_distribution": "#d62728"}
EXPECTED_MAXMIN = {"yield": ("x32", "x36"), "size": ("x16", "x30"), "size_distribution": ("x32", "x41")}


def full_data_posteriors(target, raw="data/raw", grid=(-2.5, 2.5, 30)):
    """ES-BMA feature posteriors of all samples of `target` (pd.Series indexed by column name)."""
    X, y, cols = D.load(target, raw)
    return pd.Series(fit_fast(X, y, *grid)["feature_posteriors"], index=cols)


def select_maxmin(raw="data/raw", grid=(-2.5, 2.5, 30)):
    """{target: (Max column, Min column, full-data posteriors)} from the full-data ES-BMA fits."""
    out = {}
    for tg in TARGETS:
        fp = full_data_posteriors(tg, raw, grid)
        out[tg] = (fp.idxmax(), fp.idxmin(), fp)
    return out


def first_crossing(series, level=0.75):
    """Smallest n at which the curve exceeds `level` (None if never)."""
    return next((int(n) for n, v in series.items() if v > level), None)


def make_figure(means, maxmin):
    """means: {target: DataFrame indexed by n with the Max and Min columns}; maxmin: {target: (cmax, cmin, ...)}.
    Style of the published figure: double-column width 17.1 cm, x range 27-123, legend on the right, 8 pt ticks and legend, 9 pt axis labels."""
    rc = {"font.family": ["Arial", "DejaVu Sans"], "font.size": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
          "axes.labelsize": 9, "legend.fontsize": 8, "axes.linewidth": 0.8,
          "xtick.major.width": 0.8, "ytick.major.width": 0.8}
    with plt.rc_context(rc):
        fig = plt.figure(figsize=(17.1 / 2.54, 2.5))
        ax = fig.add_axes([0.08, 0.18, 0.64, 0.78])
        handles = []
        for tg in TARGETS:
            if tg not in means:
                continue
            cmax, cmin = maxmin[tg][:2]; m = means[tg]; c = COLORS[tg]; n = m.index.to_numpy()
            h1, = ax.plot(n, m[cmax].to_numpy(), "-o", color=c, lw=1.0, ms=3.2, mfc=c, mec=c, label=f"{LABELS[tg]} (Max)")
            h2, = ax.plot(n, m[cmin].to_numpy(), "--o", color=c, lw=1.0, ms=3.2, mfc="white", mec=c, mew=1.0,
                          label=f"{LABELS[tg]} (Min)")
            handles += [h1, h2]
        for yv in (0.75, 0.25):
            ax.axhline(yv, color="red", ls="--", lw=0.8, zorder=0)
        ax.set_xlim(27, 123); ax.set_xticks(np.arange(30, 121, 10))
        ax.set_ylim(0, 1); ax.set_yticks(np.arange(0, 1.01, 0.2))
        ax.set_xlabel("Number of samples"); ax.set_ylabel("Mean posterior probability")
        ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False,
                  handlelength=2.5, labelspacing=0.7)
    return fig
