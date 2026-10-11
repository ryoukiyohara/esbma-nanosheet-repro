# Paper-style drawing of Fig 8, copied unchanged from make_fig8.py (the script used to redraw the
# published Fig 8). The notebook imports make_figure() and passes it the per-trial posteriors; the
# stand-alone main()/load() of the original (reading fig8_work/fp_exp31.csv) are kept for reference.
"""Redraw Figure 8 (effect of sample size on ES-BMA feature posterior probabilities).

Data: fig8_work/fp_exp31.csv  (one row per n_data x trial; x1..x16 = posterior
probability of each feature, i.e. the fp.pickle of
namiuchi/output/exp31/{n}/trial_{i}/ on cpu1, 11 sample sizes x 30 trials).

(a)-(c): mean over the 30 trials for n = 30, 45, 60 (bar charts).
(d), (e): box plots (Tukey, 1.5 IQR) over the 30 trials of x9 and x13 vs n.

Outputs: figs/Fig8_sample_size.png (300 dpi) and figs/Fig8_sample_size.tif
(RGB, LZW, 600 dpi tag, 4039 px wide = 17.1 cm).
"""
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "fig8_work", "fp_exp31.csv")
OUTDIR = os.path.join(HERE, "figs")

# ---- colours sampled from the published panel (image5.png) ----------------
GREY_BAR = "#C2C2C2"
GREY_TXT = "#595959"
GREY_FRAME = "#D1D1D1"
X9_LIGHT = "#8191AE"     # title / labels / band fill (alpha 0.5)
X9_DARK = "#58657E"      # band border, arrow
X9_FILL = "#606879"      # box fill
X9_LINE = "#5A6378"      # whiskers, caps
X9_MED = "#2E323B"
X9_FRAME = "#636C7E"
X13_LIGHT = "#A3B794"
X13_DARK = "#6F7D65"
X13_FILL = "#A5C291"
X13_LINE = "#94B87E"
X13_MED = "#6D9D4D"
X13_FRAME = "#ACCA99"
BAND_ALPHA = 0.5

NS = list(range(30, 85, 5))

plt.rcParams.update({
    "font.family": ["Arial", "DejaVu Sans"],
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "mathtext.fontset": "stix",
    "svg.fonttype": "none",
})

FS_TICK = 9.5
FS_LABEL = 12.5
FS_TITLE = 14
FS_PANEL = 19
FS_N = 12


def load():
    data = {}
    with open(DATA, newline="") as f:
        for row in csv.DictReader(f):
            n = int(row["n_data"])
            data.setdefault(n, []).append([float(row[f"x{j}"]) for j in range(1, 17)])
    return {n: np.array(v) for n, v in data.items()}


def blend(hex_c, alpha, bg):
    c = np.array(matplotlib.colors.to_rgb(hex_c))
    b = np.array(matplotlib.colors.to_rgb(bg))
    return tuple(alpha * c + (1 - alpha) * b)


def style_axes(ax, frame, txt, xsize=FS_TICK):
    for s in ax.spines.values():
        s.set_color(frame)
        s.set_linewidth(0.8)
    ax.tick_params(length=0, pad=2.5, labelsize=FS_TICK, colors=txt)
    ax.tick_params(axis="x", labelsize=xsize)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_fontweight("bold")


def bar_panel(ax, mean, n):
    x = np.arange(1, 17)
    colors = [GREY_BAR] * 16
    # bars under the highlight bands are shown with the band tint already applied
    colors[8] = blend(X9_LIGHT, BAND_ALPHA, GREY_BAR)
    colors[12] = blend(X13_LIGHT, BAND_ALPHA, GREY_BAR)
    ax.bar(x, mean, width=0.8, color=colors, zorder=2)
    ax.set_xlim(0.2, 16.8)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_yticklabels([f"{v:.1f}" for v in np.arange(0, 1.01, 0.2)])
    ax.set_facecolor("none")
    ax.set_ylabel("Probability", fontsize=FS_LABEL, color=GREY_TXT, labelpad=3)
    ax.set_xlabel(r"Variable $x_n$", fontsize=FS_LABEL, color=GREY_TXT, labelpad=0)
    ax.text(0.04, 0.86, f"n = {n}", transform=ax.transAxes, fontsize=FS_N,
            color=GREY_TXT, fontweight="bold", va="center")
    style_axes(ax, GREY_FRAME, GREY_TXT, xsize=FS_TICK - 1)


def box_panel(ax, values, fill, line, med, frame, txt, title):
    bp = ax.boxplot(values, positions=range(len(NS)), widths=0.8, whis=1.5,
                    patch_artist=True, manage_ticks=False,
                    boxprops=dict(facecolor=fill, edgecolor=fill, linewidth=0.8),
                    medianprops=dict(color=med, linewidth=1.0),
                    whiskerprops=dict(color=line, linewidth=0.9),
                    capprops=dict(color=line, linewidth=0.9),
                    flierprops=dict(marker="D", markersize=2.6, markerfacecolor=line,
                                    markeredgecolor="none"))
    ax.set_xlim(-0.55, len(NS) - 0.45)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(range(len(NS)))
    ax.set_xticklabels([str(n) for n in NS])
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_yticklabels([f"{v:.1f}" for v in np.arange(0, 1.01, 0.2)])
    ax.set_ylabel("Probability", fontsize=FS_LABEL, color=txt, labelpad=4)
    ax.set_xlabel("Number of sample data", fontsize=FS_LABEL, color=txt, labelpad=1)
    ax.set_title(title, fontsize=FS_TITLE, color=txt, fontweight="bold", pad=7)
    style_axes(ax, frame, txt)
    return bp


def make_figure(data):
    W_IN = 4039 / 600.0               # 17.1 cm
    H_IN = W_IN * 1059 / 1222         # aspect of the published figure
    fig = plt.figure(figsize=(W_IN, H_IN), facecolor="white")

    def fx(px):
        return px / 1222

    def fy(py):
        return 1 - py / 1059

    # left column (a)-(c): [left, bottom, width, height] from the published layout
    left, right = fx(190), fx(600)
    rows = [(78, 336), (400, 658), (722, 980)]
    axs = []
    for (t, b), n in zip(rows, (30, 45, 60)):
        ax = fig.add_axes([left, fy(b), right - left, fy(t) - fy(b)])
        bar_panel(ax, data[n].mean(axis=0), n)
        axs.append(ax)

    # right column (d), (e)
    dl, dr = fx(730), fx(1205)
    axd = fig.add_axes([dl, fy(462), dr - dl, fy(108) - fy(462)])
    axe = fig.add_axes([dl, fy(958), dr - dl, fy(604) - fy(958)])
    box_panel(axd, [data[n][:, 8] for n in NS], X9_FILL, X9_LINE, X9_MED, X9_FRAME,
              X9_LIGHT, r"Variable $x_9$")
    box_panel(axe, [data[n][:, 12] for n in NS], X13_FILL, X13_LINE, X13_MED, X13_FRAME,
              X13_LIGHT, r"Variable $x_{13}$")

    # panel letters
    for lab, (x, y) in {"(a)": (40, 100), "(b)": (40, 410), "(c)": (40, 720),
                        "(d)": (630, 100), "(e)": (630, 610)}.items():
        fig.text(fx(x), fy(y), lab, fontsize=FS_PANEL, fontweight="bold",
                 va="center", ha="left", color="black")

    # vertical highlight bands through (a)-(c), drawn behind the (transparent) axes
    fig.canvas.draw()
    to_fig = fig.transFigure.inverted()
    ax_a, ax_c = axs[0], axs[2]
    band_top = fy(116)
    band_bot = fy(1046)
    title_y = fy(50)
    for var, light, dark in ((9, X9_LIGHT, X9_DARK), (13, X13_LIGHT, X13_DARK)):
        x0 = to_fig.transform(ax_a.transData.transform((var - 0.48, 0)))[0]
        x1 = to_fig.transform(ax_a.transData.transform((var + 0.48, 0)))[0]
        fig.patches.append(Rectangle((x0, band_bot), x1 - x0, band_top - band_bot,
                                     transform=fig.transFigure, facecolor=light,
                                     alpha=BAND_ALPHA, edgecolor="none", zorder=-2))
        fig.patches.append(Rectangle((x0, band_bot), x1 - x0, band_top - band_bot,
                                     transform=fig.transFigure, facecolor="none",
                                     edgecolor=dark, linewidth=1.2, zorder=-1))
        xc = (x0 + x1) / 2
        fig.patches.append(FancyArrowPatch((xc, fy(68)), (xc, band_top + 0.003),
                                           transform=fig.transFigure,
                                           arrowstyle="-|>,head_width=3.0,head_length=4.2",
                                           color=dark, linewidth=2.0, zorder=3))
        if var == 9:
            fig.text(xc + fx(52), title_y, r"Variable $x_9$", ha="right", va="center",
                     fontsize=FS_TITLE, color=light, fontweight="bold")
        else:
            fig.text(xc - fx(26), title_y, r"Variable $x_{13}$", ha="left", va="center",
                     fontsize=FS_TITLE, color=light, fontweight="bold")
    return fig


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    data = load()
    assert sorted(data) == NS and all(v.shape == (30, 16) for v in data.values())
    fig = make_figure(data)
    png = os.path.join(OUTDIR, "Fig8_sample_size.png")
    fig.savefig(png, dpi=300, facecolor="white")
    hi = os.path.join(OUTDIR, "_Fig8_hi.png")
    fig.savefig(hi, dpi=1200, facecolor="white")
    plt.close(fig)
    im = Image.open(hi).convert("RGB")
    w = 4039
    h = round(im.height * w / im.width)
    im = im.resize((w, h), Image.LANCZOS)
    tif = os.path.join(OUTDIR, "Fig8_sample_size.tif")
    im.save(tif, compression="tiff_lzw", dpi=(600, 600))
    os.remove(hi)
    print(png, tif, im.size)


if __name__ == "__main__":
    main()
