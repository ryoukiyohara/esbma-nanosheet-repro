"""Loading the experimental datasets in data/raw.

yield_y1.csv              80 samples x 16 descriptors, target y1 = exfoliation yield (%)
size_y2.csv              122 samples x 18 descriptors, target y2 = lateral size
size_distribution_y3.csv  97 samples x 15 descriptors, target y3 = size distribution
Descriptor columns keep the names of the original 41-descriptor table (x2, x4, ...); the paper
renumbers the 16 yield descriptors as x1..x16 in column order (PAPER_NAMES below).
"""
import os

import numpy as np
import pandas as pd

YIELD_COLS = ["x2", "x4", "x5", "x8", "x10", "x14", "x17", "x18", "x26", "x28", "x29", "x31",
              "x32", "x34", "x36", "x40"]
PAPER_NAMES = [f"x{i + 1}" for i in range(16)]
DESCRIPTION = ["Molecular length", "Boiling point", "Density", "Viscosity", "Surface tension",
               "Dipole moment", "HSP polarity (solv.)", "HSP H-bond (solv.)", "Interlayer distance",
               "x (TG)", "Interlayer density", "HSP polarity (guest)", "HSP H-bond (guest)",
               "Δ Polarizability", "Δ Heat capacity", "HSP distance"]

TARGETS = {"yield": ("yield_y1.csv", "y1", "サンプル名"),
           "size": ("size_y2.csv", "y2", "No."),
           "size_distribution": ("size_distribution_y3.csv", "y3", "No.")}


def load(target="yield", root="data/raw"):
    """Return X (n, p) float, y (n,), descriptor column names (original numbering)."""
    fn, ycol, idcol = TARGETS[target]
    df = pd.read_csv(os.path.join(root, fn))
    X = df.drop(columns=[idcol, ycol])
    return X.to_numpy(float), df[ycol].to_numpy(float), list(X.columns)
