"""Shared driver for the sample-size (subsampling) studies of Fig 8 and Fig 9.

Scheme (unchanged from the original esbma_nanosheet*.py): for data size n and trial i
    random.seed(i); np.random.seed(i); idx = random.choices(range(N_total), k=n)   # bootstrap draw, with replacement
    exhbma standardisation of X[idx], y[idx]; ES-BMA over all 2^p models -> feature posteriors.
The draw with replacement is the original design ("bootstrap ES-BMA") and is kept; seeds are the trial numbers.
Changes: the sigma grid (default: the Fig 4 setting inverse(logspace(-2.5, 2.5, 30))), and every run uses the n it is
labelled with (the original size run labelled n = 85 had been launched with 95 samples).

Evaluator (default "fast"): src/esbma_repro/fast_evidence.py - the exhbma evidences computed with batched SVDs and
vectorised grid evaluation (identical to exhbma to ~1e-13; see README). --exhbma uses the original per-model exhbma
code instead (model-level parallelism inside one trial).
Parallelism: with the fast evaluator, one trial per worker process (single-threaded BLAS), --workers processes.

Each trial is written to its own JSON file <outdir>/<study>/n<NNN>_trial<TT>.json (atomically); existing files are
skipped, so the run can be stopped and resumed at any time.
"""
import argparse, json, os, sys, time

os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import numpy as np                                                   # noqa: E402
from esbma_repro import data as D, esbma, subsample, fast_evidence  # noqa: E402

STUDIES = {"fig8_exp31_yield": ("yield", list(range(30, 85, 5))),
           "fig9_size": ("size", list(range(30, 125, 5))),
           "fig9_size_distribution": ("size_distribution", list(range(30, 95, 5)))}


def outdir_for(base, study, grid):
    lo, hi, gn = grid
    return os.path.join(base, study if (lo, hi, gn) == (-2.5, 2.5, 30) else f"{study}_grid{gn}_{lo:g}_{hi:g}")


def run_trial(study, target, n, t, grid, out, exhbma_workers=None):
    fn = os.path.join(out, f"n{n:03d}_trial{t:02d}.json")
    if os.path.exists(fn):
        return None
    X, y, cols = D.load(target, os.path.join(ROOT, "data", "raw"))
    t1 = time.time()
    idx = subsample.trial_indices(len(X), n, t)
    if exhbma_workers:
        r = esbma.fit_exhaustive(X[idx], y[idx], *grid, n_jobs=exhbma_workers); ev = "exhbma"
    else:
        r = fast_evidence.fit_fast(X[idx], y[idx], *grid); ev = "fast"
    rec = dict(study=study, target=target, n=n, trial=t, grid=list(grid), evaluator=ev, sample_indexs=idx,
               feature_columns=cols, fp=np.asarray(r["feature_posteriors"]).tolist(),
               log_evidence=float(r["log_likelihood"]), seconds=time.time() - t1)
    with open(fn + ".tmp", "w") as f:
        json.dump(rec, f)
    os.replace(fn + ".tmp", fn)
    return study, n, t, rec["seconds"]


def main(studies):
    ap = argparse.ArgumentParser(description="subsampling ES-BMA for " + ", ".join(studies))
    ap.add_argument("--workers", type=int, default=int(os.environ.get("ESBMA_WORKERS", 32)),
                    help="parallel worker processes (default 32 or $ESBMA_WORKERS)")
    ap.add_argument("--n", type=int, nargs="*", default=None, help="data sizes (default: all of the study)")
    ap.add_argument("--trials", type=int, nargs="*", default=list(range(30)), help="trial numbers (default 0-29)")
    ap.add_argument("--grid", type=float, nargs=3, default=[-2.5, 2.5, 30], metavar=("LO", "HI", "N"),
                    help="sigma grid inverse(logspace(LO, HI, N)); default = Fig 4 setting -2.5 2.5 30")
    ap.add_argument("--exhbma", action="store_true", help="use the original (slow) exhbma per-model code")
    ap.add_argument("--outdir", default=os.path.join(ROOT, "results", "recompute_grid30"))
    a = ap.parse_args()
    grid = (a.grid[0], a.grid[1], int(a.grid[2]))
    tasks = []
    for study in studies:
        target, ns = STUDIES[study]
        out = outdir_for(a.outdir, study, grid); os.makedirs(out, exist_ok=True)
        for n in (a.n or ns):
            for t in a.trials:
                if not os.path.exists(os.path.join(out, f"n{n:03d}_trial{t:02d}.json")):
                    tasks.append((study, target, n, t, grid, out))
    tasks.sort(key=lambda x: -(2 ** D.load(x[1], os.path.join(ROOT, "data", "raw"))[0].shape[1]))   # largest first
    print(f"[{'+'.join(studies)}] grid={grid} evaluator={'exhbma' if a.exhbma else 'fast'} workers={a.workers} "
          f"trials to run={len(tasks)} -> {a.outdir}", flush=True)
    t0 = time.time(); done = 0
    if a.exhbma:
        for tk in tasks:
            res = run_trial(*tk, exhbma_workers=a.workers); done += res is not None
            print(f"{res} done={done}/{len(tasks)} elapsed {(time.time() - t0) / 60:.1f} min", flush=True)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = [ex.submit(run_trial, *tk) for tk in tasks]
            for f in as_completed(futs):
                res = f.result(); done += 1
                if res:
                    print(f"[{res[0]}] n={res[1]} trial={res[2]} {res[3]:.0f}s  done {done}/{len(tasks)}  "
                          f"elapsed {(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"finished: {done} trial(s) in {(time.time() - t0) / 60:.1f} min", flush=True)
