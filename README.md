# ES-BMA for nanosheet exfoliation – reproduction package

Code and data that reproduce the analyses and figures of the paper on exhaustive-search Bayesian model
averaging (ES-BMA) for the exfoliation of layered materials into nanosheets (Nanoscale Advances), intended
for the Data Availability Statement (GitHub / Zenodo).

## Licence

This package is released as version v1.0 (the state at the time of submission of the paper).

| Part | Licence | File |
|---|---|---|
| Our code: `src/esbma_repro/`, `scripts/`, the notebooks, `Dockerfile`, `run_nb.sh` | Apache License 2.0 | [`LICENSE`](LICENSE) |
| Data: `data/raw/` (experimental data) and `data/intermediate/` (cached results) | CC BY-NC 4.0 | [`data/LICENSE`](data/LICENSE) |
| Vendored ES-BMA core `src/exhbma/` (exhbma 0.1.12, https://github.com/okada-lab/exhbma) | MIT, Copyright (c) 2022-2026 Koki Obinata (unchanged upstream licence) | [`src/exhbma/LICENSE`](src/exhbma/LICENSE) |

If you use the data or the code, please cite the paper.

## Unified ES-BMA setting

All ES-BMA analyses of the yield data use the setting of Fig 4: `exhbma` 0.1.12 (vendored in `src/exhbma`),
σ<sub>ε</sub> and σ<sub>w</sub> on the grid `inverse(logspace(-2.5, 2.5, 30))` (30 log-spaced points from 10<sup>-2.5</sup> to
10<sup>2.5</sup>, prior ∝ 1/σ), all 2<sup>16</sup> = 65,536 models (including the intercept-only model), model prior with
inclusion probability 0.5, X standardised and y centred with the exhbma scalers; feature posterior = sum of the posterior
probabilities of the models that contain the feature. One fit of the 80 samples serves Fig 3, 4, 5 and 7; the synthetic
study (Fig 6) uses the same ES-BMA and the same Bolasso as Fig 4.

The sample-size studies of Fig 8 and Fig 9 (1,290 ES-BMA fits) were recomputed with the same grid using the fast
evaluator described below (11 min on the analysis server); all figures and numbers of the notebook therefore use one
ES-BMA setting.

## Fast evaluator (`src/esbma_repro/fast_evidence.py`)

Where exhbma spends its time: for every model S, `MarginalLinearRegression` computes one full SVD of X<sub>S</sub> and
then loops in Python over the 30 × 30 = 900 (σ<sub>ε</sub>, σ<sub>w</sub>) grid points, creating 900 `LinearRegression`
objects; each computes a coefficient vector and a log likelihood with ~15 small numpy calls, and the 900 values are
integrated over the grid once for the evidence and once per coefficient (plus a pydantic object per model). The arithmetic
is tiny; the ≈ 60 ms per model is Python/numpy call overhead (≈ 65 CPU-min per 65,536-model fit).

The fast evaluator evaluates the same closed form,
log p(y|S,σ<sub>ε</sub>,σ<sub>w</sub>) = −n/2 log 2π − ½ Σ<sub>i</sub> log(σ<sub>ε</sub>² + σ<sub>w</sub>² s<sub>i</sub>²) − (n−k)/2 log σ<sub>ε</sub>²
− ½[Σ<sub>i</sub> (Uᵀy)<sub>i</sub>²/(σ<sub>ε</sub>² + σ<sub>w</sub>² s<sub>i</sub>²) + (yᵀy − Σ<sub>i</sub>(Uᵀy)<sub>i</sub>²)/σ<sub>ε</sub>²]
+ ½[log σ<sub>ε</sub>² − log(n var y + σ<sub>ε</sub>²)],
with the thin SVDs of hundreds of X<sub>S</sub> of equal size computed in one batched LAPACK call and all 900 grid points
evaluated by broadcasting. Grid integration (1/σ prior weights, cell areas, factor ¼), model prior (α = 0.5), the
intercept-only model and the BMA aggregation (logsumexp with indicator weights) are those of exhbma. Coefficients are not
needed for posteriors; for the 200 models drawn in Fig 3 they are taken from exhbma's per-model call.

| | exhbma (original loop) | fast evaluator, 1 core |
|---|---|---|
| yield trial (65,536 models, 30×30 grid) | ≈ 60 CPU-min (46 s with 100 workers) | 10 s |
| size trial (262,144 models) | ≈ 4 CPU-h (182 s with 100 workers) | 43 s |
| size-distribution trial (32,768 models) | ≈ 30 CPU-min (23 s with 100 workers) | 4.7 s |

Verification against exhbma (maximum absolute differences): full yield fit, all 65,536 models: log evidence 2.8e-13,
feature posteriors 2.0e-14, BMA evidence 5.7e-14, per-grid-point log likelihood 1.4e-6 on values of order 10⁷ (relative
1e-13); nine subsampling trials (trial 1 of yield n = 30/55/80, size n = 30/75/120, size distribution n = 30/60/90),
compared with exhbma run on the same data: log evidence ≤ 4.3e-13, feature posteriors ≤ 4.6e-14; with the original
20-point grid the stored original exhbma trials are reproduced to ≤ 5.3e-14. Every recomputation in the notebook and in
`scripts/recompute_grid30/` uses the fast evaluator by default; `ESBMA_EXHBMA=1` (notebook) or `--exhbma` (scripts)
switches back to the original exhbma loop.

## Contents

```
README.md, requirements.txt, Dockerfile, .gitignore
esbma_paper_reproduction.ipynb              the notebook: one section per paper item
esbma_paper_reproduction.executed.ipynb     executed copy (cached results)
esbma_paper_reproduction.recompute_quick.ipynb  executed copy with ESBMA_RECOMPUTE=1 ESBMA_QUICK=1
data/raw/
  yield_y1.csv                 80 samples x 16 descriptors, y1 = yield (%)          (Fig 3-8)
  size_y2.csv                  122 samples x 18 descriptors, y2 = lateral size       (Fig 9)
  size_distribution_y3.csv     97 samples x 15 descriptors, y3 = size distribution   (Fig 9)
data/intermediate/             cached intermediate results (loaded when RECOMPUTE = False)
  esbma_yield_g30.npz            unified ES-BMA fit of the yield data: marginal log-likelihood of all 65,536 models,
                                 coefficients of the 200 models shown in Fig 3, feature posteriors
  fig4_fp_g30_main.csv           ES-BMA feature posteriors (same fit) + original Bolasso rates (std<1e-12 guard)
  fig4_bolasso_rate_real80_fixed.csv   Bolasso selection rates, fixed standardisation
  fig5_post_real_g30.npz         log joint posterior of (sigma_eps, sigma_w), N = 10..80 (float32)
  fig6_synth_x18_r095_30reps_g30.json  synthetic study: Bolasso rates, ES-BMA posteriors, targets, 30 realisations
  fig7_dos_cve_real_states.csv.gz      5-fold CVE of all 65,535 descriptor subsets, sorted
  fig7_bolasso_supports_real_fixed.csv / ..._cveopt_real_fixed.csv   the 100 Bolasso bootstrap models
  fig8_fig9_subsampling_trials.csv     per-trial posteriors + sample indices of the 1,290 subsampling fits (unified grid)
  fig9_mean_fp_pickles.csv       per-n mean posteriors (trials 0-29) plotted in Fig 9 (unified grid)
  legacy/                        results of the earlier, mixed settings (20-point grid for Fig 3, 8, 9; 31-point grid on
                                 10^-2..10^2 for Fig 5 and 7; separate BMA code for Fig 6) - kept for traceability only
src/exhbma/                    exhbma 0.1.12 (MIT), byte-identical to the PyPI wheel except __init__.py
src/esbma_repro/               analysis code, copied from the original scripts with the algorithms unchanged
  esbma.py      ES-BMA driver (parallel loop around exhbma; identical results to ExhaustiveLinearRegression.fit)
  fast_evidence.py  fast vectorised evaluator of the exhbma evidences (default for all recomputations)
  bolasso.py    Bolasso and CVE-optimal-alpha Bolasso   (~/esbma_bolasso_fix/run_fix.py)
  dos.py        density of states of the CVE            (datalinages/paper_esbma/01_bolasso/run_dos_cve.py)
  synth.py      synthetic single-variable study: run_rep_unified (paper) and the original bma_core version
  subsample.py  sample-size study                       (notebook/experiment3/esbma_nanosheet.py)
  fig3.py       BFE-rank prediction plots               (notebook/experiment5/compare_model.py)
  make_fig8.py  paper drawing of Fig 8 (unchanged copy of make_fig8.py)
  fig9.py       Fig 9: Max / Min selection from the full-data posteriors, drawing in the style of the paper
  data.py       data loading, descriptor names
scripts/extraction/            how the cached files were made (from the original outputs / from the unified fits)
scripts/verification/          equivalence test of the parallel ES-BMA driver; full recomputation of the old Fig 6
scripts/recompute_grid30/      Fig 8 / Fig 9 recomputation with the unified grid (run; see below)
scripts/run_nb_docker.sh       runs the notebook in the original analysis Docker image (server)
figures/                       figures and checks_summary.csv written by the notebook
```

## How to run the notebook

```bash
# option A: Docker (the analysis environment: python:3.9 + requirements.txt)
docker build -t esbma-repro .
docker run --rm -v "$PWD":/repro esbma-repro jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=-1 esbma_paper_reproduction.ipynb --output esbma_paper_reproduction.executed.ipynb

# option B: virtual environment (Python 3.9)
python3.9 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
    esbma_paper_reproduction.ipynb --output esbma_paper_reproduction.executed.ipynb
```
Run from the repository root (all paths are relative). Switches (each section also has its own `RECOMPUTE_FIGx`):

| environment variable | effect |
|---|---|
| `ESBMA_RECOMPUTE=1` | every section recomputes its intermediate results from `data/raw` and compares them with the cache |
| `ESBMA_QUICK=1` | with recompute: Fig 5 only N = 80, Fig 6 only realisation #26, Fig 8 trial 0 at n = 30 and 80, Fig 9 trial 0 at n = 30 |
| `ESBMA_NJOBS=k` | parallel workers of the exhbma path / DoS (default all cores) |
| `ESBMA_EXHBMA=1` | use the original exhbma loop instead of the fast evaluator (≈ 400× slower) |

Recomputed files go to `results/`; `data/intermediate/` is never overwritten. The last cell prints all reproduced numbers
next to their expected values (also written to `figures/checks_summary.csv`).

## What each section reproduces

| Section | Paper item | Cached input | Recomputation |
|---|---|---|---|
| 1 | Fig 3: weight diagrams of BFE ranks 1–100 and 59,951–60,050; cross-validated predictions (DoS protocol) of BFE ranks 1, 100, 60,000 with their CVE | `esbma_yield_g30.npz`, `fig7_dos_cve_real_states.csv.gz` | unified ES-BMA fit |
| 2 | Fig 4 + text: Bolasso vs ES-BMA (N = 80); mean / median probabilities | `fig4_*.csv` | same fit; Bolasso 100 bootstraps |
| 3 | Fig 5 (+ Fig S2): joint posterior of (σ<sub>ε</sub>, σ<sub>w</sub>), MAP, convergence with N | `fig5_post_real_g30.npz` | same fit (N = 80) + 7 fits (N = 10–70) |
| 4 | Fig 6: synthetic y = 1.0·x8 + ε, R² = 0.95, 30 realisations; #26; false positives | `fig6_synth_x18_r095_30reps_g30.json` | 30 × (ES-BMA + Bolasso) |
| 5 | Fig 7: DoS of the CVE of 65,535 models; Bolasso / ES-BMA capture of the CVE top 100 | `fig7_*`, `esbma_yield_g30.npz` | DoS, both Bolassos, same fit |
| 6 | Fig 8: posterior vs sample size (yield, n = 30…80, 30 trials); x9, x13 | `fig8_fig9_subsampling_trials.csv` | 330 fits (≈ 1 CPU-h) |
| 7 | Fig 9: mean posterior of the max/min descriptor vs n (yield, size, size distribution) | same + `fig9_mean_fp_pickles.csv` | 1,290 fits (≈ 8 CPU-h) |

Descriptor columns keep the names of the original 41-descriptor table (x2, x4, …, x40); the paper renumbers the 16
yield descriptors x1–x16 in column order (paper x8 = column `x18`, HSP hydrogen-bonding term of the solvent).

## Changes from the earlier settings (paper numbers to update)

| Item | earlier value (setting) | unified setting |
|---|---|---|
| Fig 3 rank-1 model | {x2,x8,x9,x13,x14,x16} (20-pt grid) | unchanged |
| Fig 3 (c)–(e) | prediction on one test fold (16 samples, KFold seed 42), train/test RMSE | **cross-validated (out-of-fold) predictions with the DoS protocol (KFold(5) seed 43, train-fold standardisation, OLS), CVE** |
| Fig 3 rank 1: CVE | (RMSE train/test 12.9 / 18.7) | **15.297** (CVE rank 18) |
| Fig 3 rank-100 model | {x2,x3,x4,x8,x11,x13,x14,x16} | {x2,x4,x6,x8,x9,x11,x13,x14,x16} |
| Fig 3 rank 100: CVE | (13.5 / 16.9) | **15.516** (CVE rank 78) |
| Fig 3 rank-60,000 model | {x2,x3,x4,x14} | {x2,x5,x6,x10,x12,x15} |
| Fig 3 rank 60,000: CVE | (16.9 [actually 16.99] / 24.9) | **20.076** (CVE rank 61,889) |
| Fig 3 CVE of BFE ranks 1–100 / 59,951–60,050 | – | median 15.57 (15.02–15.98) / 19.52 (17.63–20.48) |
| Fig 3 consistent variables (top 100) | x2,x8,x9,x13,x14,x16 | x8, x13, x14 (+) and x16 (−) in all 100, x2 (+) in 99; x9 (−, 65) and x11 (+, 67) also sign-consistent |

Why the single fold was replaced: the earlier panels (c)–(e) showed one 16-sample test fold. With so few test samples the
test RMSE is dominated by which samples fall into that fold (unified grid: 18.74 / 19.40 / 19.21 for ranks 1 / 100 / 60,000,
i.e. no visible difference between a top and a low-ranked model). Out-of-fold predictions use every sample once as test data,
and their CVE is the same quantity as the DoS of Fig 7 (`fig3.cv_prediction` is built on `dos.build_folds`; the notebook
asserts that the three CVEs equal the DoS values), so Fig 3 and Fig 7 are directly comparable.
| Fig 5 MAP (log10 σε, log10 σw) | 1.20, 0.80 (31-pt grid, ±2) | 1.12, 0.78 |
| Fig 5 posterior sd (log10 σε, log10 σw) | 0.02, 0.15 | 0.04, 0.15 (grid spacing 0.17) |
| Fig 6 Bolasso false positives / run | 1.17 (0.33 solvent + 0.83 unrelated) | 0.97 (0.30 + 0.67); ≥ 1 FP in 19/30 runs |
| Fig 6 Bolasso mean prob. solvent / unrelated | 0.43 / 0.47 | 0.26 / 0.30 |
| Fig 6 ES-BMA false positives / run | 0.07 (2/30 runs) | 0.03 (1/30 runs; x11 = 0.82 in #3) |
| Fig 6 ES-BMA mean prob. of false variables | 0.10–0.12 | 0.06–0.07 |
| Fig 6 #26 Bolasso > 0.75 | x5 0.92, x15 0.90 | x5 0.94, x13 0.77, x15 0.96 |
| Fig 6 #26 ES-BMA max of others | ≤ 0.24 | ≤ 0.09 |
| Fig 7 ES-BMA top 100 in CVE top 100 | 41 | 44 |
| Fig 7 CVE rank of lowest-BFE model | 18 | 18 |
| Fig 7 BFE rank of lowest-CVE model | 4 | 3 |
| Fig 7 {x2,x8,x13,x14,x16}: CVE rank / BFE rank | 64 / 6 | 64 / 38 |
| Fig 4, Fig 7 Bolasso numbers | – | unchanged |
| Fig 8 x13 median posterior, n = 30 → 80 | 0.66 → 0.97 (mean 0.64 → 0.91) | 0.63 → 0.97 (mean 0.64 → 0.93) |
| Fig 8 x9 | stays ≈ 0.45 (mean 0.45 → 0.50) | stays ≈ 0.45 (median), mean 0.45 → 0.51 |
| Fig 8 mean posteriors n = 30 / 45 / 60 | see legacy CSV | x2 0.60/0.70/0.82, x8 0.67/0.73/0.76, x13 0.64/0.78/0.89, x14 0.66/0.77/0.88 |
| Fig 9 Max / Min descriptors | yield x34 / x5, size x4 / x21, size distribution x32 / x8 | **yield x32 / x36, size x16 / x30, size distribution x32 / x41** (highest / lowest posterior on the full data, see below) |
| Fig 9 yield Max > 0.75 first at n | 50 (x34) | **40** (x32; 0.93 at n = 80) |
| Fig 9 size distribution Max > 0.75 first at n | 65 (x32) | 65 (x32; 0.84 at n = 90) |
| Fig 9 size Max > 0.75 first at n | 80 (x4; with the n = 85 run actually using 95 samples) | **100** (x16; 0.83 at n = 120) |
| Fig 9 Min descriptors at the largest n | 0.34 / 0.38 / 0.38 (x5, x21, x8) | 0.38 / 0.38 / 0.41 (x36, x30, x41) |

Fig 9 Max / Min definition: Max = the descriptor with the highest ES-BMA posterior inclusion probability on the **full
data set** of the target (no subsampling; unified setting: 30-point grid, log<sub>10</sub> σ from −2.5 to 2.5, exhbma
standardisation), Min = the one with the lowest; the curves are the means over trials 0–29 at each n. Full-data
posteriors: yield x32 0.995 / x36 0.273 (N = 80), size x16 0.920 / x30 0.323 (N = 122), size distribution x32 0.961 /
x41 0.295 (N = 97). The selection is made automatically (`esbma_repro.fig9.select_maxmin`) in the notebook and in
`aggregate.py`, and asserted. The earlier figure used yield x34/x5, size x4/x21, size distribution x32/x8, which do not
follow this definition.

Note on Fig 6: the earlier synthetic study used its own BMA code (`bma_core.py`: grid logspace(-2, 1.5, 30), y standardised,
intercept marginalised, single-descriptor models skipped) and `LassoCV` with 50 bootstraps; the unified version uses
exhbma and the Fig 4 Bolasso. The targets (noise realisations) are identical.

## Recomputing Fig 8 and Fig 9 (`scripts/recompute_grid30/`)

* `run_all.py` – all studies in one pool of worker processes (one trial per process, single-threaded BLAS, largest fits
  first). Per-study scripts: `fig8_exp31_yield.py` (yield, n = 30…80, 330 fits; = Fig 9 yield, `fig9_yield.py` is an
  alias), `fig9_size.py` (n = 30…120, 570 fits, 2<sup>18</sup> models each; n = 85 now really uses 85 samples – the
  original had been launched with 95), `fig9_size_distribution.py` (n = 30…90, 390 fits; 30 trials per n).
* Sampling scheme and seeds unchanged: `random.seed(trial)`, `random.choices` (draws **with replacement**, the original
  "bootstrap ES-BMA" design). Grid default = unified grid (`--grid LO HI N`), evaluator default = fast (`--exhbma`).
* One JSON file per trial in `results/recompute_grid30/<study>/`; finished trials are skipped (resumable).

```bash
uptime                                                            # check that the server is otherwise idle
scripts/recompute_grid30/launch.sh run_all 100                    # 100 worker processes (use 32 if others are busy)
ls results/recompute_grid30/fig9_size/*.json | wc -l              # progress (finished trials per study)
tail -f results/recompute_grid30/logs/run_all.log
docker stop esbma_grid30_run_all                                  # stop (resume = launch again)
python scripts/recompute_grid30/aggregate.py --install            # CSVs (notebook format), Fig 8 / Fig 9, install
```

Measured on the 128-core server (2026-10-11): `run_all` with 100 workers finished all 1,290 fits in **11.1 min**; load
average ≈ 80 during the run (≈ 9,900 %CPU over 122 Python processes; per-process time rose from 43 s to ≈ 95 s for a size
fit because of memory-bandwidth contention). Single-core time per trial: yield 10 s, size 43 s, size distribution 4.7 s,
i.e. ≈ 8.3 CPU-h in total → ≈ 16–25 min with 32 workers.

`aggregate.py` plots, for Fig 9, the descriptors with the highest / lowest ES-BMA posterior on the full data of each
target (yield x32/x36, size x16/x30, size distribution x32/x41; selected automatically and asserted).

## Verification and run times

* `RECOMPUTE = False`: executed end-to-end in the original Docker image on the analysis server and in a fresh `.venv`
  from `requirements.txt` on macOS (arm64); all printed checks match their expected values (table at the end of the
  notebook, 59 checks). Whole notebook < 1 min (≈ 30 s of it for the three full-data fits of Fig 9).
* `ESBMA_RECOMPUTE=1 ESBMA_QUICK=1` on the server (fast evaluator): the unified fit reproduces the cached
  log-likelihoods to 3e-13 and the posteriors to 2e-14 (coefficients identical); Bolasso and the DoS are identical;
  Fig 6 #26 to 2e-14; the Fig 8/9 trial-0 checks to ≤ 1e-16.
* `ESBMA_RECOMPUTE=1 ESBMA_QUICK=1` with the fast evaluator: 2 min for the whole notebook.
* The unified synthetic study (30 realisations) took 21 min with the exhbma loop (110 workers); ≈ 6 min with the fast
  evaluator on one core.
* The parallel driver `esbma.fit_exhaustive` was checked against the untouched serial `ExhaustiveLinearRegression.fit`
  (identical log-likelihoods, coefficients and posteriors).

## Provenance of the original (legacy) results

Paths relative to `esbma_research-main/user/` on the analysis server unless stated otherwise: Fig 3
`namiuchi/output/exp1/run2/model.pickle` and `namiuchi/notebook/experiment5/*`; Fig 4 `redo_fp_main.py`, `fp_g30_main.csv`,
`~/esbma_bolasso_fix/run_fix.py`; Fig 5 `namiuchi/notebook/experiment1/montage_conv.py`; Fig 6 `~/esbma_synth/`; Fig 7
`namiuchi/datalinages/paper_esbma/01_bolasso/run_dos_cve.py`, `run_bfe_g31.py`; Fig 8 `namiuchi/notebook/experiment3/` and
`namiuchi/output/exp31/`, `make_fig8.py`; Fig 9 `katsuta/notebook/experiment3/` and `namiuchi/katsuta_analysis/analyze.py`.

## Other points for the authors

1. Fig 8/9 subsampling draws with replacement (`random.choices`), so an "n-sample" set contains duplicates.
2. exhbma includes the intercept-only model (65,536 models in the BMA); the DoS covers the 65,535 non-empty subsets.
