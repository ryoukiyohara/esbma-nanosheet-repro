#!/bin/bash
# Execute the notebook inside the original analysis image (Python 3.9, exhbma deps, nbconvert 7.16.6).
# Usage: ./run_nb.sh [output.ipynb]   (env ESBMA_RECOMPUTE / ESBMA_QUICK / ESBMA_NJOBS are passed through)
OUT=${1:-esbma_paper_reproduction.executed.ipynb}
docker run --rm --user $(id -u):$(id -g) -e HOME=/tmp -e MPLCONFIGDIR=/tmp \
  -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
  -e ESBMA_RECOMPUTE=${ESBMA_RECOMPUTE:-0} -e ESBMA_QUICK=${ESBMA_QUICK:-0} -e ESBMA_NJOBS=${ESBMA_NJOBS:--1} \
  -v "$PWD":/repro -w /repro --entrypoint jupyter esbma_research-main-jupyter \
  nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 esbma_paper_reproduction.ipynb --output "$OUT"
