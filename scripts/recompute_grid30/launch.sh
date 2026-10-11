#!/bin/bash
# Launch the Fig 8 / Fig 9 recomputation with the unified ES-BMA grid (Fig 4 setting) in the background.
#
#   scripts/recompute_grid30/launch.sh <study> [workers] [extra args...]
#     study   : run_all (all studies, one pool) | fig8_exp31_yield | fig9_size | fig9_size_distribution  (fig9 yield = fig8)
#     workers : worker processes, one trial each, single-threaded BLAS (default 32; check uptime first)
#     extra   : passed to the study script, e.g.  --n 30 40 --trials 0 1
#   Runs inside the original analysis image (esbma_research-main-jupyter) if docker is available, otherwise with
#   $PYTHON (default python3). Run from anywhere; paths are relative to the repository root.
#
# Progress : ls results/recompute_grid30/<study>/*.json | wc -l      (one file per finished trial)
#            tail -f results/recompute_grid30/logs/<study>.log
# Stop     : docker stop esbma_grid30_<study>      (docker)   or   kill $(cat results/recompute_grid30/logs/<study>.pid)
# Resume   : run the same command again (finished trials are skipped).
# Afterwards: python scripts/recompute_grid30/aggregate.py [--install]
set -eu
STUDY=${1:?study}; WORKERS=${2:-32}; shift || true; shift || true
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
LOG=$ROOT/results/recompute_grid30/logs; mkdir -p "$LOG"
if command -v docker >/dev/null && docker image inspect esbma_research-main-jupyter >/dev/null 2>&1 && [ -z "${NO_DOCKER:-}" ]; then
  docker run -d --rm --name "esbma_grid30_$STUDY" --user "$(id -u):$(id -g)" -e HOME=/tmp -e MPLCONFIGDIR=/tmp \
    -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e PYTHONUNBUFFERED=1 \
    -v "$ROOT":/repro -w /repro --entrypoint python esbma_research-main-jupyter \
    "scripts/recompute_grid30/$STUDY.py" --workers "$WORKERS" "$@" > "$LOG/$STUDY.cid"
  nohup docker logs -f "esbma_grid30_$STUDY" > "$LOG/$STUDY.log" 2>&1 &
  echo "started container esbma_grid30_$STUDY; log $LOG/$STUDY.log; stop: docker stop esbma_grid30_$STUDY"
else
  cd "$ROOT"
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
    nohup "${PYTHON:-python3}" "scripts/recompute_grid30/$STUDY.py" --workers "$WORKERS" "$@" > "$LOG/$STUDY.log" 2>&1 &
  echo $! > "$LOG/$STUDY.pid"
  echo "started PID $(cat "$LOG/$STUDY.pid"); log $LOG/$STUDY.log; stop: kill \$(cat $LOG/$STUDY.pid)"
fi
