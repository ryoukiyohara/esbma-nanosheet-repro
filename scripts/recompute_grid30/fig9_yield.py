"""Fig 9, yield. The original Fig 9 yield trials (katsuta/.../output/yield) are identical to the Fig 8 trials
(output/exp31: same data, seeds, n and grid), so this is the same computation and writes into the same directory as
fig8_exp31_yield.py: running either one fills both (finished trials are skipped)."""
from subsampling_core import main
if __name__ == "__main__":
    main(["fig8_exp31_yield"])
