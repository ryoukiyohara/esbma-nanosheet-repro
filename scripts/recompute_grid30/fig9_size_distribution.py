"""Fig 9, size distribution (97 samples, 15 descriptors): n = 30, 35, ..., 90, trials 0-29.
(The original had 50 trials for n <= 80, but only trials 0-29 entered the plotted mean; 30 trials are run here.)"""
from subsampling_core import main
if __name__ == "__main__":
    main(["fig9_size_distribution"])
