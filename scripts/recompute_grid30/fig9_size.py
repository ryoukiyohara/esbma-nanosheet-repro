"""Fig 9, size (122 samples, 18 descriptors -> 262,144 models per fit): n = 30, 35, ..., 120, trials 0-29.
Correction: n = 85 now really uses 85 samples (the original run labelled 85 used 95)."""
from subsampling_core import main
if __name__ == "__main__":
    main(["fig9_size"])
