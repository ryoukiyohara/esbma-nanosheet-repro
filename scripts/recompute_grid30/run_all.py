"""All studies (Fig 8 = Fig 9 yield, Fig 9 size, Fig 9 size distribution) in ONE worker pool, largest fits first."""
from subsampling_core import main
if __name__ == "__main__":
    main(["fig9_size", "fig8_exp31_yield", "fig9_size_distribution"])
