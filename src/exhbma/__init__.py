# Vendored copy of exhbma 0.1.12 (https://github.com/okada-lab/exhbma, MIT licence, (c) Koki Obinata).
# The algorithm files (exhaustive_search.py, linear_regression.py, constant_regression.py,
# integrate.py, probabilities.py, scaler.py, plot.py) are byte-identical to the PyPI wheel
# exhbma==0.1.12 that was installed in the analysis environment (verified against the wheel's
# RECORD sha256 hashes). Only this __init__ was changed: the deprecated `pkg_resources` version
# lookup was replaced by a constant so the package imports on current setuptools.
from .constant_regression import ConstantRegression, MarginalConstantRegression
from .exhaustive_search import ExhaustiveLinearRegression
from .integrate import (
    integrate_log_values_in_line,
    integrate_log_values_in_square,
    validate_list_dimension,
)
from .linear_regression import LinearRegression, MarginalLinearRegression
from .plot import feature_posterior, sigma_posterior, weight_diagram
from .probabilities import RandomVariable, gamma, inverse, uniform
from .scaler import StandardScaler

__version__ = "0.1.12"
