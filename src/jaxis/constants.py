"""Constants used throughout the jaxis package."""

import jax
from jax.typing import DTypeLike

# Random number generation
DEFAULT_RNG_SEED: int = 0

# Numerical tolerances
DEFAULT_ATOL: float = 1e-6
DEFAULT_RTOL: float = 1e-6

# Testing/validation
DEFAULT_TRIALS: int = 60

# Data types
DEFAULT_DTYPE: DTypeLike = jax.numpy.float32
