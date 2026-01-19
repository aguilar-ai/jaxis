"""
# Testing Regime: Invariant Permutability

Should PASS:

- Sum, mean, max, min
- Variance, standard deviation
- Median, mode
- Norms (L1, L2, etc.)
- Product of all elements

Should FAIL:

- Identity
- Element-wise operations
- First/last element selector
- Weighted sum with positional weights
- Any function where output shape matches input shape (usually)
"""

import jaxis.semantics as jxs
from jaxis.common_typing import Array
from jaxis.fn import DeferredCall

# Should PASS


def test_sum_function(sum_fn: DeferredCall[Array]) -> None:
  # NOTE: Here, `is_permutation_invariant` generates float32 inputs and then compares. For
  # a reduction like sum, permuting inputs changes the order of floating-point additions,
  # which can shfit results by > 1e-6, even though the function is mathematically
  # invariant. Hence, we relax the tolerance to 1e-5.
  assert jxs.is_permutation_invariant(sum_fn, atol=1e-5, rtol=1e-5)


def test_mean_function(mean_fn: DeferredCall[Array]) -> None:
  assert jxs.is_permutation_invariant(mean_fn)


def test_max_function(max_fn: DeferredCall[Array]) -> None:
  assert jxs.is_permutation_invariant(max_fn)


def test_min_function(min_fn: DeferredCall[Array]) -> None:
  assert jxs.is_permutation_invariant(min_fn)


def test_variance_function(var_fn: DeferredCall[Array]) -> None:
  assert jxs.is_permutation_invariant(var_fn)


def test_std_function(std_fn: DeferredCall[Array]) -> None:
  assert jxs.is_permutation_invariant(std_fn)


def test_median_function(median_fn: DeferredCall[Array]) -> None:
  assert jxs.is_permutation_invariant(median_fn)


def test_prod_function(prod_fn: DeferredCall[Array]) -> None:
  assert jxs.is_permutation_invariant(prod_fn)


def test_l1_norm_function(l1_norm_fn: DeferredCall[Array]) -> None:
  assert jxs.is_permutation_invariant(l1_norm_fn)


def test_l2_norm_function(l2_norm_fn: DeferredCall[Array]) -> None:
  assert jxs.is_permutation_invariant(l2_norm_fn)


# Should FAIL


def test_identity_function(identity_fn: DeferredCall[Array]) -> None:
  assert not jxs.is_permutation_invariant(identity_fn)


def test_square_function(square_fn: DeferredCall[Array]) -> None:
  assert not jxs.is_permutation_invariant(square_fn)


def test_sin_function(sin_fn: DeferredCall[Array]) -> None:
  assert not jxs.is_permutation_invariant(sin_fn)


def test_relu_function(relu_fn: DeferredCall[Array]) -> None:
  assert not jxs.is_permutation_invariant(relu_fn)


def test_first_element_function(first_element_fn: DeferredCall[Array]) -> None:
  assert not jxs.is_permutation_invariant(first_element_fn)


def test_last_element_function(last_element_fn: DeferredCall[Array]) -> None:
  assert not jxs.is_permutation_invariant(last_element_fn)


def test_weighted_sum_function(weighted_sum_fn: DeferredCall[Array]) -> None:
  assert not jxs.is_permutation_invariant(weighted_sum_fn)
