"""
# Testing Regime: Equivariant Permutability

Should PASS:

- Scalar functions applied elementwise: $x \to x^2$, $x \to \sin(x)$, $x \to \text{relu}(x)$
- Identity
- Any function satisfying element-wise independence (it implies equivariance)

Should FAIL:

- Sorting (canonical counterexample)
- Cumulative sum/product
- Position-dependent operations: $x \to [2 \cdot x_1, x_2, x_3, \dots]$
- Reverse: $x \to x[::-1]$
"""

import jaxis.semantics as jxs
from jaxis.common_typing import Array
from jaxis.fn import DeferredCall

# Should PASS


def test_identity_function(identity_fn: DeferredCall[Array]) -> None:
  assert jxs.is_permutation_equivariant(identity_fn)


def test_square_function(square_fn: DeferredCall[Array]) -> None:
  assert jxs.is_permutation_equivariant(square_fn)


def test_sin_function(sin_fn: DeferredCall[Array]) -> None:
  assert jxs.is_permutation_equivariant(sin_fn)


def test_relu_function(relu_fn: DeferredCall[Array]) -> None:
  assert jxs.is_permutation_equivariant(relu_fn)


# Should FAIL


def test_matmul_function(matmul_fn: DeferredCall[Array]) -> None:
  assert not jxs.is_permutation_equivariant(matmul_fn)


def test_sort_function(sort_fn: DeferredCall[Array]) -> None:
  assert not jxs.is_permutation_equivariant(sort_fn)


def test_cumsum_function(cumsum_fn: DeferredCall[Array]) -> None:
  assert not jxs.is_permutation_equivariant(cumsum_fn)


def test_cumprod_function(cumprod_fn: DeferredCall[Array]) -> None:
  assert not jxs.is_permutation_equivariant(cumprod_fn)
