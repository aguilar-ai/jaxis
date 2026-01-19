"""
# Testing Regime: Element-wise Independence

Should PASS:

- Identity: x → x
- Scalar functions applied elementwise: x → x², x → sin(x), x → relu(x)
- Constant addition/multiplication: x → x + c, x → cx

Should FAIL:

- Sum/mean (output depends on all inputs)
- Softmax (normalization creates cross-dependencies)
- Cumulative sum
- Any pairwise interaction: x → x + mean(x)
"""

import jaxis.semantics as jxs
from jaxis.common_typing import Array
from jaxis.fn import DeferredCall

# Should PASS


def test_identity_function(identity_fn: DeferredCall[Array]) -> None:
  assert jxs.is_elementwise_independent(identity_fn)


def test_square_function(square_fn: DeferredCall[Array]) -> None:
  assert jxs.is_elementwise_independent(square_fn)


def test_sin_function(sin_fn: DeferredCall[Array]) -> None:
  assert jxs.is_elementwise_independent(sin_fn)


def test_relu_function(relu_fn: DeferredCall[Array]) -> None:
  assert jxs.is_elementwise_independent(relu_fn)


def test_add_constant_function(add_constant_fn: DeferredCall[Array]) -> None:
  assert jxs.is_elementwise_independent(add_constant_fn)


def test_multiply_constant_function(multiply_constant_fn: DeferredCall[Array]) -> None:
  assert jxs.is_elementwise_independent(multiply_constant_fn)


# Should FAIL


def test_sum_function(sum_fn: DeferredCall[Array]) -> None:
  assert not jxs.is_elementwise_independent(sum_fn)


def test_mean_function(mean_fn: DeferredCall[Array]) -> None:
  assert not jxs.is_elementwise_independent(mean_fn)


def test_softmax_function(softmax_fn: DeferredCall[Array]) -> None:
  assert not jxs.is_elementwise_independent(softmax_fn)


def test_cumsum_function(cumsum_fn: DeferredCall[Array]) -> None:
  assert not jxs.is_elementwise_independent(cumsum_fn)


def test_add_mean_function(add_mean_fn: DeferredCall[Array]) -> None:
  assert not jxs.is_elementwise_independent(add_mean_fn)
