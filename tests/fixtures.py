import jax
import jax.numpy as jnp
import pytest

from jaxis.common_typing import Array, Sentinel
from jaxis.fn import DeferredCall


@pytest.fixture
def identity_fn():
  def f(x: Array) -> Array: return x  # fmt: off

  return DeferredCall(f, input_spec=("x", (32, 32, 32)))


@pytest.fixture
def square_fn():
  def f(x: Array) -> Array: return x**2  # fmt: off

  return DeferredCall(f, input_spec=("x", (32, 32, 32)))


@pytest.fixture
def sin_fn():
  def f(x: Array) -> Array: return jnp.sin(x)  # fmt: off

  return DeferredCall(f, input_spec=("x", (32, 32, 32)))


@pytest.fixture
def relu_fn():
  def f(x: Array) -> Array: return jnp.maximum(x, 0)  # fmt: off

  return DeferredCall(f, input_spec=("x", (32, 32, 32)))


@pytest.fixture
def matmul_fn():
  def f(x: Array, y: Array) -> Array: return jnp.matmul(x, y)  # fmt: off

  return DeferredCall(f, input_spec=("x", (32, 32)), y=jnp.ones((32, 32)))


@pytest.fixture
def sort_fn():
  def f(x: Array) -> Array: return jnp.sort(x, axis=-1)  # fmt: off

  return DeferredCall(f, input_spec=("x", (32, 32, 32), -1))


@pytest.fixture
def cumsum_fn():
  def f(x: Array) -> Array: return jnp.cumsum(x, axis=-1)  # fmt: off

  return DeferredCall(f, input_spec=("x", (32, 32, 32), -1))


@pytest.fixture
def cumprod_fn():
  def f(x: Array) -> Array: return jnp.cumprod(x, axis=-1)  # fmt: off

  return DeferredCall(f, input_spec=("x", (32, 32, 32), -1))


@pytest.fixture
def add_constant_fn():
  def f(x: Array) -> Array: return x + 5.0  # fmt: off

  return DeferredCall(f, input_spec=("x", (32, 32, 32)))


@pytest.fixture
def multiply_constant_fn():
  def f(x: Array) -> Array: return x * 3.0  # fmt: off

  return DeferredCall(f, input_spec=("x", (32, 32, 32)))


@pytest.fixture
def sum_fn():
  def f(x: Array) -> Array: return jnp.sum(x, axis=-1, keepdims=True)  # fmt: off

  return DeferredCall(f, input_spec=("x", (32, 32, 32), -1))


@pytest.fixture
def mean_fn():
  def f(x: Array) -> Array: return jnp.mean(x, axis=-1, keepdims=True)  # fmt: off

  return DeferredCall(f, input_spec=("x", (32, 32, 32), -1))


@pytest.fixture
def softmax_fn():
  def f(x: Array) -> Array: return jax.nn.softmax(x, axis=-1)  # fmt: off

  return DeferredCall(f, input_spec=("x", (32, 32, 32), -1))


@pytest.fixture
def add_mean_fn():
  def f(x: Array) -> Array: return x + jnp.mean(x, axis=-1, keepdims=True)  # fmt: off

  return DeferredCall(f, input_spec=("x", (32, 32, 32), -1))


@pytest.fixture
def max_fn():
  def f(x: Array) -> Array: return jnp.max(x, axis=-1, keepdims=True)  # fmt: off

  return DeferredCall(f, input_spec=("x", (32, 32, 32), -1))


@pytest.fixture
def min_fn():
  def f(x: Array) -> Array: return jnp.min(x, axis=-1, keepdims=True)  # fmt: off

  return DeferredCall(f, input_spec=("x", (32, 32, 32), -1))


@pytest.fixture
def var_fn():
  def f(x: Array) -> Array: return jnp.var(x, axis=-1, keepdims=True)  # fmt: off

  return DeferredCall(f, input_spec=("x", (32, 32, 32), -1))


@pytest.fixture
def std_fn():
  def f(x: Array) -> Array: return jnp.std(x, axis=-1, keepdims=True)  # fmt: off

  return DeferredCall(f, input_spec=("x", (32, 32, 32), -1))


@pytest.fixture
def median_fn():
  def f(x: Array) -> Array: return jnp.median(x, axis=-1, keepdims=True)  # fmt: off

  return DeferredCall(f, input_spec=("x", (32, 32, 32), -1))


@pytest.fixture
def prod_fn():
  def f(x: Array) -> Array: return jnp.prod(x, axis=-1, keepdims=True)  # fmt: off

  return DeferredCall(f, input_spec=("x", (32, 32, 32), -1))


@pytest.fixture
def l1_norm_fn():
  def f(x: Array) -> Array: return jnp.sum(jnp.abs(x), axis=-1, keepdims=True)  # fmt: off

  return DeferredCall(f, input_spec=("x", (32, 32, 32), -1))


@pytest.fixture
def l2_norm_fn():
  def f(x: Array) -> Array: return jnp.sqrt(jnp.sum(x**2, axis=-1, keepdims=True))  # fmt: off

  return DeferredCall(f, input_spec=("x", (32, 32, 32), -1))


@pytest.fixture
def first_element_fn():
  def f(x: Array) -> Array: return x[..., 0:1]  # fmt: off

  return DeferredCall(f, input_spec=("x", (32, 32, 32), -1))


@pytest.fixture
def last_element_fn():
  def f(x: Array) -> Array: return x[..., -1:]  # fmt: off

  return DeferredCall(f, input_spec=("x", (32, 32, 32), -1))


@pytest.fixture
def weighted_sum_fn():
  def f(x: Array) -> Array:
    weights = jnp.arange(1, x.shape[-1] + 1, dtype=x.dtype)
    return jnp.sum(x * weights, axis=-1, keepdims=True)  # fmt: off

  return DeferredCall(f, input_spec=("x", (32, 32, 32), -1))


@pytest.fixture
def masked_sum_fn():
  def f(x: Array, mask: Array) -> Array:
    return x * mask[..., jnp.newaxis]

  return DeferredCall(
    f,
    input_spec=[
      ("x", (32, 32, 32), 1, Sentinel.DEFAULT),
      ("mask", (32, 32), 1, Sentinel.MASK),
    ],
  )


@pytest.fixture
def masked_mean_fn():
  def f(x: Array, mask: Array) -> Array:
    mask_expanded = mask[..., jnp.newaxis]
    return (x * mask_expanded) / (jnp.sum(mask_expanded, axis=-1, keepdims=True) + 1e-8)

  return DeferredCall(
    f,
    input_spec=[
      ("x", (32, 32, 32), 1, Sentinel.DEFAULT),
      ("mask", (32, 32), 1, Sentinel.MASK),
    ],
  )


@pytest.fixture
def unmasked_sum_fn():
  def f(x: Array, mask: Array) -> Array:
    # This ignores the mask and adds a global sum, making it non-invariant
    return x + jnp.sum(x, axis=(0, 1, 2), keepdims=True)

  return DeferredCall(
    f,
    input_spec=[
      ("x", (32, 32, 32), 1, Sentinel.DEFAULT),
      ("mask", (32, 32), 1, Sentinel.MASK),
    ],
  )


@pytest.fixture
def global_softmax_fn():
  def f(x: Array, mask: Array) -> Array:
    # Softmax over all dimensions, making it non-invariant
    return jax.nn.softmax(x, axis=(0, 1, 2))

  return DeferredCall(
    f,
    input_spec=[
      ("x", (32, 32, 32), 1, Sentinel.DEFAULT),
      ("mask", (32, 32), 1, Sentinel.MASK),
    ],
  )


@pytest.fixture
def batch_norm_fn():
  def f(x: Array, mask: Array) -> Array:
    # Standard batch norm ignores masks
    mean = jnp.mean(x, axis=(0, 1), keepdims=True)
    var = jnp.var(x, axis=(0, 1), keepdims=True)
    return (x - mean) / jnp.sqrt(var + 1e-5)

  return DeferredCall(
    f,
    input_spec=[
      ("x", (32, 32, 32), 1, Sentinel.DEFAULT),
      ("mask", (32, 32), 1, Sentinel.MASK),
    ],
  )
