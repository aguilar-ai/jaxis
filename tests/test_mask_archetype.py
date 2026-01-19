import jax.numpy as jnp

from jaxis.archetype.mask import Mask
from jaxis.common_typing import Sentinel


def masked_sum(x, mask):
  """
  Simple mask-invariant function.
  x: (N, D), mask: (N,)
  """
  return x * mask[..., jnp.newaxis]


def test_mask_archetype_basic():
  N, D = 5, 3
  mask_archetype = Mask(masked_sum)
  mask_archetype.with_input_spec(
    [("x", (N, D), 0, Sentinel.DEFAULT), ("mask", (N,), 0, Sentinel.MASK)]
  ).with_output_spec(0)

  result = mask_archetype.verify()
  assert result.passed is True


def test_mask_archetype_fails_on_non_invariant():
  def non_invariant_fn(x, mask):
    # Output at each position depends on the sum of ALL x,
    # so changing a masked x will change unmasked outputs.
    return x + jnp.sum(x, axis=0, keepdims=True)

  N, D = 5, 3
  mask_archetype = Mask(non_invariant_fn)
  mask_archetype.with_input_spec(
    [("x", (N, D), 0, Sentinel.DEFAULT), ("mask", (N,), 0, Sentinel.MASK)]
  ).with_output_spec(0)

  result = mask_archetype.verify()
  assert result.passed is False
