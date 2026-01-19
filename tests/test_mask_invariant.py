"""
# Mask Invariance

Should PASS:

- Masked mean: average only over mask=1 positions
- Element-wise ops that output zero (or unchanged) at masked positions
- Proper attention with causal/padding masks

Should FAIL:

- Global softmax (normalization uses all values)
- Unmasked mean/sum
- Batch normalization (statistics from all elements)
- Any operation where masked inputs affect unmasked outputs
"""

import jaxis.semantics as jxs
from jaxis.common_typing import Array
from jaxis.fn import DeferredCall

# Should PASS


def test_masked_sum_function(masked_sum_fn: DeferredCall[Array]) -> None:
  assert jxs.is_mask_invariant(masked_sum_fn)


def test_masked_mean_function(masked_mean_fn: DeferredCall[Array]) -> None:
  assert jxs.is_mask_invariant(masked_mean_fn)


# Should FAIL


def test_unmasked_sum_function(unmasked_sum_fn: DeferredCall[Array]) -> None:
  assert not jxs.is_mask_invariant(unmasked_sum_fn)


def test_global_softmax_function(global_softmax_fn: DeferredCall[Array]) -> None:
  assert not jxs.is_mask_invariant(global_softmax_fn)


def test_batch_norm_function(batch_norm_fn: DeferredCall[Array]) -> None:
  assert not jxs.is_mask_invariant(batch_norm_fn)
