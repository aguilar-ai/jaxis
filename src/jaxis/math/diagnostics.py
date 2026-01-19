from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from jaxis.math.typing import NumInf, NumNaN, NumNonFinite

if TYPE_CHECKING:
  from jaxis.common_typing import Array


def num_nonfinite(
  tgt: Array,
) -> NumNonFinite:
  """
  Count of NaN or Inf in tgt.
  """
  count = jnp.sum(~jnp.isfinite(tgt))
  return NumNonFinite(int(count.item()))


def num_nan(
  tgt: Array,
) -> NumNaN:
  """
  Count of NaN in tgt.
  """
  count = jnp.sum(jnp.isnan(tgt))
  return NumNaN(int(count.item()))


def num_inf(
  tgt: Array,
) -> NumInf:
  """
  Count of +/-Inf in tgt.
  """
  count = jnp.sum(jnp.isinf(tgt))
  return NumInf(int(count.item()))
