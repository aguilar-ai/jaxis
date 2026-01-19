from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import warnings

from jaxis.math.typing import MaxAbsDelta, MeanAbsErr, RootMeanSquareError

if TYPE_CHECKING:
  from jaxis.common_typing import Array


def mean_abs_err(
  ref: Array,
  tgt: Array,
) -> MeanAbsErr:
  """
  Compute the mean absolute error between the reference and target arrays.
  """
  diff = jnp.abs(tgt - ref)
  if diff.size == 0:
    warnings.warn(
      "mean_abs_err() received an empty region; returning 0 and skipping check.",
      UserWarning,
    )
    return MeanAbsErr(0.0)
  mean_diff = jnp.mean(diff)
  return MeanAbsErr(mean_diff.item())


def rmse(
  ref: Array,
  tgt: Array,
) -> RootMeanSquareError:
  """
  Compute the root mean square error between the reference and target arrays.
  """
  diff = tgt - ref
  if diff.size == 0:
    warnings.warn(
      "rmse() received an empty region; returning 0 and skipping check.",
      UserWarning,
    )
    return RootMeanSquareError(0.0)
  diff_squared = diff**2
  mean_diff_squared = jnp.mean(diff_squared)
  rmse = jnp.sqrt(mean_diff_squared)
  return RootMeanSquareError(rmse.item())


def max_abs_delta(
  ref: Array,
  tgt: Array,
) -> MaxAbsDelta:
  """
  Compute the maximum absolute delta between the reference and target arrays.
  """
  diff = jnp.abs(tgt - ref)
  if diff.size == 0:
    warnings.warn(
      "max_abs_delta() received an empty region; returning 0 and skipping check.",
      UserWarning,
    )
    return MaxAbsDelta(0.0)
  max_diff = jnp.max(diff)
  return MaxAbsDelta(max_diff.item())
