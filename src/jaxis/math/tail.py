from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import warnings

from jaxis.constants import DEFAULT_ATOL, DEFAULT_RTOL
from jaxis.math.typing import FractionOver1, MaxNormalizedViolation, Quantile

if TYPE_CHECKING:
  from jaxis.common_typing import Array


def quantile(
  ref: Array,
  tgt: Array,
  quantile: float,
  *,
  atol: float = DEFAULT_ATOL,
  rtol: float = DEFAULT_RTOL,
  eps: float = 1e-12,
) -> Quantile:
  """
  Compute the quantile of the absolute difference between the reference and target
  arrays.
  """
  if ref.size == 0:
    warnings.warn(
      "quantile() received an empty region; returning 0 and skipping check.",
      UserWarning,
    )
    return Quantile(0.0)
  diff = jnp.abs(tgt - ref)
  normalized_violation = diff / (atol + rtol * jnp.abs(ref) + eps)

  return Quantile(jnp.quantile(normalized_violation, quantile).item())


def max_normalized_violation(
  ref: Array,
  tgt: Array,
  *,
  atol: float = DEFAULT_ATOL,
  rtol: float = DEFAULT_RTOL,
  eps: float = 1e-12,
) -> MaxNormalizedViolation:
  """
  Compute the maximum normalized violation between the reference and target arrays.
  """
  if ref.size == 0:
    warnings.warn(
      "max_normalized_violation() received an empty region; returning 0 and skipping check.",
      UserWarning,
    )
    return MaxNormalizedViolation(0.0)
  diff = jnp.abs(tgt - ref)
  normalized_violation = diff / (atol + rtol * jnp.abs(ref) + eps)

  return MaxNormalizedViolation(jnp.max(normalized_violation).item())


def fraction_over_1(
  ref: Array,
  tgt: Array,
  *,
  atol: float = DEFAULT_ATOL,
  rtol: float = DEFAULT_RTOL,
  eps: float = 1e-12,
) -> FractionOver1:
  """
  Compute the fraction of elements in the reference and target arrays that have a normalized violation greater than 1.
  """
  if ref.size == 0:
    warnings.warn(
      "fraction_over_1() received an empty region; returning 0 and skipping check.",
      UserWarning,
    )
    return FractionOver1(0.0)
  diff = jnp.abs(tgt - ref)
  normalized_violation = diff / (atol + rtol * jnp.abs(ref) + eps)

  return FractionOver1(
    normalized_violation[normalized_violation > 1].size / normalized_violation.size
  )


def p95(
  ref: Array,
  tgt: Array,
  *,
  atol: float = DEFAULT_ATOL,
  rtol: float = DEFAULT_RTOL,
  eps: float = 1e-12,
) -> Quantile:
  """
  Compute the 95th percentile of the normalized violation.
  """
  return quantile(ref, tgt, 0.95, atol=atol, rtol=rtol, eps=eps)


def p99(
  ref: Array,
  tgt: Array,
  *,
  atol: float = DEFAULT_ATOL,
  rtol: float = DEFAULT_RTOL,
  eps: float = 1e-12,
) -> Quantile:
  """
  Compute the 99th percentile of the normalized violation.
  """
  return quantile(ref, tgt, 0.99, atol=atol, rtol=rtol, eps=eps)


def p999(
  ref: Array,
  tgt: Array,
  *,
  atol: float = DEFAULT_ATOL,
  rtol: float = DEFAULT_RTOL,
  eps: float = 1e-12,
) -> Quantile:
  """
  Compute the 99.9th percentile of the normalized violation.
  """
  return quantile(ref, tgt, 0.999, atol=atol, rtol=rtol, eps=eps)
