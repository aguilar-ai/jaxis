from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import warnings

from jaxis.math.typing import (
  LeakageEnergyRatio,
  OffTargetMaxAbs,
  OffTargetRatio,
  OnTargetMag,
)

if TYPE_CHECKING:
  from jaxis.common_typing import Array


def off_target_ratio(
  delta_on: Array,
  delta_off: Array,
  *,
  eps: float = 1e-12,
) -> OffTargetRatio:
  """
  Compute the ratio of the maximum absolute off-target delta to the maximum absolute
  on-target delta.
  """
  if delta_off.size == 0:
    warnings.warn(
      "off_target_ratio() received an empty off-target region; returning 0 and skipping check.",
      UserWarning,
    )
    max_off = 0.0
  else:
    max_off = float(jnp.max(jnp.abs(delta_off)).item())

  if delta_on.size == 0:
    warnings.warn(
      "off_target_ratio() received an empty on-target region; returning 0 and skipping check.",
      UserWarning,
    )
    max_on = 0.0
  else:
    max_on = float(jnp.max(jnp.abs(delta_on)).item())

  ratio = max_off / (max_on + eps)
  return OffTargetRatio(ratio)


def leakage_energy_ratio(
  delta_on: Array,
  delta_off: Array,
  *,
  eps: float = 1e-12,
) -> LeakageEnergyRatio:
  """
  Compute the ratio of the L2 norm of the off-target delta to the L2 norm of the
  on-target delta.
  """
  norm_off = float(jnp.linalg.norm(delta_off).item())
  norm_on = float(jnp.linalg.norm(delta_on).item())
  ratio = norm_off / (norm_on + eps)
  return LeakageEnergyRatio(ratio)


def off_target_max_abs(
  delta_off: Array,
) -> OffTargetMaxAbs:
  """
  Compute the maximum absolute off-target delta.
  """
  if delta_off.size == 0:
    warnings.warn(
      "off_target_max_abs() received an empty off-target region; returning 0 and skipping check.",
      UserWarning,
    )
    return OffTargetMaxAbs(0.0)
  max_off = float(jnp.max(jnp.abs(delta_off)).item())
  return OffTargetMaxAbs(max_off)


def on_target_mag(
  delta_on: Array,
) -> OnTargetMag:
  """
  Compute the L2 norm of the on-target delta (or absolute value if scalar).
  """
  if delta_on.size == 0:
    warnings.warn(
      "on_target_mag() received an empty on-target region; returning 0 and skipping check.",
      UserWarning,
    )
    return OnTargetMag(0.0)
  mag = float(jnp.linalg.norm(delta_on).item())
  return OnTargetMag(mag)
