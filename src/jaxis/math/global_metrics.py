from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from jaxis.math.typing import RelL1, RelL2, RelLInf

if TYPE_CHECKING:
  from jaxis.common_typing import Array


def rel_l1(
  ref: Array,
  tgt: Array,
  *,
  eps: float = 1e-12,
) -> RelL1:
  """
  Compute the relative L1 norm between the reference and target arrays.
  """
  diff = jnp.ravel(tgt - ref)
  ref_flat = jnp.ravel(ref)
  diff_l1 = float(jnp.linalg.norm(diff, ord=1).item())
  ref_l1 = float(jnp.linalg.norm(ref_flat, ord=1).item())
  rel_l1_val = diff_l1 / (ref_l1 + eps)

  return RelL1(rel_l1_val)


def rel_l2(
  ref: Array,
  tgt: Array,
  *,
  eps: float = 1e-12,
) -> RelL2:
  """
  Compute the relative L2 norm between the reference and target arrays.
  """
  diff = jnp.ravel(tgt - ref)
  ref_flat = jnp.ravel(ref)
  diff_l2 = float(jnp.linalg.norm(diff, ord=2).item())
  ref_l2 = float(jnp.linalg.norm(ref_flat, ord=2).item())
  rel_l2_val = diff_l2 / (ref_l2 + eps)

  return RelL2(rel_l2_val)


def rel_linf(
  ref: Array,
  tgt: Array,
  *,
  eps: float = 1e-12,
) -> RelLInf:
  """
  Compute the relative L-infinity norm between the reference and target arrays.
  """
  diff = jnp.ravel(tgt - ref)
  ref_flat = jnp.ravel(ref)
  diff_linf = float(jnp.linalg.norm(diff, ord=jnp.inf).item())
  ref_linf = float(jnp.linalg.norm(ref_flat, ord=jnp.inf).item())
  rel_linf_val = diff_linf / (ref_linf + eps)

  return RelLInf(rel_linf_val)
