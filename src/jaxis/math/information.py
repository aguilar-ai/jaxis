from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import jax
import jax.numpy as jnp

from jaxis.math.typing import JSDivergence, KLDivergence

if TYPE_CHECKING:
  from jaxis.common_typing import Array


def kl_divergence(
  ref: Array,
  tgt: Array,
  *,
  normalization_fn: Callable[[Array], Array] = jax.nn.softmax,
) -> KLDivergence:
  """
  Compute the KL divergence between the reference and target arrays.
  """
  ref_normalized = normalization_fn(ref)
  tgt_normalized = normalization_fn(tgt)

  if ref_normalized.any() <= 0 or tgt_normalized.any() <= 0:
    raise ValueError("Reference and target arrays must have positive values.")

  kl_divergence = jnp.sum(ref_normalized * jnp.log(ref_normalized / tgt_normalized))

  return KLDivergence(kl_divergence.item())


def js_divergence(
  ref: Array,
  tgt: Array,
  *,
  normalization_fn: Callable[[Array], Array] = jax.nn.softmax,
) -> JSDivergence:
  """
  Compute the Jensen-Shannon divergence between the reference and target arrays.
  """
  p = normalization_fn(ref)
  q = normalization_fn(tgt)

  if p.any() <= 0 or q.any() <= 0:
    raise ValueError("Reference and target arrays must have positive values.")

  m = 0.5 * (p + q)

  kl_pm = jnp.sum(p * jnp.log(p / m))
  kl_qm = jnp.sum(q * jnp.log(q / m))

  js_divergence = 0.5 * (kl_pm + kl_qm)

  return JSDivergence(js_divergence.item())
