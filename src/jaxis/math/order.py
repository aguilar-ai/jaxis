from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from jaxis.math.typing import KendallTauDistance, TopKOverlap

if TYPE_CHECKING:
  from jaxis.common_typing import Array


def topk_overlap(
  ref: Array,
  tgt: Array,
  k: int,
) -> TopKOverlap:
  """
  Compute the top-k overlap (Jaccard similarity) between the reference and target scores.
  """
  # We assume scores are for the last dimension
  ref_topk = jnp.argsort(ref, axis=-1)[..., -k:]
  tgt_topk = jnp.argsort(tgt, axis=-1)[..., -k:]

  # Count overlap. Since these are indices, we can use sets or just check membership.
  # In JAX, we can use broadcasting to find common elements.
  # ref_topk: (..., k), tgt_topk: (..., k)
  # Broad cast to (..., k, 1) and (..., 1, k)
  match = ref_topk[..., :, jnp.newaxis] == tgt_topk[..., jnp.newaxis, :]
  # match: (..., k, k)
  overlap_count = jnp.sum(jnp.any(match, axis=-1), axis=-1)
  # overlap_count: (...)

  # Jaccard = overlap / (k + k - overlap)
  jaccard = overlap_count / (2 * k - overlap_count)

  return TopKOverlap(jnp.mean(jaccard).item())


def kendall_tau_distance(
  ref: Array,
  tgt: Array,
) -> KendallTauDistance:
  """
  Compute the Kendall tau distance between the reference and target arrays.
  """
  # Simplified O(N^2) implementation for JAX
  # For larger arrays, this might be slow, but for property tests it's usually fine.
  ref_flat = ref.ravel()
  tgt_flat = tgt.ravel()

  n = ref_flat.shape[0]
  if n < 2:
    return KendallTauDistance(0.0)

  # Get all pairs (i, j) where i < j
  i, j = jnp.triu_indices(n, k=1)

  ref_diff = ref_flat[i] - ref_flat[j]
  tgt_diff = tgt_flat[i] - tgt_flat[j]

  concordant = jnp.sum(jnp.sign(ref_diff) == jnp.sign(tgt_diff))
  discordant = jnp.sum(jnp.sign(ref_diff) != jnp.sign(tgt_diff))

  tau = (concordant - discordant) / (n * (n - 1) / 2)
  distance = 1.0 - tau

  return KendallTauDistance(distance.item())
