from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from jaxis.math.typing import CosineDistance

if TYPE_CHECKING:
  from jaxis.common_typing import Array


def cosine_distance(
  ref: Array,
  tgt: Array,
) -> CosineDistance:
  """
  Compute the cosine distance between the reference and target arrays.
  """
  flattened_ref = jnp.reshape(ref, (-1,))
  flattened_tgt = jnp.reshape(tgt, (-1,))

  dot_product = float(jnp.dot(flattened_ref, flattened_tgt).item())

  norm_ref = float(jnp.linalg.norm(flattened_ref).item())
  norm_tgt = float(jnp.linalg.norm(flattened_tgt).item())

  cosine_similarity = dot_product / (norm_ref * norm_tgt)
  cosine_dist = 1.0 - cosine_similarity

  return CosineDistance(cosine_dist)
