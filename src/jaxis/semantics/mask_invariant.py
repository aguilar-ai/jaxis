"""
# Mask Invariance

Mask invariance means that if you mark certain outputs as "don't care" or "not present,"
changing the values of those masked inputs won't affect the outputs for the unmasked
positions. You can put any garbage in the padded slots — zero, random noise, whatever —
and the real outputs stay the same. The function genuinely ignores what it's told to
ignore.

More formally, let $f(x, m)$ be a function that takes an input $x$ and a mask $m \in \{0,1\}^n$, and
returns an output $y$. Mask invariance requires that for all $i$ where $m_i = 1$, we have
$\partial y_i / \partial x_j = 0$ whenever $m_j = 0$. Equivalently, letting $S = \{j : m_j = 1\}$ denote the
support, there exists a function $g$ such that $y_S = g(x_S, S)$ — the outputs on the
support are fully determined by the inputs on the support. This property composes: if two
mask-invariant functions are chained with consistent masking, the result is
mask-invariant.

We test this by perturbing masked inputs and checking if the outputs for the unmasked
positions remain the same.
"""

from __future__ import annotations

from typing import TypeVar

import jax
import jax.numpy as jnp
from jax.typing import DTypeLike

from jaxis.common_typing import Array, Sentinel, TestState
from jaxis.constants import (
  DEFAULT_ATOL,
  DEFAULT_DTYPE,
  DEFAULT_RNG_SEED,
  DEFAULT_RTOL,
  DEFAULT_TRIALS,
)
from jaxis.dsl.ast import Expr
from jaxis.dsl.ast import Metric as M
from jaxis.exceptions import MaskShapeMismatchError
from jaxis.fn.caller import DeferredCall
from jaxis.semantics._spec_utils import (
  compute_leakage_metrics,
  compute_region_metrics,
  evaluate_spec,
  metrics_in_expr,
)

Y = TypeVar("Y")


# Default gate: strict tail metrics on unmasked outputs plus leakage caps.
DEFAULT_SPEC: Expr = (
  (M.NUM_NONFINITE.eq(0))
  & (M.P99 <= 1.0)
  & (M.FRACTION_OVER_1 <= 0.01)
  & (M.MAX <= 3.0)
  & (M.MAX_ABS_DELTA <= 1e-6)
  & (M.OFF_TARGET_MAX_ABS <= 1e-6)
  & (M.LEAKAGE_ENERGY_RATIO <= 1e-3)
)


def is_mask_invariant(
  fn: DeferredCall[Y],
  *,
  rng: Array = jax.random.key(DEFAULT_RNG_SEED),
  test_spec: Expr = DEFAULT_SPEC,
  atol: float = DEFAULT_ATOL,
  rtol: float = DEFAULT_RTOL,
  dtype: DTypeLike = DEFAULT_DTYPE,
) -> bool:
  """
  Checks if a function satisfies Mask Invariance.

  Mask invariance means that changing values of masked inputs does not affect
  outputs for the unmasked positions.

  Args:
    fn: The deferred call to evaluate. Must include both data and mask tensors.
    rng: Random number generator key.
    test_spec: Metric specification to decide pass/fail.
    atol: Absolute tolerance for metric evaluation.
    rtol: Relative tolerance for metric evaluation.
    dtype: Data type for random inputs.

  Returns:
    True if the function satisfies the property.
  """
  spec_i = next(
    input_spec for input_spec in fn.input_spec if input_spec.sentinel == Sentinel.DEFAULT
  )
  shape_i = spec_i.shape
  axis_i = spec_i.axis

  spec_m = next(
    input_spec for input_spec in fn.input_spec if input_spec.sentinel == Sentinel.MASK
  )
  shape_m = spec_m.shape
  axis_m = spec_m.axis

  spec_o = fn.output_spec
  axis_o = spec_o.axis

  if shape_i[axis_i] != shape_m[axis_m]:
    raise MaskShapeMismatchError(shape_i, shape_m, axis_i, axis_m)

  if shape_i[: len(shape_m)] != shape_m:
    raise MaskShapeMismatchError(shape_i, shape_m, axis_i, axis_m)

  keys = jax.random.split(rng, 3)
  x_key, m_key, perturb_key = keys

  # 1: Creating a random input and a random mask
  x = jax.random.normal(x_key, shape_i, dtype=dtype)  # pyright: ignore[reportUnknownMemberType]
  m = jax.random.bernoulli(m_key, shape=shape_m)  # pyright: ignore[reportUnknownMemberType]
  # Ensure at least one unmasked position so the checked region is non-empty.
  if not jnp.any(m):
    m = m.at[0].set(True)

  # 2: Changing a random subset of the masked inputs to a random value
  perturbation = jax.random.normal(perturb_key, shape_i, dtype=dtype)  # pyright: ignore[reportUnknownMemberType]
  x_ = jnp.where(m[..., None] if len(shape_i) > len(shape_m) else m, x, perturbation)

  # 3: Calculating the outputs for the original and perturbed inputs
  fn_x = fn.with_generated_inputs(
    **{
      spec_i.arg_name: x,
      spec_m.arg_name: m,
    }
  )
  f_x = fn_x.call_and_retrieve_output()

  fn_x_ = fn.with_generated_inputs(
    **{
      spec_m.arg_name: m,
      spec_i.arg_name: x_,
    }
  )
  f_x_ = fn_x_.call_and_retrieve_output()

  shape_o = f_x.shape

  if shape_o[: len(shape_m)] != shape_m:
    raise MaskShapeMismatchError(shape_o, shape_m, axis_o, axis_m)

  extra_dims = len(shape_o) - len(shape_m)
  m_expanded = m.reshape(shape_m + (1,) * extra_dims)
  m_broadcast = jnp.broadcast_to(m_expanded, shape_o)

  ref = f_x[m_broadcast]
  tgt = f_x_[m_broadcast]

  needed = metrics_in_expr(test_spec)
  leakage_needed = {
    metric
    for metric in needed
    if metric
    in {
      M.OFF_TARGET_RATIO,
      M.LEAKAGE_ENERGY_RATIO,
      M.OFF_TARGET_MAX_ABS,
      M.ON_TARGET_MAG,
    }
  }
  region_needed = needed - leakage_needed

  metrics: dict[M, float] = {}
  if region_needed:
    metrics.update(
      compute_region_metrics(
        region_needed,
        ref=ref,
        tgt=tgt,
        atol=atol,
        rtol=rtol,
      )
    )

  if leakage_needed:
    delta = f_x_ - f_x
    delta_off = delta[m_broadcast]  # unmasked outputs should stay same
    delta_on = delta[~m_broadcast]  # masked outputs allowed to change
    metrics.update(
      compute_leakage_metrics(
        leakage_needed,
        delta_on=delta_on,
        delta_off=delta_off,
      )
    )

  return bool(evaluate_spec(test_spec, metrics))


def is_mask_invariant_trials(
  fn: DeferredCall[Y],
  trials: int = DEFAULT_TRIALS,
  *,
  rng: Array = jax.random.key(DEFAULT_RNG_SEED),
  test_spec: Expr = DEFAULT_SPEC,
  atol: float = DEFAULT_ATOL,
  rtol: float = DEFAULT_RTOL,
  dtype: DTypeLike = DEFAULT_DTYPE,
) -> TestState:
  """
  Repeats the Mask Invariance experiment over multiple trials.

  Args:
    fn: The deferred call to evaluate.
    trials: Number of randomized trials to perform.
    rng: Random number generator key.
    test_spec: Metric specification to decide pass/fail.
    atol: Absolute tolerance for metric evaluation.
    rtol: Relative tolerance for metric evaluation.
    dtype: Data type for random inputs.

  Returns:
    A TestState showing whether every trial passed.
  """
  rngs = jax.random.split(rng, trials)
  for i, r in enumerate(rngs):
    if not is_mask_invariant(
      fn,
      rng=r,
      test_spec=test_spec,
      atol=atol,
      rtol=rtol,
      dtype=dtype,
    ):
      return TestState(passed=False, test_spec=test_spec, rng=r, trial_index=i)
  return TestState(passed=True, test_spec=test_spec, rng=None, trial_index=None)
