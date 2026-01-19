"""
# Triangular Dependence (Causal, Temporal Dependence)

Triangular dependence means that each output in a sequence can only "see" inputs that came
before it or at the same position — never inputs that come later. If you're computing the
fifth output, it can depend on inputs one through five, but not on inputs six, seven, and
so on. Information flows forward through the sequence, never backward.

More precisely, for a function mapping a sequence of $n$ inputs to $n$ outputs, triangular
dependence requires that output $i$ is a function only of inputs $1$ through $i$. Changing input
$j$ can only affect outputs at positions $j$ or later — it cannot retroactively influence
earlier outputs. This is the property that makes autoregressive generation possible: you
can compute outputs one at a time, left to right, without needing to know future inputs.

Formally, consider the Jacobian matrix $J$ where entry $J_{ij} = \partial \text{output}_i / \partial \text{input}_j$. Triangular
dependence requires that $J_{ij} = 0$ whenever $j > i$ — the Jacobian is lower-triangular.
Equivalently, if we denote the function as $f$ and let $x_{\leq i}$ represent the prefix of inputs
up to position $i$, then triangular dependence means $\text{output}_i = g_i(x_{\leq i})$ for some
function $g_i$ that depends only on that prefix. This structure is preserved under
composition: composing two triangular-dependent functions yields another
triangular-dependent function.

Testing for this is quite simple: you perturb inputs at position j and verify that outputs
at positions i < j remain unchanged.
"""

from __future__ import annotations

from typing import Callable, TypeVar, cast

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
from jaxis.exceptions import InputShapeTooSmallError
from jaxis.fn.caller import DeferredCall
from jaxis.semantics._spec_utils import (
  compute_leakage_metrics,
  compute_region_metrics,
  evaluate_spec,
  metrics_in_expr,
)

Y = TypeVar("Y")


DEFAULT_SPEC: Expr = (
  (M.NUM_NONFINITE.eq(0))
  & (M.P99 <= 1.0)
  & (M.FRACTION_OVER_1 <= 0.01)
  & (M.MAX <= 3.0)
  & (M.OFF_TARGET_MAX_ABS <= 1e-6)
  & (M.LEAKAGE_ENERGY_RATIO <= 1e-3)
)


def is_triangular_dependent(
  fn: DeferredCall[Y],
  *,
  rng: Array = jax.random.key(DEFAULT_RNG_SEED),
  input_to_output_pos: Callable[[int], int] = lambda j: j,
  test_spec: Expr = DEFAULT_SPEC,
  atol: float = DEFAULT_ATOL,
  rtol: float = DEFAULT_RTOL,
  dtype: DTypeLike = DEFAULT_DTYPE,
) -> bool:
  """
  Checks if a function satisfies Triangular Dependence (Causal/Temporal Dependence).

  Triangular dependence means that each output in a sequence only depends on inputs
  at its own position or earlier. Information flows forward, never backward.

  Args:
    fn: The deferred call to evaluate.
    rng: Random number generator key.
    input_to_output_pos: Maps input index to output index if sequence lengths differ.
    test_spec: Metric specification to decide pass/fail.
    atol: Absolute tolerance for metric evaluation.
    rtol: Relative tolerance for metric evaluation.
    dtype: Data type for random inputs.

  Returns:
    True if the function is triangular dependent.
  """
  x_key, j_key, perturb_key = cast(tuple[Array, Array, Array], jax.random.split(rng, 3))

  input_spec = next(
    input_spec for input_spec in fn.input_spec if input_spec.sentinel == Sentinel.DEFAULT
  )

  axis_in = input_spec.axis
  shape_in = input_spec.shape

  axis_out = fn.output_spec.axis

  if shape_in[axis_in] - 1 == 0:
    raise InputShapeTooSmallError(shape_in, axis_in)

  x = jax.random.normal(x_key, shape_in, dtype=dtype)  # type: ignore

  # 1: Picking a random j to perturb along the input axis
  j = jax.random.randint(j_key, (), minval=1, maxval=shape_in[axis_in], dtype=jnp.int32)  # type: ignore
  j = j.item()

  # 2: Creating perturbation by perturbing all values of x along the input axis for index >= j
  slices_x_ = [slice(None)] * len(shape_in)
  slices_x_[axis_in] = slice(j, None)
  slices_x_ = tuple(slices_x_)
  perturbation_shape = list(shape_in)
  perturbation_shape[axis_in] = shape_in[axis_in] - j
  perturbation_shape = tuple(perturbation_shape)
  perturbation = jax.random.normal(perturb_key, perturbation_shape, dtype=dtype)  # pyright: ignore[reportUnknownMemberType]

  # 3: Creating x' by perturbing all values along the input axis for index >= j
  x_ = x.at[slices_x_].set(perturbation)

  # 4: Calculating f(x) and f(x')
  fn_x = fn.with_generated_input(input_spec.arg_name, x)
  f_x = fn_x.call_and_retrieve_output()

  fn_x_ = fn.with_generated_input(input_spec.arg_name, x_)
  f_x_ = fn_x_.call_and_retrieve_output()

  # 5: Checking if f(x) = f(x') for all i < j
  slices_f_x = [slice(None)] * len(f_x.shape)
  slices_f_x[axis_out] = slice(None, input_to_output_pos(j))
  slices_f_x = tuple(slices_f_x)
  f_x_lt_j = f_x[slices_f_x]
  f_x__lt_j = f_x_[slices_f_x]

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
        ref=f_x_lt_j,
        tgt=f_x__lt_j,
        atol=atol,
        rtol=rtol,
      )
    )

  if leakage_needed:
    delta = f_x_ - f_x
    # off-target: prefix outputs that must not change
    delta_off = delta[slices_f_x]
    # on-target: future outputs allowed to change
    slices_future = list(slices_f_x)
    slices_future[axis_out] = slice(input_to_output_pos(j), None)
    delta_on = delta[tuple(slices_future)]
    metrics.update(
      compute_leakage_metrics(
        leakage_needed,
        delta_on=delta_on,
        delta_off=delta_off,
      )
    )

  return bool(evaluate_spec(test_spec, metrics))


def is_triangular_dependent_trials(
  fn: DeferredCall[Y],
  trials: int = DEFAULT_TRIALS,
  *,
  rng: Array = jax.random.key(DEFAULT_RNG_SEED),
  input_to_output_pos: Callable[[int], int] = lambda j: j,
  test_spec: Expr = DEFAULT_SPEC,
  atol: float = DEFAULT_ATOL,
  rtol: float = DEFAULT_RTOL,
  dtype: DTypeLike = DEFAULT_DTYPE,
) -> TestState:
  """
  Repeats the Triangular Dependence experiment over multiple trials.

  Args:
    fn: The deferred call to evaluate.
    trials: Number of randomized trials to perform.
    rng: Random number generator key.
    input_to_output_pos: Maps input index to output index if sequence lengths differ.
    test_spec: Metric specification to decide pass/fail.
    atol: Absolute tolerance for metric evaluation.
    rtol: Relative tolerance for metric evaluation.
    dtype: Data type for random inputs.

  Returns:
    A TestState showing whether every trial passed.
  """
  rngs = jax.random.split(rng, trials)
  for i, r in enumerate(rngs):
    if not is_triangular_dependent(
      fn,
      rng=r,
      input_to_output_pos=input_to_output_pos,
      test_spec=test_spec,
      atol=atol,
      rtol=rtol,
      dtype=dtype,
    ):
      return TestState(passed=False, test_spec=test_spec, rng=r, trial_index=i)
  return TestState(passed=True, test_spec=test_spec, rng=None, trial_index=None)
