"""
# Element-wise Independence

Elementwise independence means that each output depends only on its corresponding
input, and changing one input affects only its own output and nothing else.

A function $f: X^n \to Y^n$ satisfies element-wise independence if there exists a function
$g: X \to Y$ such that:

$f(x_1, x_2, \dots, x_n) = (g(x_1), g(x_2), \dots, g(x_n))$

Or equivalently, $f(x)_i = g(x_i)$ for all $i$.

You can also think of this as "diagonal equivariance." The term "diagonal" refers to
the Jacobian of the function: for a function $f$ mapping $n$ inputs to $n$ outputs, the
Jacobian is an $n \times n$ matrix where the $(i, j)$ entry measures how output $i$ depends on
input $j$ ($\partial \text{output}_i / \partial \text{input}_j$). When each input only affects its own corresponding
output, all off-diagonal entries are zero — meaning output $i$ does not depend on input
$j$ when $i \neq j$. Only the diagonal entries can be non-zero.

The "equivariance" part highlights that this property is preserved under permutations:
if you permute the inputs, the outputs are permuted in the same way. The function
maintains the correspondence between input and output positions.

Thus, "diagonal equivariance" means both: (1) a diagonal dependency structure (no
cross-talk between coordinates) and (2) equivariant behavior under reordering.
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
from jaxis.fn.caller import DeferredCall
from jaxis.semantics._spec_utils import (
  compute_leakage_metrics,
  compute_region_metrics,
  evaluate_spec,
  metrics_in_expr,
)

Y = TypeVar("Y")


# Default gate: require low leakage plus tail-control on off-target slice.
DEFAULT_SPEC: Expr = (
  (M.NUM_NONFINITE.eq(0))
  & (M.LEAKAGE_ENERGY_RATIO <= 1e-3)
  & (M.OFF_TARGET_MAX_ABS <= 1e-6)
  & (M.P99 <= 1.0)
  & (M.FRACTION_OVER_1 <= 1e-3)
  & (M.MAX <= 2.0)
)


def is_elementwise_independent(
  fn: DeferredCall[Y],
  *,
  rng: Array = jax.random.key(DEFAULT_RNG_SEED),
  test_spec: Expr = DEFAULT_SPEC,
  atol: float = DEFAULT_ATOL,
  rtol: float = DEFAULT_RTOL,
  dtype: DTypeLike = DEFAULT_DTYPE,
) -> bool:
  """
  Checks if a function satisfies Elementwise Independence (Diagonal Equivariance).

  Elementwise Independence means that each output depends only on its corresponding
  input, and changing one input affects only its own output and nothing else.
  Also known as "Diagonal Equivariance."

  Args:
    fn: The deferred call to evaluate.
    rng: Random number generator key.
    test_spec: Metric specification to decide pass/fail.
    atol: Absolute tolerance for metric evaluation.
    rtol: Relative tolerance for metric evaluation.
    dtype: Data type for random inputs.

  Returns:
    True if the function satisfies the property.
  """
  x_key, i_key, perturb_key = jax.random.split(rng, 3)

  input_spec = next(
    input_spec for input_spec in fn.input_spec if input_spec.sentinel == Sentinel.DEFAULT
  )

  shape = input_spec.shape
  input_axis = input_spec.axis
  output_axis = fn.output_spec.axis

  # Generate random input
  x = jax.random.normal(x_key, shape, dtype=dtype)  # pyright: ignore[reportUnknownMemberType]

  # Select a random index along the input axis to perturb
  n = shape[input_axis]
  i = jax.random.randint(i_key, (), minval=0, maxval=n, dtype=jnp.int32)  # pyright: ignore[reportUnknownMemberType]
  i = i.item()

  # Create perturbed input: replace slice at index i with random values
  slice_shape = list(shape)
  slice_shape[input_axis] = 1
  perturbation = jax.random.normal(perturb_key, slice_shape, dtype=dtype)  # pyright: ignore[reportUnknownMemberType]

  # Build index array for put_along_axis
  indices = jnp.expand_dims(i, axis=tuple(range(x.ndim)))
  indices = jnp.broadcast_to(indices, slice_shape)

  x_ = jnp.put_along_axis(x, indices, perturbation, axis=input_axis, inplace=False)  # type: ignore

  fn_x = fn.with_generated_input(input_spec.arg_name, x)
  f_x = fn_x.call_and_retrieve_output()

  # Element-wise independence requires a 1:1 mapping along the target axis.
  if f_x.ndim == 0:
    return False

  if not -f_x.ndim <= output_axis < f_x.ndim:
    return False

  input_axis_norm = input_axis % x.ndim
  output_axis_norm = output_axis % f_x.ndim

  if f_x.shape[output_axis_norm] != x.shape[input_axis_norm]:
    return False

  fn_x_ = fn.with_generated_input(input_spec.arg_name, x_)
  f_x_ = fn_x_.call_and_retrieve_output()

  # Compare all outputs except at index i
  f_x_rest = jnp.delete(f_x, i, axis=output_axis_norm)
  f_x_rest_ = jnp.delete(f_x_, i, axis=output_axis_norm)

  # Metrics: off-target equality + leakage ratios using deltas.
  off_target_metrics = metrics_in_expr(test_spec)
  leakage_needed = {
    m
    for m in off_target_metrics
    if m
    in {
      M.OFF_TARGET_RATIO,
      M.LEAKAGE_ENERGY_RATIO,
      M.OFF_TARGET_MAX_ABS,
      M.ON_TARGET_MAG,
    }
  }
  region_needed = off_target_metrics - leakage_needed

  metrics: dict[M, float] = {}
  if region_needed:
    metrics.update(
      compute_region_metrics(
        region_needed,
        ref=f_x_rest,
        tgt=f_x_rest_,
        atol=atol,
        rtol=rtol,
      )
    )

  if leakage_needed:
    delta = f_x_ - f_x
    # off-target excludes the perturbed index
    delta_off = jnp.delete(delta, i, axis=output_axis_norm)
    delta_on = jnp.take(delta, i, axis=output_axis_norm)
    metrics.update(
      compute_leakage_metrics(
        leakage_needed,
        delta_on=delta_on,
        delta_off=delta_off,
      )
    )

  return bool(evaluate_spec(test_spec, metrics))


def is_elementwise_independent_trials(
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
  Repeats the Elementwise Independence experiment over multiple trials.

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
    if not is_elementwise_independent(
      fn,
      rng=r,
      test_spec=test_spec,
      atol=atol,
      rtol=rtol,
      dtype=dtype,
    ):
      return TestState(passed=False, test_spec=test_spec, rng=r, trial_index=i)
  return TestState(passed=True, test_spec=test_spec, rng=None, trial_index=None)
