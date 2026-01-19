"""
# Equivariant Permutability

A function is considered equivariant permutable if it behaves consistently with respect to
permutations of its inputs — meaning if you permute the inputs and then apply the function,
you get the same result as applying the function first and then permuting the outputs in a
corresponding way. More formally, a function $f$ is equivariant to permutations if for any
permutation $\pi$, we have $f(\pi(x)) = \pi(f(x))$. The function "commutes" with the permutation
operation.

This module simply provides functions to check if a given function is equivariant
permutable.
"""

from typing import TypeVar

import jax
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
from jaxis.exceptions import OutputAxisMissingError
from jaxis.fn.caller import DeferredCall
from jaxis.semantics._spec_utils import (
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
  & (M.REL_L2 <= 1e-4)
)


def is_permutation_equivariant(
  fn: DeferredCall[Y],
  *,
  rng: Array = jax.random.key(DEFAULT_RNG_SEED),
  test_spec: Expr = DEFAULT_SPEC,
  atol: float = DEFAULT_ATOL,
  rtol: float = DEFAULT_RTOL,
  dtype: DTypeLike = DEFAULT_DTYPE,
) -> bool:
  """
  Checks if a function satisfies Permutation Equivariance.

  A function is permutation equivariant if permuting the inputs and then applying
  the function yields the same result as applying the function first and then
  permuting the outputs in a corresponding way: $f(\pi(x)) = \pi(f(x))$.

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
  x_key, pi_key = jax.random.split(rng, 2)

  input_spec = next(
    input_spec for input_spec in fn.input_spec if input_spec.sentinel == Sentinel.DEFAULT
  )

  x = jax.random.normal(x_key, input_spec.shape, dtype=dtype)  # pyright: ignore[reportUnknownMemberType]
  pi_x = jax.random.permutation(pi_key, x, axis=int(input_spec.axis))  # type: ignore

  fn_pi_x = fn.with_generated_input(input_spec.arg_name, pi_x)
  f_pi_x = fn_pi_x.call_and_retrieve_output()

  fn_x = fn.with_generated_input(input_spec.arg_name, x)
  f_x = fn_x.call_and_retrieve_output()

  if fn.output_spec.axis >= len(f_x.shape):
    raise OutputAxisMissingError(int(fn.output_spec.axis), f_x.shape)

  pi_axis: int = int(fn.output_spec.axis)
  pi_f_x = jax.random.permutation(pi_key, f_x, axis=pi_axis)  # type: ignore

  needed = metrics_in_expr(test_spec)
  metrics = compute_region_metrics(
    needed,
    ref=pi_f_x,
    tgt=f_pi_x,
    atol=atol,
    rtol=rtol,
  )

  return bool(evaluate_spec(test_spec, metrics))


def is_permutation_equivariant_trials(
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
  Repeats the Permutation Equivariance experiment over multiple trials.

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
  # TODO: Use jax.vmap to speed up the computation. Blocked by DeferredCall not being a
  # valid JAX PyTree.
  for i, r in enumerate(rngs):
    if not is_permutation_equivariant(
      fn,
      rng=r,
      test_spec=test_spec,
      atol=atol,
      rtol=rtol,
      dtype=dtype,
    ):
      return TestState(passed=False, test_spec=test_spec, rng=r, trial_index=i)
  return TestState(passed=True, test_spec=test_spec, rng=None, trial_index=None)
