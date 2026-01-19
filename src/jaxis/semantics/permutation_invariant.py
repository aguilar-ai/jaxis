"""
# Invariant Permutability

A function is considered invariant permutable if permuting the inputs does not change the
output. More formally, a function $f$ is invariant permutable if for any permutation $\pi$, we
have $f(\pi(x)) = f(x)$.

This is a stronger property than equivariant permutability, which only requires that the
output is the same up to a permutation of the inputs. Examples of these include aggregate
statistics like mean, median, and mode.
"""

from __future__ import annotations

from typing import TypeVar, cast

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


def is_permutation_invariant(
  fn: DeferredCall[Y],
  *,
  rng: Array = jax.random.key(DEFAULT_RNG_SEED),
  test_spec: Expr = DEFAULT_SPEC,
  atol: float = DEFAULT_ATOL,
  rtol: float = DEFAULT_RTOL,
  dtype: DTypeLike = DEFAULT_DTYPE,
) -> bool:
  """
  Checks if a function satisfies Permutation Invariance.

  A function is permutation invariant if permuting the inputs does not change
  the output: $f(\pi(x)) = f(x)$.

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
  x_key, pi_key = cast(tuple[Array, Array], jax.random.split(rng, 2))

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

  needed = metrics_in_expr(test_spec)
  metrics = compute_region_metrics(
    needed,
    ref=f_x,
    tgt=f_pi_x,
    atol=atol,
    rtol=rtol,
  )

  return bool(evaluate_spec(test_spec, metrics))


def is_permutation_invariant_trials(
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
  Repeats the Permutation Invariance experiment over multiple trials.

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
    if not is_permutation_invariant(
      fn,
      rng=r,
      test_spec=test_spec,
      atol=atol,
      rtol=rtol,
      dtype=dtype,
    ):
      return TestState(passed=False, test_spec=test_spec, rng=r, trial_index=i)
  return TestState(passed=True, test_spec=test_spec, rng=None, trial_index=None)
