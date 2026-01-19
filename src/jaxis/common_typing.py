from __future__ import annotations

import enum
from typing import NamedTuple, TypeAlias

import jax

from jaxis.dsl.ast import Expr
from jaxis.math.typing import (
  CosineDistance,
  FractionOver1,
  JSDivergence,
  KendallTauDistance,
  KLDivergence,
  LeakageEnergyRatio,
  MaxAbsDelta,
  MaxNormalizedViolation,
  MeanAbsErr,
  NumInf,
  NumNaN,
  NumNonFinite,
  OffTargetMaxAbs,
  OffTargetRatio,
  OnTargetMag,
  Quantile,
  RelL1,
  RelL2,
  RelLInf,
  RootMeanSquareError,
  TopKOverlap,
)

Array: TypeAlias = jax.Array
AxisIndex: TypeAlias = int


@enum.unique
class Sentinel(enum.Enum):
  """A sentinel value to mark an axis for a specific treatment by a semantic property
  testing function."""

  DEFAULT = enum.auto()
  "Mark an axis as the default axis."

  MASK = enum.auto()
  "Mark an axis as a mask."


InputSpec: TypeAlias = (
  # arg_name and shape
  tuple[str, tuple[int, ...]] 
  # arg_name, shape, axis
  | tuple[str, tuple[int, ...], int | None]
  # arg_name, shape, axis, sentinel
  | tuple[str, tuple[int, ...], int | None, Sentinel | None]
)  # fmt: off
"""(arg_name, shape, axis, sentinel) - The input specification for a deferred call.

- `arg_name`: The name of the argument that the semantic property will be evaluated on.
- `shape`: The shape of the input array.
- `axis`: The input axis that the semantic property will be evaluated on.
- `sentinel`: The sentinel value for the input array.
"""


class InputSpec_(NamedTuple):
  arg_name: str
  """The name of the argument that the semantic property will be evaluated on."""
  shape: tuple[int, ...]
  """The shape of the input array."""
  axis: AxisIndex
  """The input axis that the semantic property will be evaluated on.
   If none, the whole array is considered.
  """
  sentinel: Sentinel = Sentinel.DEFAULT
  """The sentinel value for the input array.
  """

  @staticmethod
  def _from_user_facing_spec(spec: InputSpec) -> "InputSpec_":
    match spec:
      case (arg_name, shape):
        return InputSpec_(arg_name, shape, axis=-1)
      case (arg_name, shape, axis):
        return InputSpec_(
          arg_name,
          shape,
          axis=axis if axis is not None else -1,
        )
      case (arg_name, shape, axis, sentinel):
        return InputSpec_(
          arg_name,
          shape,
          axis=axis if axis is not None else -1,
          sentinel=sentinel if sentinel is not None else Sentinel.DEFAULT,
        )
      case _:
        raise ValueError(f"Invalid input spec: {spec}")

  @staticmethod
  def from_user_facing_spec(spec: InputSpec | list[InputSpec]) -> list["InputSpec_"]:
    if isinstance(spec, list):
      return [InputSpec_._from_user_facing_spec(s) for s in spec]
    return [InputSpec_._from_user_facing_spec(spec)]


OutputSpec: TypeAlias = (
  # axis
  (int | None)
  # axis, tuple_index
  | tuple[int, int]
)  # fmt: off
"""
(axis, tuple_index) - The output specification for a deferred call.
"""


class OutputSpec_(NamedTuple):
  axis: AxisIndex
  """The output axis that the semantic property will be evaluated on.
   If none, the whole array is considered.
  """
  tuple_i: int | None = None
  """
  If the function returns a tuple, the index of the tuple element 
  to evaluate the semantic property on. If none, it is implied that
  the function returns a single value.
  """

  @staticmethod
  def from_user_facing_spec(
    spec: OutputSpec, *, input_axis: AxisIndex | None = None
  ) -> "OutputSpec_":
    match spec:
      case (axis, tuple_i):
        return OutputSpec_(AxisIndex(axis), tuple_i)
      case None:
        if input_axis is None:
          raise ValueError("Input axis is required when output axis is None")
        return OutputSpec_(input_axis)
      case axis:
        return OutputSpec_(AxisIndex(axis))


class TestState(NamedTuple):
  """
  The result of a multi-trial semantic property test.

  Attributes:
    passed: True if all trials satisfied the test specification.
    test_spec: The metric specification used for the test.
    rng: The random key used for the trial that failed (if any).
    trial_index: The index of the trial that failed (if any).
    ... (metrics): Actual values for each metric computed during the last trial.
  """

  passed: bool

  test_spec: Expr

  quantile: Quantile | None = None
  max_normalized_violation: MaxNormalizedViolation | None = None
  fraction_over_1: FractionOver1 | None = None
  mean_abs_err: MeanAbsErr | None = None
  root_mean_square_error: RootMeanSquareError | None = None
  max_abs_delta: MaxAbsDelta | None = None
  rel_l1: RelL1 | None = None
  rel_l2: RelL2 | None = None
  rel_l_inf: RelLInf | None = None
  cosine_distance: CosineDistance | None = None
  kl_divergence: KLDivergence | None = None
  js_divergence: JSDivergence | None = None
  kendall_tau_distance: KendallTauDistance | None = None
  top_k_overlap: TopKOverlap | None = None
  num_non_finite: NumNonFinite | None = None
  num_nan: NumNaN | None = None
  num_inf: NumInf | None = None
  off_target_ratio: OffTargetRatio | None = None
  leakage_energy_ratio: LeakageEnergyRatio | None = None
  off_target_max_abs: OffTargetMaxAbs | None = None
  on_target_mag: OnTargetMag | None = None

  rng: Array | None = None
  trial_index: int | None = None
