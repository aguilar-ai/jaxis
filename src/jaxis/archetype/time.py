from __future__ import annotations

from typing import Any, Callable, TypeVar

from jaxis.archetype.common import Archetype, ArchetypeResult
from jaxis.common_typing import TestState
from jaxis.console import Console
from jaxis.dsl.ast import Expr
from jaxis.fn import DeferredCall
from jaxis.semantics import (
  is_permutation_equivariant_trials,
  is_triangular_dependent_trials,
)
from jaxis.semantics.permutation_equivariant import (
  DEFAULT_SPEC as PERMUTATION_EQUIVARIANT_SPEC,
)
from jaxis.semantics.triangular_dependent import DEFAULT_SPEC as TRIANGULAR_DEPENDENT_SPEC

Y = TypeVar("Y")


class Time(Archetype[Y]):
  """
  The Time archetype for causal or prefix-safe axes.

  This archetype verifies Triangular Dependence: each output in a sequence only
  depends on inputs at its own position or earlier.
  """

  def __init__(
    self,
    fn: Callable[..., Y],
    *,
    permutation_equivariant: bool = False,
    triangular_spec: Expr | None = None,
    permutation_spec: Expr | None = None,
  ):
    """
    Initializes a time archetype.

    Parameters:
    - fn: The function to test for semantic properties typical of time operations and axes.
    - permutation_equivariant: Whether to test for permutation equivariance. Defaults to False.
    - triangular_spec: Optional Metric DSL expression for the triangular dependence gate.
    - permutation_spec: Optional Metric DSL expression for the optional permutation gate.
    """
    super().__init__(fn)
    self.permutation_equivariant = permutation_equivariant
    self.triangular_spec = triangular_spec
    self.permutation_spec = permutation_spec

  def _describe_header(self, c: Console, result: ArchetypeResult) -> Console:
    return c.with_panel(
      "Time Archetype",
      self._header_content(
        result,
        properties=[
          "Triangular Dependence",
          "Permutation Equivariance (optional, but rare)",
        ],
      ),
    )

  def with_triangular_spec(self, spec: Expr) -> "Time[Y]":
    self.triangular_spec = spec
    return self

  def with_permutation_spec(self, spec: Expr) -> "Time[Y]":
    self.permutation_spec = spec
    return self

  def verify(self, **kwargs: Any) -> ArchetypeResult:
    """
    Runs the semantic property checks for this archetype.

    Args:
      **kwargs: Keyword arguments for the function under test.
    """
    self.kwargs.update(kwargs)
    input_spec, output_spec = self._require_specs()
    deferred_call = DeferredCall(
      self.fn,
      input_spec=input_spec,
      output_spec=output_spec,
      **self.kwargs,
    )
    triangular_dependent = is_triangular_dependent_trials(
      deferred_call,
      trials=self.trials,
      rng=self.rng,
      test_spec=self.triangular_spec or TRIANGULAR_DEPENDENT_SPEC,
      atol=self.atol,
      rtol=self.rtol,
      dtype=self.dtype,
    )

    permutation_equivariant: TestState | None = None
    if self.permutation_equivariant:
      permutation_equivariant = is_permutation_equivariant_trials(
        deferred_call,
        trials=self.trials,
        rng=self.rng,
        test_spec=self.permutation_spec or PERMUTATION_EQUIVARIANT_SPEC,
        atol=self.atol,
        rtol=self.rtol,
        dtype=self.dtype,
      )
    checks = [self._check_result("Triangular Dependence", triangular_dependent)]
    if self.permutation_equivariant and permutation_equivariant is not None:
      checks.append(
        self._check_result("Permutation Equivariance", permutation_equivariant)
      )

    return self._finalize_result(checks)
