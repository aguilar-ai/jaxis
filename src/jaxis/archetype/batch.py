from __future__ import annotations

from typing import Any, Callable, TypeVar

from jaxis.archetype.common import Archetype, ArchetypeResult
from jaxis.console import Console
from jaxis.dsl.ast import Expr
from jaxis.fn.caller import DeferredCall
from jaxis.semantics import (
  is_elementwise_independent_trials,
  is_permutation_equivariant_trials,
)
from jaxis.semantics.elementwise_independent import (
  DEFAULT_SPEC as ELEMENTWISE_INDEPENDENT_SPEC,
)
from jaxis.semantics.permutation_equivariant import (
  DEFAULT_SPEC as PERMUTATION_EQUIVARIANT_SPEC,
)

Y = TypeVar("Y")


class Batch(Archetype[Y]):
  """
  The Batch archetype for sample axes that should be independent and order-agnostic.

  This archetype bundles:
  - Element-wise Independence: Samples do not cross-talk.
  - Permutation Equivariance: Sample order does not matter.
  """

  def __init__(
    self,
    fn: Callable[..., Y],
    *,
    elementwise_spec: Expr | None = None,
    permutation_spec: Expr | None = None,
  ):
    """
    Initializes a batch archetype.

    Parameters:
    - fn: The function to test for semantic properties typical of batch operations.
    - elementwise_spec: Optional Metric DSL expression for elementwise-independence gate.
    - permutation_spec: Optional Metric DSL expression for permutation-equivariance gate.
    """
    super().__init__(fn)
    self.elementwise_spec = elementwise_spec
    self.permutation_spec = permutation_spec

  def _describe_header(self, c: Console, result: ArchetypeResult) -> Console:
    return c.with_panel(
      "Batch Archetype",
      self._header_content(
        result,
        properties=["Element-wise Independence", "Permutation Equivariance"],
      ),
    )

  def with_elementwise_spec(self, spec: Expr) -> "Batch[Y]":
    self.elementwise_spec = spec
    return self

  def with_permutation_spec(self, spec: Expr) -> "Batch[Y]":
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

    permutation_equivariant = is_permutation_equivariant_trials(
      deferred_call,
      trials=self.trials,
      rng=self.rng,
      test_spec=self.permutation_spec or PERMUTATION_EQUIVARIANT_SPEC,
      atol=self.atol,
      rtol=self.rtol,
    )
    elementwise_independent = is_elementwise_independent_trials(
      deferred_call,
      trials=self.trials,
      rng=self.rng,
      test_spec=self.elementwise_spec or ELEMENTWISE_INDEPENDENT_SPEC,
      atol=self.atol,
      rtol=self.rtol,
      dtype=self.dtype,
    )

    checks = [
      self._check_result("Permutation Equivariance", permutation_equivariant),
      self._check_result("Element-wise Independence", elementwise_independent),
    ]

    return self._finalize_result(checks)
