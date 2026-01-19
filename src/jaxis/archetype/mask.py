from __future__ import annotations

from typing import Any, Callable, TypeVar

from jaxis.archetype.common import Archetype, ArchetypeResult
from jaxis.console import Console
from jaxis.dsl.ast import Expr
from jaxis.fn.caller import DeferredCall
from jaxis.semantics.mask_invariant import (
  DEFAULT_SPEC as MASK_INVARIANT_SPEC,
)
from jaxis.semantics.mask_invariant import (
  is_mask_invariant_trials,
)

Y = TypeVar("Y")


class Mask(Archetype[Y]):
  """
  The Mask archetype for masked/padded axes.

  This archetype verifies Mask Invariance: changing masked/padded inputs must not
  affect unmasked outputs.
  """

  def __init__(self, fn: Callable[..., Y], *, test_spec: Expr | None = None):
    """
    Initializes a mask archetype.

    Parameters:
    - fn: The function to test for semantic properties typical of mask operations.
    - test_spec: Optional Metric DSL expression overriding the default mask-invariant gates.
    """
    super().__init__(fn)
    self.mask_test_spec = test_spec

  def _describe_header(self, c: Console, result: ArchetypeResult) -> Console:
    return c.with_panel(
      "Mask Archetype",
      self._header_content(result, properties=["Mask Invariance"]),
    )

  def with_test_spec(self, test_spec: Expr) -> "Mask[Y]":
    """Override the default mask-invariant gate profile."""
    self.mask_test_spec = test_spec
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

    mask_invariant = is_mask_invariant_trials(
      deferred_call,
      trials=self.trials,
      rng=self.rng,
      test_spec=self.mask_test_spec or MASK_INVARIANT_SPEC,
      atol=self.atol,
      rtol=self.rtol,
      dtype=self.dtype,
    )
    checks = [self._check_result("Mask Invariance", mask_invariant)]
    return self._finalize_result(checks)
