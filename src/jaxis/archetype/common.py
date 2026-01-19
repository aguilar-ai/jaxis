from __future__ import annotations

import pickle
import uuid
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from time import time
from typing import Any, Callable, Generic, Literal, Self, TypeVar, cast, overload

import jax
from jax.typing import DTypeLike
from platformdirs import user_data_dir

from jaxis.common_typing import (
  Array,
  InputSpec,
  InputSpec_,
  OutputSpec,
  OutputSpec_,
  TestState,
)
from jaxis.console import Console
from jaxis.constants import (
  DEFAULT_ATOL,
  DEFAULT_DTYPE,
  DEFAULT_RNG_SEED,
  DEFAULT_RTOL,
  DEFAULT_TRIALS,
)
from jaxis.exceptions import InputSpecMissingError, OutputSpecMissingError

Y = TypeVar("Y")
ReportFormat = Literal["rich", "text", "json"]


def _serialize_rng(rng: Array) -> tuple[int, ...] | str:
  try:
    return tuple(int(value) for value in jax.device_get(rng))
  except TypeError:
    return str(rng)


def _serialize_value(value: Any) -> Any:
  if isinstance(value, (str, int, float, bool)) or value is None:
    return value
  if isinstance(value, Path):
    return str(value)
  if isinstance(value, dict):
    items = cast(dict[Any, Any], value).items()
    return {str(key): _serialize_value(val) for key, val in items}
  if isinstance(value, (list, tuple)):
    values = cast(list[Any] | tuple[Any, ...], value)
    return [_serialize_value(val) for val in values]
  return str(value)


def _format_value(value: Any, *, max_chars: int = 120) -> str:
  if isinstance(value, jax.Array):
    return f"Array(shape={value.shape}, dtype={value.dtype})"
  if hasattr(value, "shape") and hasattr(value, "dtype"):
    return f"Array(shape={value.shape}, dtype={value.dtype})"
  text = str(value)
  if len(text) > max_chars:
    return f"{text[: max_chars - 3]}..."
  return text


def _serialize_input_spec(spec: InputSpec_) -> dict[str, Any]:
  return {
    "arg_name": spec.arg_name,
    "shape": spec.shape,
    "axis": int(spec.axis),
    "sentinel": spec.sentinel.name,
  }


def _serialize_output_spec(spec: OutputSpec_) -> dict[str, Any]:
  return {
    "axis": int(spec.axis),
    "tuple_index": spec.tuple_i,
  }


@dataclass(frozen=True)
class CheckResult:
  """
  The result of a single property check within an archetype.

  Attributes:
    name: Name of the property checked (e.g., "Permutation Equivariance").
    passed: True if the property was satisfied.
    trial_index: Index of the first trial that failed.
    rng: Random key used for the trial that failed.
    artifact_path: Path to serialized failure context if artifacts are enabled.
  """

  name: str
  passed: bool
  trial_index: int | None
  rng: Array | None
  artifact_path: Path | None

  def to_dict(self) -> dict[str, Any]:
    return {
      "name": self.name,
      "passed": self.passed,
      "trial_index": self.trial_index,
      "rng": _serialize_rng(self.rng) if self.rng is not None else None,
      "artifact_path": str(self.artifact_path) if self.artifact_path else None,
    }


@dataclass(frozen=True)
class ArchetypeResult:
  """
  The comprehensive result of an archetype verification run.

  Attributes:
    archetype: Name of the archetype (e.g., "Batch").
    function: Qualified name of the function under test.
    passed: True if all properties were satisfied.
    checks: Tuple of individual property check results.
    input_specs: The inferred input specifications used.
    output_specs: The inferred output specifications used.
    args: Keyword arguments passed to the function.
    trials: Number of trials performed.
    atol: Absolute tolerance used.
    rtol: Relative tolerance used.
    dtype: Data type used.
    rng: The base RNG key for the run.
    artifacts_enabled: True if failure artifacts were recorded.
    artifacts_dir: Directory where artifacts are stored.
  """

  archetype: str
  function: str
  passed: bool
  checks: tuple[CheckResult, ...]
  input_specs: tuple[InputSpec_, ...]
  output_specs: tuple[OutputSpec_, ...]
  args: dict[str, Any]
  trials: int
  atol: float
  rtol: float
  dtype: DTypeLike
  rng: Array
  artifacts_enabled: bool
  artifacts_dir: Path | None

  def __bool__(self) -> bool:
    return self.passed

  def to_dict(self) -> dict[str, Any]:
    return {
      "archetype": self.archetype,
      "function": self.function,
      "passed": self.passed,
      "checks": [check.to_dict() for check in self.checks],
      "input_specs": [_serialize_input_spec(spec) for spec in self.input_specs],
      "output_specs": [_serialize_output_spec(spec) for spec in self.output_specs],
      "args": {key: _serialize_value(val) for key, val in self.args.items()},
      "trials": self.trials,
      "atol": self.atol,
      "rtol": self.rtol,
      "dtype": str(self.dtype),
      "rng": _serialize_rng(self.rng),
      "artifacts_enabled": self.artifacts_enabled,
      "artifacts_dir": str(self.artifacts_dir) if self.artifacts_dir else None,
    }


def write_state_to_path(
  payload: dict[str, Any], *, directory: Path | None = None
) -> Path:
  try:
    v = version("jaxis")
  except PackageNotFoundError:
    v = "0.0.0"

  base_dir = directory
  if base_dir is None:
    base_dir = Path(
      user_data_dir(
        "jaxis",
        "aguilar_ai",
        version=v,
        ensure_exists=True,
      )
    ) / str(int(time()))

  base_dir.mkdir(parents=True, exist_ok=True)
  path = base_dir / f"{uuid.uuid4()}.pkl"

  with open(path, "wb") as f:
    pickle.dump(payload, f)

  return path


class Archetype(Generic[Y]):
  def __init__(self, fn: Callable[..., Y]):
    self.fn = fn
    self.input_specs: list[InputSpec_] = []
    self.output_specs: list[OutputSpec_] = []
    self.rng: Array = jax.random.key(DEFAULT_RNG_SEED)
    self.atol: float = DEFAULT_ATOL
    self.rtol: float = DEFAULT_RTOL
    self.dtype: DTypeLike = DEFAULT_DTYPE
    self.trials: int = DEFAULT_TRIALS
    self.kwargs: dict[str, Any] = {}
    self.input_spec_user_facing: list[InputSpec] | InputSpec | None = None
    self.output_spec_user_facing: OutputSpec | None = None
    self.save_failure_artifacts: bool = False
    self.artifacts_dir: Path | None = None
    self.result: ArchetypeResult | None = None

  @overload
  def with_input_spec(
    self,
    input_spec: InputSpec,
  ) -> Self: ...

  @overload
  def with_input_spec(
    self,
    input_spec: list[InputSpec],
  ) -> Self: ...

  def with_input_spec(
    self,
    input_spec: InputSpec | list[InputSpec],
  ) -> Self:
    """
    Sets the input specification(s) for the archetype.

    Args:
      input_spec: (arg_name, shape, axis, sentinel) or a list of such tuples.
    """
    self.input_spec_user_facing = input_spec
    self.input_specs.extend(InputSpec_.from_user_facing_spec(input_spec))

    return self

  def with_output_spec(
    self,
    output_spec: OutputSpec,
  ) -> Self:
    """
    Sets the output specification for the archetype.

    Args:
      output_spec: axis or (axis, tuple_index).
    """
    self.output_spec_user_facing = output_spec
    self.output_specs.append(OutputSpec_.from_user_facing_spec(output_spec))
    return self

  def with_rng(
    self,
    rng: Array,
  ) -> Self:
    """Sets the random number generator key for the archetype."""
    self.rng = rng
    return self

  def with_tolerances(
    self,
    atol: float,
    rtol: float,
  ) -> Self:
    """Sets the absolute and relative tolerances for the archetype."""
    self.atol = atol
    self.rtol = rtol
    return self

  def with_dtype(
    self,
    dtype: DTypeLike,
  ) -> Self:
    """Sets the data type for randomized inputs."""
    self.dtype = dtype
    return self

  def with_trials(
    self,
    trials: int,
  ) -> Self:
    """Sets the number of randomized trials to perform."""
    self.trials = trials
    return self

  def with_args(self, **kwargs: Any) -> Self:
    """Sets additional keyword arguments to pass to the function under test."""
    self.kwargs.update(kwargs)
    return self

  def with_failure_artifacts(
    self,
    *,
    enabled: bool = True,
    directory: Path | str | None = None,
  ) -> Self:
    """Enable or disable saving failure artifacts to disk."""
    if not enabled:
      self.save_failure_artifacts = False
      self.artifacts_dir = None
      return self

    self.save_failure_artifacts = True
    self.artifacts_dir = Path(directory) if directory is not None else None
    return self

  def _function_qualname(self) -> str:
    return f"{self.fn.__module__}.{self.fn.__qualname__}"

  def _require_specs(self) -> tuple[InputSpec | list[InputSpec], OutputSpec]:
    if self.input_spec_user_facing is None or len(self.input_specs) == 0:
      raise InputSpecMissingError()
    if self.output_spec_user_facing is None or len(self.output_specs) == 0:
      raise OutputSpecMissingError()
    return (self.input_spec_user_facing, self.output_spec_user_facing)

  def _build_failure_payload(
    self,
    *,
    check_name: str,
    state: TestState,
  ) -> dict[str, Any]:
    return {
      "archetype": self.__class__.__name__,
      "check": check_name,
      "function": self._function_qualname(),
      "rng": jax.device_get(state.rng) if state.rng is not None else None,
      "trial_index": state.trial_index,
      "test_spec": str(state.test_spec) if hasattr(state, "test_spec") else None,
      "input_specs": [_serialize_input_spec(spec) for spec in self.input_specs],
      "output_specs": [_serialize_output_spec(spec) for spec in self.output_specs],
      "args": {key: _serialize_value(val) for key, val in self.kwargs.items()},
      "trials": self.trials,
      "atol": self.atol,
      "rtol": self.rtol,
      "dtype": str(self.dtype),
      "base_rng": jax.device_get(self.rng),
    }

  def _check_result(self, name: str, state: TestState) -> CheckResult:
    artifact_path = None
    if not state.passed and self.save_failure_artifacts and state.rng is not None:
      artifact_path = write_state_to_path(
        self._build_failure_payload(check_name=name, state=state),
        directory=self.artifacts_dir,
      )

    return CheckResult(
      name=name,
      passed=state.passed,
      trial_index=state.trial_index,
      rng=state.rng,
      artifact_path=artifact_path,
    )

  def _finalize_result(self, checks: list[CheckResult]) -> ArchetypeResult:
    result = ArchetypeResult(
      archetype=self.__class__.__name__,
      function=self._function_qualname(),
      passed=all(check.passed for check in checks),
      checks=tuple(checks),
      input_specs=tuple(self.input_specs),
      output_specs=tuple(self.output_specs),
      args=dict(self.kwargs),
      trials=self.trials,
      atol=self.atol,
      rtol=self.rtol,
      dtype=self.dtype,
      rng=self.rng,
      artifacts_enabled=self.save_failure_artifacts,
      artifacts_dir=self.artifacts_dir if self.save_failure_artifacts else None,
    )
    self.result = result
    return result

  def _header_content(self, result: ArchetypeResult, *, properties: list[str]) -> str:
    status = "PASSED" if result.passed else "FAILED"
    failed_checks = [check.name for check in result.checks if not check.passed]
    failed_summary = ", ".join(failed_checks) if failed_checks else "None"
    properties_block = "\n".join(f"- {name}" for name in properties)
    return f"""
- Result: **{status}**
- Function: `{result.function}`
- Failed Checks: {failed_summary}

Testable Semantic Properties:
{properties_block}
"""

  def _describe_header(self, c: Console, result: ArchetypeResult) -> Console:
    raise NotImplementedError

  def _report_rich(self, result: ArchetypeResult) -> None:
    c = Console()
    c = self._describe_header(c, result)

    c = c.with_table(
      "Checks",
      [
        ("Check", {"style": "bold"}),
        ("Passed", {"style": "bold"}),
        ("Trial", {"style": "bold"}),
        ("Artifact", {"style": "bold"}),
      ],
      [
        (
          check.name,
          str(check.passed),
          "-" if check.trial_index is None else str(check.trial_index),
          "-" if check.artifact_path is None else str(check.artifact_path),
        )
        for check in result.checks
      ],
    )

    c = c.with_table(
      "Inferred Input Specs",
      [
        ("Argument Name", {"style": "cyan bold"}),
        ("Shape", {"style": "bold"}),
        ("Axis", {"style": "bold"}),
        ("Sentinel", {"style": "bold"}),
      ],
      [
        (
          input_spec.arg_name,
          str(input_spec.shape),
          str(input_spec.axis),
          input_spec.sentinel.name,
        )
        for input_spec in result.input_specs
      ],
    )

    c = c.with_table(
      "Inferred Output Spec",
      [
        ("Tuple Index", {"style": "bold"}),
        ("Axis", {"style": "bold"}),
      ],
      [
        (
          "-" if output_spec.tuple_i is None else str(output_spec.tuple_i),
          str(output_spec.axis),
        )
        for output_spec in result.output_specs
      ],
    )

    c = c.with_table(
      "Arguments",
      [
        ("Name", {"style": "bold"}),
        ("Value", {"style": "bold"}),
      ],
      [(arg_name, _format_value(value)) for arg_name, value in result.args.items()],
    )

    c = c.with_table(
      "Testing Parameters",
      [
        ("Parameter", {"style": "bold"}),
        ("Value", {"style": "bold"}),
      ],
      [
        ("Trials", str(result.trials)),
        ("Absolute Tolerance", str(result.atol)),
        ("Relative Tolerance", str(result.rtol)),
        ("DType", str(result.dtype)),
        ("RNG", str(_serialize_rng(result.rng))),
        ("Artifacts Enabled", str(result.artifacts_enabled)),
        (
          "Artifacts Directory",
          "-" if result.artifacts_dir is None else str(result.artifacts_dir),
        ),
      ],
    )

    c.print()

  def _report_text(self, result: ArchetypeResult) -> str:
    lines: list[str] = []
    lines.append(f"{result.archetype} Archetype")
    lines.append(f"Function: {result.function}")
    lines.append(f"Passed: {result.passed}")
    lines.append(f"Artifacts Enabled: {result.artifacts_enabled}")
    if result.artifacts_dir is not None:
      lines.append(f"Artifacts Directory: {result.artifacts_dir}")
    lines.append("")

    lines.append("Checks:")
    for check in result.checks:
      status = "PASSED" if check.passed else "FAILED"
      trial = "-" if check.trial_index is None else str(check.trial_index)
      rng = "-" if check.rng is None else str(_serialize_rng(check.rng))
      artifact = "-" if check.artifact_path is None else str(check.artifact_path)
      lines.append(
        f"- {check.name}: {status} (trial={trial}, rng={rng}, artifact={artifact})"
      )
    lines.append("")

    lines.append("Input Specs:")
    for spec in result.input_specs:
      lines.append(
        f"- {spec.arg_name}: shape={spec.shape}, axis={spec.axis}, "
        f"sentinel={spec.sentinel.name}"
      )
    lines.append("")

    lines.append("Output Specs:")
    for spec in result.output_specs:
      tuple_index = "-" if spec.tuple_i is None else str(spec.tuple_i)
      lines.append(f"- axis={spec.axis}, tuple_index={tuple_index}")
    lines.append("")

    lines.append("Arguments:")
    for name, value in result.args.items():
      lines.append(f"- {name}: {_format_value(value)}")
    lines.append("")

    lines.append("Testing Parameters:")
    lines.append(f"- trials: {result.trials}")
    lines.append(f"- atol: {result.atol}")
    lines.append(f"- rtol: {result.rtol}")
    lines.append(f"- dtype: {result.dtype}")
    lines.append(f"- rng: {_serialize_rng(result.rng)}")

    return "\n".join(lines)

  def report(self, *, format: ReportFormat = "rich") -> str | dict[str, Any] | None:
    """Render the most recent verification result."""
    if self.result is None:
      raise ValueError("No verification result available. Call verify() first.")

    if format == "rich":
      self._report_rich(self.result)
      return None
    if format == "text":
      return self._report_text(self.result)
    if format == "json":
      return self.result.to_dict()

    raise ValueError(f"Unsupported report format: {format}")

  def describe(self) -> None:
    self.report(format="rich")
