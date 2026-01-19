from __future__ import annotations

import inspect
from typing import (
  Any,
  Callable,
  Generic,
  ParamSpec,
  TypeVar,
  cast,
)

from ..common_typing import (
  Array,
  InputSpec,
  InputSpec_,
  OutputSpec,
  OutputSpec_,
  Sentinel,
)
from ..exceptions import (
  InputAlreadyProvidedError,
  InvalidArgumentsError,
  MissingArgumentsError,
)

X = ParamSpec("X")
Y = TypeVar("Y")


class DeferredCall(Generic[Y]):
  """
  A deferred call to a function with associated input and output specifications.

  DeferredCall wraps a function and its arguments, allowing semantic property
  checkers to perturb specific inputs and evaluate the results against
  an expected output axis.

  Args:
    fn: The function to be called.
    input_spec: Specification(s) for the input axes to be evaluated.
    output_spec: Specification for the output axis to be evaluated.
    **kwargs: Arguments to be passed to the function.
  """

  def __init__(
    self,
    fn: Callable[..., Y],
    *,
    input_spec: InputSpec | list[InputSpec],
    output_spec: OutputSpec | None = None,
    **kwargs: Any,
  ) -> None:
    self.fn = fn
    self.kwargs: dict[str, Any] = kwargs or {}
    self.input_spec = InputSpec_.from_user_facing_spec(input_spec)
    input_axis = next(
      input_spec.axis
      for input_spec in self.input_spec
      if input_spec.sentinel == Sentinel.DEFAULT
    )
    self.output_spec = OutputSpec_.from_user_facing_spec(
      output_spec, input_axis=input_axis
    )

    self.signature = inspect.signature(fn)
    self.param_names = list(self.signature.parameters.keys())

    # Check that the input argument is not already provided
    for input_spec in self.input_spec:
      if input_spec.arg_name in self.kwargs:
        raise InputAlreadyProvidedError(input_spec.arg_name)

    # Check that the input argument is a valid parameter
    for input_spec in self.input_spec:
      if input_spec.arg_name not in self.param_names:
        raise MissingArgumentsError({input_spec.arg_name})

    # Check if any arguments are provided that are not in the function parameters
    if any(name not in self.param_names for name in self.kwargs):
      raise InvalidArgumentsError(set(self.kwargs.keys()) - set(self.param_names))

    # Check that all other parameters are provided
    input_spec_arg_names = [input_spec.arg_name for input_spec in self.input_spec]
    if not all(
      name in self.kwargs for name in self.param_names if name not in input_spec_arg_names
    ):
      raise MissingArgumentsError(
        set(self.param_names) - set(self.kwargs.keys()) - set(input_spec_arg_names)
      )

  def with_generated_input(self, arg_name: str, value: Any) -> "DeferredCall[Y]":
    self.kwargs[arg_name] = value
    return self

  def with_generated_inputs(self, **kwargs: Any) -> "DeferredCall[Y]":
    self.kwargs.update(kwargs)
    return self

  def __call__(self) -> Y:
    return self.fn(**self.kwargs)

  def call_and_retrieve_output(self) -> Array:
    if self.output_spec.tuple_i is not None:
      return cast(Array, cast(tuple[Y, ...], self())[self.output_spec.tuple_i])
    else:
      return cast(Array, self())
