from __future__ import annotations


class MissingArgumentsError(Exception):
  """Raised when a deferred call is constructed with missing arguments."""

  def __init__(self, missing_args: set[str]):
    self.missing_args = missing_args
    super().__init__(f"Missing argument(s): `{'`, `'.join(missing_args)}`")


class InvalidArgumentsError(Exception):
  """Raised when a deferred call is constructed with invalid arguments."""

  def __init__(self, invalid_args: set[str]):
    self.invalid_args = invalid_args
    super().__init__(
      f"Invalid argument(s): `{'`, `'.join(invalid_args)}`."
      " Check if your function takes these arguments."
    )


class InputAlreadyProvidedError(Exception):
  """Raised when a deferred call is constructed with an input argument that has already been provided."""

  def __init__(self, input_arg: str):
    self.input_arg = input_arg
    super().__init__(f"Input argument `{input_arg}` has already been provided.")


class InputSpecMissingError(Exception):
  """Raised when an archetype is missing an input spec."""

  def __init__(self) -> None:
    super().__init__("Input spec must be provided. Call with_input_spec(...).")


class OutputSpecMissingError(Exception):
  """Raised when an archetype is missing an output spec."""

  def __init__(self) -> None:
    super().__init__("Output spec must be provided. Call with_output_spec(...).")


class PositionalArgumentsProvidedError(Exception):
  """Raised when a deferred call is constructed with positional arguments."""

  def __init__(self):
    super().__init__(
      "All arguments to your function must be provided as keyword arguments."
    )


class OutputAxisMissingError(Exception):
  """Raised when the output axis is missing from the actual shape of the output."""

  def __init__(self, missing_axis: int, actual_shape: tuple[int, ...]) -> None:
    self.missing_axis = missing_axis
    self.actual_shape = actual_shape
    super().__init__(
      f"Axis {missing_axis} is missing from the output. The actual shape of the output is"
      f" {actual_shape} and the maximum allowed axis is {len(actual_shape) - 1}. Please "
      f"provide a valid output axis or use a different function."
    )


class MaskShapeMismatchError(Exception):
  """Raised when the shape of the mask is not the same as the shape of the input."""

  def __init__(
    self, shape_i: tuple[int, ...], shape_m: tuple[int, ...], axis_i: int, axis_m: int
  ):
    self.shape_i = shape_i
    self.shape_m = shape_m
    self.axis_i = axis_i
    self.axis_m = axis_m
    super().__init__(
      f"Shape of mask {shape_m} does not match shape of input {shape_i} along axis {axis_m}."
    )


class InputShapeTooSmallError(Exception):
  """Raised when the input shape is too small to be perturbed."""

  def __init__(self, shape_in: tuple[int, ...], axis_in: int):
    self.shape_in = shape_in
    self.axis_in = axis_in
    super().__init__(
      f"Input shape {shape_in} is too small to be perturbed along axis {axis_in}."
    )
