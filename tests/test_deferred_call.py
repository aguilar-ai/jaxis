from typing import Callable

import pytest

from jaxis.exceptions import (
  InputAlreadyProvidedError,
  InvalidArgumentsError,
  MissingArgumentsError,
)
from jaxis.fn.caller import DeferredCall


@pytest.fixture
def add():
  def _add(x: int, y: int) -> int:
    return x + y

  return _add


def test_deferred_call(add: Callable[[int, int], int]) -> None:
  deferred = DeferredCall(add, input_spec=("x", (1,)), y=1)
  assert deferred.fn is add
  assert deferred.input_spec[0].arg_name == "x"
  assert deferred.kwargs == {"y": 1}

  deferred = deferred.with_generated_input("x", 1)
  assert deferred.call_and_retrieve_output() == 2


def test_deferred_call_with_missing_arguments(add: Callable[[int, int], int]) -> None:
  with pytest.raises(MissingArgumentsError):
    DeferredCall(add, input_spec=("x", (1,)))


def test_deferred_call_with_invalid_arguments(add: Callable[[int, int], int]) -> None:
  with pytest.raises(InvalidArgumentsError):
    DeferredCall(add, input_spec=("x", (1,)), z=1, y=3)


def test_deferred_call_with_input_already_provided(
  add: Callable[[int, int], int],
) -> None:
  with pytest.raises(InputAlreadyProvidedError):
    DeferredCall(add, input_spec=("x", (1,)), x=1, y=3)
