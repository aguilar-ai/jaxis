<p align="center">
  <img src="docs/public/logo-red.svg" alt="Jaxis Logo" width="220">
</p>

# Executable invariants across JAX transformations

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/docs-jaxis--docs.vercel.app-blue.svg)](https://jaxis-docs.vercel.app/)

Jaxis lets you specify and verify executable invariants about axis semantics (batch, time, 
mask, etc.) in JAX programs, including across transformations and controlled semantic 
ablations.  It does this by generating randomized inputs, applying structured 
perturbations and local subgraph substitutions, and evaluating robust metrics to decide 
pass/fail in the presence of floating-point noise.

## What You Get

- **Archetypes**: high-level presets that bundle multiple semantic checks (Batch, Time,
  Mask) into a single `verify(...)` call.
- **Semantic properties**: atomic checks like permutation equivariance/invariance,
  mask invariance, triangular dependence, and elementwise independence.
- **Metric DSL**: compose pass/fail gates using metrics (`P99`, `MAX`, `REL_L2`, …) and
  boolean logic.
- **Reproducibility**: multi-trial helpers return a `TestState` with the failing trial’s
  RNG state so failures can be replayed.

## Install

```sh
uv pip install git+https://github.com/aguilar-ai/jaxis.git
```

## Documentation

You can find very exhaustive documentation on the 
[Jaxis Docs](https://jaxis-docs.vercel.app/) website.

## Quick Start

### Archetypes

The fastest path is an archetype: declare the axis you care about and call `verify`.

```py
import jax.numpy as jnp

from jaxis.archetype import Batch


def loss(wb, X, y):
  w, b = wb[:1], wb[1]
  yhat = (X @ w).ravel() + b
  return (yhat - y) ** 2


b = (
  Batch(loss)
    .with_input_spec(("X", (128, 1), 0))  # Jaxis generates X with a batch axis at 0
    .with_output_spec(0)                  # output’s batch axis is also 0
)

result = b.verify(
  wb=jnp.ones((2,)),
  y=jnp.ones((128,)),
)

assert result.passed
```

Read more about archetypes 
[in the documentation.](https://jaxis-docs.vercel.app/quick-start#archetypes).

### Semantic Properties

Use semantic properties directly when you want precision and composability.

```py
import jax.numpy as jnp

from jaxis.fn import DeferredCall
from jaxis.semantics import is_permutation_equivariant_trials


def square(x):
  return jnp.square(x)


fn = DeferredCall(
  square,
  input_spec=("x", (32, 128), -1),
  output_spec=-1,
)

state = is_permutation_equivariant_trials(fn, trials=24)
assert state.passed
```

Read more about semantic properties 
[in the documentation.](https://jaxis-docs.vercel.app/quick-start#semantic-properties).

#### Metric DSL & Custom Gates

Every semantic check accepts a `test_spec` expression built from `jaxis.dsl.Metric`.

```py
from jaxis.dsl import Metric
from jaxis.semantics import is_permutation_equivariant

spec = (Metric.P99 < 1.0) & (Metric.MAX < 3.0)
ok = is_permutation_equivariant(fn, test_spec=spec)
```

## Documentation

You can find the full documentation on the 
[Jaxis Docs](https://jaxis-docs.vercel.app/) website.

## License

Licensed under the Apache License, Version 2.0. See `LICENSE`.
