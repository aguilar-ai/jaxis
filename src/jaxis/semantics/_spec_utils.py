"""Helpers for evaluating Metric DSL expressions in semantic property checks."""

from __future__ import annotations

from typing import Iterable, Mapping

from jaxis.common_typing import Array
from jaxis.dsl.ast import Expr, Metric, MetricExpr
from jaxis.math import (
  cosine_distance,
  fraction_over_1,
  max_abs_delta,
  max_normalized_violation,
  mean_abs_err,
  leakage_energy_ratio,
  num_inf,
  num_nan,
  num_nonfinite,
  p95,
  p99,
  p999,
  off_target_max_abs,
  off_target_ratio,
  on_target_mag,
  rel_l1,
  rel_l2,
  rel_linf,
  rmse,
)


def metrics_in_expr(expr: Expr) -> set[Metric]:
  """Collect the Metrics referenced by a Metric/Bool expression."""

  out: set[Metric] = set()

  def visit(e: Expr) -> None:
    if isinstance(e, MetricExpr):
      out.add(e.metric)
      return
    for term in e.terms:  # type: ignore[attr-defined]
      visit(term)

  visit(expr)
  return out


def compute_region_metrics(
  metrics: Iterable[Metric],
  ref: Array,
  tgt: Array,
  *,
  atol: float,
  rtol: float,
  eps: float = 1e-12,
) -> dict[Metric, float]:
  """Compute a subset of region-based metrics needed for a given spec.

  Only metrics that do not require extra hyperparameters (e.g., top-k size) are
  supported here. Unsupported metrics raise a ValueError to surface misuse.
  """

  computed: dict[Metric, float] = {}

  for metric in metrics:
    if metric is Metric.P95:
      computed[metric] = float(p95(ref, tgt, atol=atol, rtol=rtol, eps=eps))
    elif metric is Metric.P99:
      computed[metric] = float(p99(ref, tgt, atol=atol, rtol=rtol, eps=eps))
    elif metric is Metric.P999:
      computed[metric] = float(p999(ref, tgt, atol=atol, rtol=rtol, eps=eps))
    elif metric is Metric.MAX:
      computed[metric] = float(
        max_normalized_violation(ref, tgt, atol=atol, rtol=rtol, eps=eps)
      )
    elif metric is Metric.FRACTION_OVER_1:
      computed[metric] = float(fraction_over_1(ref, tgt, atol=atol, rtol=rtol, eps=eps))
    elif metric is Metric.MEAN_ABS_ERR:
      computed[metric] = float(mean_abs_err(ref, tgt))
    elif metric is Metric.RMSE:
      computed[metric] = float(rmse(ref, tgt))
    elif metric is Metric.MAX_ABS_DELTA:
      computed[metric] = float(max_abs_delta(ref, tgt))
    elif metric is Metric.REL_L1:
      computed[metric] = float(rel_l1(ref, tgt, eps=eps))
    elif metric is Metric.REL_L2:
      computed[metric] = float(rel_l2(ref, tgt, eps=eps))
    elif metric is Metric.REL_LINF:
      computed[metric] = float(rel_linf(ref, tgt, eps=eps))
    elif metric is Metric.COSINE_DISTANCE:
      computed[metric] = float(cosine_distance(ref, tgt))
    elif metric is Metric.NUM_NONFINITE:
      computed[metric] = float(num_nonfinite(tgt))
    elif metric is Metric.NUM_NAN:
      computed[metric] = float(num_nan(tgt))
    elif metric is Metric.NUM_INF:
      computed[metric] = float(num_inf(tgt))
    else:
      raise ValueError(f"Unsupported metric in this context: {metric.value}")

  return computed


def compute_leakage_metrics(
  metrics: Iterable[Metric],
  delta_on: Array,
  delta_off: Array,
  *,
  eps: float = 1e-12,
) -> dict[Metric, float]:
  """Compute locality/leakage metrics for elementwise/triangular checks."""

  computed: dict[Metric, float] = {}

  for metric in metrics:
    if metric is Metric.OFF_TARGET_RATIO:
      computed[metric] = float(off_target_ratio(delta_on, delta_off, eps=eps))
    elif metric is Metric.LEAKAGE_ENERGY_RATIO:
      computed[metric] = float(leakage_energy_ratio(delta_on, delta_off, eps=eps))
    elif metric is Metric.OFF_TARGET_MAX_ABS:
      computed[metric] = float(off_target_max_abs(delta_off))
    elif metric is Metric.ON_TARGET_MAG:
      computed[metric] = float(on_target_mag(delta_on))
    else:
      raise ValueError(f"Unsupported leakage metric: {metric.value}")

  return computed


def evaluate_spec(spec: Expr, metrics: Mapping[Metric, float]) -> bool:
  """Evaluate a Metric/Bool expression against a metrics mapping."""

  return spec.evaluate(metrics)
