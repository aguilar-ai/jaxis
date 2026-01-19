from __future__ import annotations

import enum
from typing import Mapping, NamedTuple, Union


class ComparisonOperator(enum.Enum):
  LT = "<"
  LE = "<="
  EQ = "=="
  NE = "!="
  GE = ">="
  GT = ">"

  def compare(self, lhs: float, rhs: float) -> bool:
    if self is ComparisonOperator.LT:
      return lhs < rhs
    if self is ComparisonOperator.LE:
      return lhs <= rhs
    if self is ComparisonOperator.EQ:
      return lhs == rhs
    if self is ComparisonOperator.NE:
      return lhs != rhs
    if self is ComparisonOperator.GE:
      return lhs >= rhs
    if self is ComparisonOperator.GT:
      return lhs > rhs

    raise AssertionError(f"Unhandled operator: {self}")


class BoolOp(enum.Enum):
  AND = "&"
  OR = "|"


Expr = Union["MetricExpr", "BoolExpr"]


class MetricExpr(NamedTuple):
  metric: Metric
  comparison_operator: ComparisonOperator
  value: int | float

  def __str__(self) -> str:
    return f"{self.metric.value} {self.comparison_operator.value} {self.value:g}"

  def __repr__(self) -> str:
    return f"MetricExpr({str(self)})"

  def __bool__(self) -> bool:
    raise TypeError(
      "MetricExpr is not a boolean. Call .evaluate(metrics) or combine with &/|."
    )

  def __and__(self, other: Expr) -> BoolExpr:
    return BoolExpr._and(self, other)  # type: ignore[attr-defined]

  def __or__(self, other: Expr) -> BoolExpr:
    return BoolExpr._or(self, other)  # type: ignore[attr-defined]

  def evaluate(self, metrics: Mapping[Metric, float]) -> bool:
    actual = float(metrics[self.metric])
    return self.comparison_operator.compare(actual, self.value)


class Metric(enum.Enum):
  """
  Metric definitions for property tests.

  Conventions these metrics assume (so the enum stays small and high-leverage):

  - You compare a reference output `ref` and a target output `tgt` on a selected region R.
  - Define elementwise absolute error on $R$:
      $\text{abs\_err} = |\text{tgt} - \text{ref}|$
  - Define elementwise "normalized violation" on $R$:
      $v = \text{abs\_err} / (\text{atol} + \text{rtol} \cdot |\text{ref}| + \epsilon)$
    where $(\text{atol}, \text{rtol})$ live in the *evaluation context*, not in the enum.
  - Most tail/coverage metrics operate on $v$ (dimensionless).
  - Norm-ratio metrics operate on $\text{ref}$ and $\text{tgt}$ (or their deltas) over $R$.
  - Some metrics ($\text{OFF\_TARGET\_*}$, $\text{ON\_TARGET\_*}$) require the test to provide an
    "on-target" subset and its complement "off-target" subset (typical for
    elementwise-independence/locality tests).
  """

  # --- Tail/coverage on normalized violation v ---
  P95 = "p95"
  """
  95th percentile of normalized violation $v$ over the region $R$.
  Good 'typical tail' signal.
  """

  P99 = "p99"
  """
  99th percentile of normalized violation $v$ over $R$. Good default strictness knob.
  """

  P999 = "p999"
  """
  99.9th percentile of normalized violation $v$ over $R$.
  Use when you want very low false-pass rate.
  """

  MAX = "max"
  """
  Maximum normalized violation $v$ over $R$.
  Use as a looser 'hard cap' alongside a tail percentile.
  """

  FRACTION_OVER_1 = "fraction_over_1"
  """
  Fraction of elements in $R$ with $v > 1.0$.
  Catches widespread mild regressions that percentiles can miss.
  """

  # --- Aggregate error on raw deltas ---
  MEAN_ABS_ERR = "mean_abs_err"
  """
  $\text{Mean}(|\text{tgt} - \text{ref}|)$ over $R$ (raw units).
  Useful when absolute scale matters (near-zero regimes).
  """

  RMSE = "rmse"
  """
  $\sqrt{\text{mean}((\text{tgt} - \text{ref})^2)}$ over $R$ (raw units).
  More sensitive to large outliers than $\text{MEAN\_ABS\_ERR}$.
  """

  MAX_ABS_DELTA = "max_abs_delta"
  """
  $\text{Max}(|\text{tgt} - \text{ref}|)$ over $R$ (raw units).
  Absolute hard cap; often paired with normalized tail metrics.
  """

  # --- Global relative error (norm ratios) ---
  REL_L1 = "rel_l1"
  """
  $\lVert \text{tgt} - \text{ref} \rVert_1 / (\lVert \text{ref} \rVert_1 + \epsilon)$ over $R$.
  Scale-free; good for catching broad shifts.
  """

  REL_L2 = "rel_l2"
  """
  $\lVert \text{tgt} - \text{ref} \rVert_2 / (\lVert \text{ref} \rVert_2 + \epsilon)$ over $R$.
  Common default for 'overall closeness'.
  """

  REL_LINF = "rel_linf"
  """
  $\lVert \text{tgt} - \text{ref} \rVert_\infty / (\lVert \text{ref} \rVert_\infty + \epsilon)$ over $R$.
  Sensitive to worst-case relative deviation.
  """

  # --- Vector similarity (especially gradients/updates) ---
  COSINE_DISTANCE = "cosine_distance"
  """
  $1 - \text{cosine\_similarity}(\text{ref}, \text{tgt})$ on flattened tensors over $R$.
  Use for gradients/updates; smaller is better.
  """

  # --- Distributional comparisons (probabilities or normalized scores) ---
  KL_DIVERGENCE = "kl_divergence"
  """
  $\text{KL}(p \parallel q)$ over $R$ (requires probability vectors $p=\text{ref}$, $q=\text{tgt}$ with smoothing/eps).
  Directional.
  """

  JS_DIVERGENCE = "js_divergence"
  """
  Jensen-Shannon divergence between $p$ and $q$ over $R$ (requires probabilities).
  Symmetric and bounded.
  """

  # --- Ordering / retrieval style comparisons ---
  TOPK_OVERLAP = "topk_overlap"
  """
  Top-$k$ index agreement between $\text{ref}$ and $\text{tgt}$ scores (requires parameter $k$).
  Typically overlap/Jaccard per row.
  """

  KENDALL_TAU_DISTANCE = "kendall_tau_distance"
  """
  $1 - \text{Kendall's tau rank correlation}$ between $\text{ref}$ and $\text{tgt}$ rankings
  (requires rankable scores). Smaller is better.
  """

  # --- Special-values diagnostics ---
  NUM_NONFINITE = "num_nonfinite"
  """
  Count of $\text{NaN}$ or $\text{Inf}$ in $\text{tgt}$ over $R$.
  High-leverage: should generally be $0$ in property tests.
  """

  NUM_NAN = "num_nan"
  """
  Count of $\text{NaN}$ in $\text{tgt}$ over $R$. Use when you want to distinguish $\text{NaN}$ failures from $\text{Inf}$ 
  failures.
  """

  NUM_INF = "num_inf"
  """
  Count of $\pm \text{Inf}$ in $\text{tgt}$ over $R$. Use when you want to distinguish overflow/$\text{Inf}$ from $\text{NaN}$.
  """

  # --- Property-structure (locality / leakage) ---
  OFF_TARGET_RATIO = "off_target_ratio"
  """
  $\max(|\Delta_\text{off}|) / (|\Delta_\text{on}| + \epsilon)$.
  Requires $\Delta = f(x') - f(x)$ and on/off-target split.
  Smaller means less leakage.
  """

  LEAKAGE_ENERGY_RATIO = "leakage_energy_ratio"
  """
  $\lVert \Delta_\text{off} \rVert_2 / (\lVert \Delta_\text{on} \rVert_2 + \epsilon)$ (or $/ (|\Delta_\text{on}| + \epsilon)$ for scalar on-target).
  Less brittle than max ratio.
  """

  OFF_TARGET_MAX_ABS = "off_target_max_abs"
  """
  $\max(|\Delta_\text{off}|)$ (raw units).
  Absolute leakage cap for cases where $|\Delta_\text{on}|$ is tiny and ratios are unstable.
  """

  ON_TARGET_MAG = "on_target_mag"
  """
  $|\Delta_\text{on}|$ (or $\lVert \Delta_\text{on} \rVert_2$ for multi-dim on-target).
  Use as a non-triviality guard and for debugging thresholds.
  """

  def __lt__(self, other: int | float) -> MetricExpr:
    return MetricExpr(self, ComparisonOperator.LT, float(other))

  def __le__(self, other: int | float) -> MetricExpr:
    return MetricExpr(self, ComparisonOperator.LE, float(other))

  def __gt__(self, other: int | float) -> MetricExpr:
    return MetricExpr(self, ComparisonOperator.GT, float(other))

  def __ge__(self, other: int | float) -> MetricExpr:
    return MetricExpr(self, ComparisonOperator.GE, float(other))

  def eq(self, other: int | float) -> MetricExpr:
    return MetricExpr(self, ComparisonOperator.EQ, float(other))

  def ne(self, other: int | float) -> MetricExpr:
    return MetricExpr(self, ComparisonOperator.NE, float(other))


class BoolExpr(NamedTuple):
  op: BoolOp
  terms: tuple[Expr, ...]

  def __bool__(self) -> bool:
    raise TypeError("BoolExpr is not a boolean. Call .evaluate(metrics).")

  def __and__(self, other: Expr) -> BoolExpr:
    return BoolExpr._and(self, other)

  def __or__(self, other: Expr) -> BoolExpr:
    return BoolExpr._or(self, other)

  @staticmethod
  def _and(left: Expr, right: Expr) -> BoolExpr:
    left_terms = (
      left.terms if isinstance(left, BoolExpr) and left.op is BoolOp.AND else (left,)
    )
    right_terms = (
      right.terms if isinstance(right, BoolExpr) and right.op is BoolOp.AND else (right,)
    )
    return BoolExpr(BoolOp.AND, left_terms + right_terms)

  @staticmethod
  def _or(left: Expr, right: Expr) -> BoolExpr:
    left_terms = (
      left.terms if isinstance(left, BoolExpr) and left.op is BoolOp.OR else (left,)
    )
    right_terms = (
      right.terms if isinstance(right, BoolExpr) and right.op is BoolOp.OR else (right,)
    )
    return BoolExpr(BoolOp.OR, left_terms + right_terms)

  def evaluate(self, metrics: Mapping[Metric, float]) -> bool:
    if self.op is BoolOp.AND:
      return all(term.evaluate(metrics) for term in self.terms)  # type: ignore[attr-defined]
    if self.op is BoolOp.OR:
      return any(term.evaluate(metrics) for term in self.terms)  # type: ignore[attr-defined]
    raise AssertionError(f"Unhandled BoolOp: {self.op}")

  def failed_leaves(
    self, metrics: Mapping[Metric, float]
  ) -> list[tuple[MetricExpr, float]]:
    """
    Returns leaf predicates that fail, along with their actual metric values.
    Useful for error messages.
    """
    out: list[tuple[MetricExpr, float]] = []

    def visit(e: Expr) -> None:
      if isinstance(e, MetricExpr):
        actual = float(metrics[e.metric])
        if not e.comparison_operator.compare(actual, e.value):
          out.append((e, actual))
        return
      for t in e.terms:
        visit(t)

    visit(self)
    return out
