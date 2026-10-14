"""Fitness terms. Every term is `f(genome) -> float` in [0, 1]; combine with `Composite`."""
from __future__ import annotations

import math
import statistics as st
from dataclasses import dataclass, field

from evomusic.genome import Genome


def quantiles5(xs) -> tuple[float, float, float, float, float]:
    """(p5, p25, p50, p75, p95) of a sample."""
    q = st.quantiles(xs, n=20)
    return (q[0], q[4], q[9], q[14], q[18])


def band_peak(x: float, q5: float, q25: float, q50: float, q75: float, q95: float, shoulder: float = 0.85) -> float:
    """Score a measurement against the distribution of real music: 1 at the real median,
    `shoulder` at p25 / p75, 0 at p5 / p95 and beyond, linear in between.

    Replaces the flat p25..p75 plateau used earlier: a plateau gives no preference inside
    the typical range, so any biased judge decides the direction there; the peak pulls
    toward typical real values without rewarding the single most predictable option."""
    if x <= q5 or x >= q95:
        return 0.0
    pts = [(q5, 0.0), (q25, shoulder), (q50, 1.0), (q75, shoulder), (q95, 0.0)]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            return max(y0, y1) if x1 <= x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return 0.0


def as5(q) -> tuple[float, float, float, float, float]:
    """Accept legacy 2- or 4-quantile tuples and widen them to 5 (median = midpoint)."""
    q = tuple(q)
    if len(q) == 5:
        return q
    if len(q) == 4:
        return (q[0], q[1], (q[1] + q[2]) / 2, q[2], q[3])
    lo, hi = q
    w = hi - lo
    return (lo - w / 4, lo, (lo + hi) / 2, hi, hi + w / 4)


@dataclass
class Composite:
    """Weighted combination of terms.

    agg="geo": weighted geometric mean, a soft AND: a piece scores high only if every judge
               is satisfied, so no single biased judge can carry it (default).
    agg="sum": weighted arithmetic mean (the original).
    """
    terms: dict[str, tuple[float, object]] = field(default_factory=dict)   # name -> (weight, fn)
    agg: str = "geo"
    eps: float = 0.02

    def add(self, name: str, weight: float, fn) -> "Composite":
        self.terms[name] = (weight, fn)
        return self

    def breakdown(self, g: Genome) -> dict[str, float]:
        return {n: v[0] for n, v in self.term_values([g]).items()}

    def __call__(self, g: Genome) -> float:
        return self.score_all([g])[0]

    def term_values(self, gs: list[Genome]) -> dict[str, list[float]]:
        """Per-term values for a population; terms with a `.batch` attribute are scored in one call."""
        return {n: (fn.batch(gs) if hasattr(fn, "batch") else [fn(g) for g in gs]) for n, (_, fn) in self.terms.items()}

    def combine(self, vals: dict[str, list[float]], n: int) -> list[float]:
        total = sum(w for w, _ in self.terms.values())
        if self.agg == "sum":
            return [sum(w * vals[k][i] for k, (w, _) in self.terms.items()) / total for i in range(n)]
        return [math.exp(sum(w * math.log(max(self.eps, vals[k][i])) for k, (w, _) in self.terms.items()) / total) for i in range(n)]

    def score_all(self, gs: list[Genome]) -> list[float]:
        return self.combine(self.term_values(gs), len(gs))
