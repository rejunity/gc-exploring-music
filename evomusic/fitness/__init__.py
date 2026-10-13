"""Fitness terms. Every term is `f(genome) -> float` in [0, 1]; combine with `Composite`."""
from __future__ import annotations

from dataclasses import dataclass, field

from evomusic.genome import Genome


@dataclass
class Composite:
    terms: dict[str, tuple[float, object]] = field(default_factory=dict)   # name -> (weight, fn)

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

    def score_all(self, gs: list[Genome]) -> list[float]:
        vals = self.term_values(gs)
        total = sum(w for w, _ in self.terms.values())
        return [sum(w * vals[n][i] for n, (w, _) in self.terms.items()) / total for i in range(len(gs))]
