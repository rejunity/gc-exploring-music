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
        return {n: fn(g) for n, (_, fn) in self.terms.items()}

    def __call__(self, g: Genome) -> float:
        total = sum(w for w, _ in self.terms.values())
        return sum(w * fn(g) for w, fn in self.terms.values()) / total
