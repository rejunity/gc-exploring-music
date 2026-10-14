"""Knobs: measurable properties of a piece, expressed as percentiles of real pop.

A knob value of 50 means "typical of real pop", 90 means "more than 90 % of real
pop phrases". Percentiles come from held-out Hooktheory VALID 8-bar windows.

    uv run python -m evomusic.knobs fit      # -> data/knob_quantiles.json (needs the n-gram model)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from evomusic.genome import Genome

ROOT = Path(__file__).resolve().parent.parent
QUANTILES = ROOT / "data" / "knob_quantiles.json"
NGRAM = ROOT / "data" / "processed" / "ngram_hooktheory.json"

KNOBS = {
    "complexity": ("Complexity", "melodic variety: how many different interval and rhythm combinations"),
    "predictability": ("Predictability", "how familiar melody and harmony are under real-pop statistics"),
    "density": ("Density", "notes per bar"),
    "repetition": ("Repetition", "fewer distinct motifs, more literal repeats"),
    "darkness": ("Darkness", "share of minor chords"),
    "harmonic_adventure": ("Harmonic adventure", "chords beyond I, IV, V and vi"),
    "energy_build": ("Energy build", "second half busier than the first"),
}

_ngram = None


def _predictability(g: Genome) -> float:
    global _ngram
    if _ngram is None:
        from evomusic.fitness.ngram import CorpusModel
        _ngram = CorpusModel.load(NGRAM)
    raw = _ngram.raw(g)
    return (raw["melody"] + raw["chords"] + raw["chord_tone"]) / 3


def measure(g: Genome) -> dict[str, float]:
    from evomusic.behaviour import behaviour
    from evomusic.fitness.descriptors import form
    b = behaviour(g)
    return {
        "complexity": b["symbol_entropy"],
        "predictability": _predictability(g),
        "density": b["notes_per_bar"],
        "repetition": -b["distinct_motifs"],
        "darkness": b["minor_share"],
        "harmonic_adventure": 1 - b["four_chord_share"],
        "energy_build": form(g)["density_ratio"],
    }


class Knobs:
    def __init__(self, path: Path = QUANTILES):
        self.q = {k: np.array(v) for k, v in json.loads(Path(path).read_text()).items()}

    def percentiles(self, g: Genome) -> dict[str, float]:
        m = measure(g)
        return {k: float(np.interp(m[k], self.q[k], np.arange(101))) for k in KNOBS}

    @staticmethod
    def match(pct: dict[str, float], targets: dict[str, float]) -> float:
        """1 when every set knob hits its target percentile, falling linearly to 0 at 50 points away."""
        if not targets:
            return 1.0
        return sum(max(0.0, 1 - abs(pct[k] - t) / 50) for k, t in targets.items()) / len(targets)


def fit(out: Path = QUANTILES) -> dict:
    from evomusic.critic.train_disc import real_windows
    real = real_windows("VALID", 4)          # held out: the n-gram model behind "predictability" was fitted on TRAIN
    ms = [measure(g) for g in real]
    q = {k: np.percentile([m[k] for m in ms], np.arange(101)).round(5).tolist() for k in KNOBS}
    out.write_text(json.dumps(q))
    return q


def main(argv: list[str]) -> None:
    if argv[:1] == ["fit"]:
        q = fit()
        for k, v in q.items():
            print(f"{k:20s} p10 {v[10]:8.3f}  p50 {v[50]:8.3f}  p90 {v[90]:8.3f}")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
