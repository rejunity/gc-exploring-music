"""Descriptor-matching fitness: pull a genome's complexity / structure descriptors
into the range real music occupies.

Each descriptor scores 1 inside the real p25..p75 band and decays linearly to 0
at p5 / p95. The term is the mean over descriptors. MAP-Elites axes are left
out (they are the knobs, not targets).

    uv run python -m evomusic.fitness.descriptors fit data/processed/desc_bands.json
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

from evomusic.complexity import compression, entropy_rate, symbol_entropy
from evomusic.genome import Genome

ROOT = Path(__file__).resolve().parent.parent.parent


def _bar_rhythm(b) -> tuple:
    t, out = 0, []
    for p, d in b:
        if p is not None:
            out.append(t)
        t += d
    return tuple(out)


def _bar_contour(b) -> tuple:
    ps = [p for p, _ in b if p is not None]
    return tuple(max(-1, min(1, y - x)) for x, y in zip(ps, ps[1:]))      # up / same / down


def phrase_parallel(g: Genome) -> tuple[float, float]:
    """How much bar k is mirrored by bar k + bars/2 (antecedent / consequent):
    fraction of identical onset rhythms, and of identical up/down contours."""
    bars = g.bar_genes()
    h = len(bars) // 2
    R = [_bar_rhythm(b) for b in bars]
    C = [_bar_contour(b) for b in bars]
    return (sum(R[i] == R[i + h] for i in range(h)) / h, sum(C[i] == C[i + h] for i in range(h)) / h)


def _onsets_jaccard(a: tuple, b: tuple) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if sa | sb else 1.0


def form(g: Genome) -> dict[str, float]:
    """Phrase-level shape: where the melodic climax sits, whether energy builds or fades,
    and how much the answer phrase (bars 5-6) echoes the question (bars 1-2)."""
    bars = g.bar_genes()
    h = len(bars) // 2
    ps = g.pitches()
    top = max(ps) if ps else 0
    climax = next((i for i, b in enumerate(bars) if any(p == top for p, _ in b)), 0)
    n1 = sum(p is not None for b in bars[:h] for p, _ in b)
    n2 = sum(p is not None for b in bars[h:] for p, _ in b)
    R = [_bar_rhythm(b) for b in bars]
    return {
        "climax_pos": climax / max(1, len(bars) - 1),
        "density_ratio": (n2 + 1) / (n1 + 1),
        "period_echo": (_onsets_jaccard(R[0], R[h]) + _onsets_jaccard(R[1], R[h + 1])) / 2,
    }


def cheap_descriptors(g: Genome) -> dict[str, float]:
    pr, pc = phrase_parallel(g)
    ps = g.pitches()
    iv = [abs(b - a) for a, b in zip(ps, ps[1:])]
    return {
        "pitch_range": (max(ps) - min(ps)) if ps else 0,
        "step_fraction": sum(i <= 2 for i in iv) / len(iv) if iv else 0,
        "leap_fraction": sum(i >= 7 for i in iv) / len(iv) if iv else 0,
        "rest_fraction": sum(d for p, d in g.melody if p is None) / g.n_steps,
        "distinct_chords": len(set(g.chords)),
        "chord_changes": sum(a != b for a, b in zip(g.chords, g.chords[1:])) / max(1, len(g.chords) - 1),
        "entropy_rate": entropy_rate(g),
        "compression": compression(g),
        "mean_pitch": st.mean(ps) if ps else 0,
        "parallel_rhythm": pr,
        "parallel_contour": pc,
        **form(g),
    }


def _score(x: float, q5: float, q25: float, q75: float, q95: float) -> float:
    if q25 <= x <= q75:
        return 1.0
    if x < q25:
        return max(0.0, (x - q5) / (q25 - q5)) if q25 > q5 else float(x >= q25)
    return max(0.0, (q95 - x) / (q95 - q75)) if q95 > q75 else float(x <= q75)


def term(bands_path: Path = ROOT / "data/processed/desc_bands.json", exclude: tuple = ()):
    """Mean band score over descriptors; `exclude` drops descriptors that are archive axes (knobs, not targets)."""
    bands = {k: v for k, v in json.loads(bands_path.read_text()).items() if k not in exclude}
    def f(g: Genome) -> float:
        d = cheap_descriptors(g)
        return sum(_score(d[k], *bands[k]) for k in bands) / len(bands)
    f.__name__ = "descriptors"
    f.detail = lambda g: {k: round(_score(v, *bands[k]), 2) for k, v in cheap_descriptors(g).items()}
    return f


def main(argv: list[str]) -> None:
    from evomusic.critic.train_disc import real_windows
    if argv[:1] != ["fit"]:
        sys.exit(__doc__)
    real = real_windows("TRAIN", 4)
    ds = [cheap_descriptors(g) for g in real]
    bands = {}
    for k in ds[0]:
        q = st.quantiles([d[k] for d in ds], n=20)
        bands[k] = (q[0], q[4], q[14], q[18])          # p5, p25, p75, p95
    Path(argv[1]).write_text(json.dumps(bands, indent=1))
    for k, v in bands.items():
        print(f"{k:16s} p5 {v[0]:7.3f}  p25 {v[1]:7.3f}  p75 {v[2]:7.3f}  p95 {v[3]:7.3f}")
    f = term(Path(argv[1]))
    print("mean term on real TRAIN windows:", round(st.mean(f(g) for g in real[:3000]), 3))


if __name__ == "__main__":
    main(sys.argv[1:])
