"""Hand-written music-theory rules, each scoring a genome in [0, 1].

Thresholds are first guesses; M1's corpus statistics should recalibrate them.
"""
from __future__ import annotations

import math
from collections import Counter

from evomusic.fitness import Composite
from evomusic.genome import CHORDS, CHORD_STEPS, MAJOR, STEPS_PER_BAR, Genome


def _band(x: float, lo: float, hi: float, soft: float) -> float:
    """1 inside [lo, hi], decaying linearly to 0 over `soft` outside."""
    if lo <= x <= hi:
        return 1.0
    d = lo - x if x < lo else x - hi
    return max(0.0, 1 - d / soft)


def scale_consistency(g: Genome) -> float:
    ps = g.pitches()
    return sum(p % 12 in MAJOR for p in ps) / len(ps) if ps else 0.0


def chord_tones_on_strong_beats(g: Genome) -> float:
    hits = tot = 0
    for start, p, _ in g.onsets():
        if p is None or start % 4 != 0:          # beats only
            continue
        w = 2 if start % CHORD_STEPS == 0 else 1  # beats 1 and 3 count double
        tot += w
        hits += w * (p % 12 in g.chord_at(start).pitch_classes())
    return hits / tot if tot else 0.0


def interval_profile(g: Genome) -> float:
    ps = g.pitches()
    if len(ps) < 3:
        return 0.0
    iv = [b - a for a, b in zip(ps, ps[1:])]
    steps = sum(abs(i) <= 2 for i in iv) / len(iv)
    big = sum(abs(i) > 12 for i in iv) / len(iv)
    # gap fill: a leap > 4 followed by a step in the opposite direction
    leaps = [(a, b) for a, b in zip(iv, iv[1:]) if abs(a) > 4]
    fill = sum((a > 0) != (b > 0) and abs(b) <= 2 for a, b in leaps) / len(leaps) if leaps else 0.5
    return 0.5 * _band(steps, 0.55, 0.8, 0.4) + 0.3 * fill + 0.2 * (1 - min(1.0, 4 * big))


def pitch_range(g: Genome) -> float:
    ps = g.pitches()
    return _band(max(ps) - min(ps), 7, 16, 8) if ps else 0.0


def rhythm(g: Genome) -> float:
    on = g.onsets()
    offbeat16 = sum(s % 2 == 1 for s, p, _ in on if p is not None)
    n = sum(p is not None for _, p, _ in on)
    if n == 0:
        return 0.0
    durs = Counter(d for _, p, d in on if p is not None)
    ent = -sum(c / n * math.log2(c / n) for c in durs.values())
    density = n / g.bars
    return 0.4 * (1 - min(1.0, 3 * offbeat16 / n)) + 0.3 * _band(ent, 1.0, 2.2, 1.0) + 0.3 * _band(density, 3, 9, 4)


def rests(g: Genome) -> float:
    r = sum(d for p, d in g.melody if p is None) / g.n_steps
    return _band(r, 0.03, 0.2, 0.3)


def cadence(g: Genome) -> float:
    last = [(s, p, d) for s, p, d in g.onsets() if p is not None]
    if not last:
        return 0.0
    s, p, d = last[-1]
    final_chord = CHORDS[g.chords[-1]]
    score = 0.0
    score += 0.4 * (p % 12 == 0)                              # ends on tonic
    score += 0.2 * (p % 12 in final_chord.pitch_classes())    # ends on chord tone
    score += 0.2 * (d >= 4)                                   # long final note
    score += 0.2 * (final_chord.name in ("I", "Imaj7") and CHORDS[g.chords[-2]].name in ("V", "V7", "IV", "Vsus4"))
    return score


def _bar_contour(bar) -> tuple:
    ps = [p for p, _ in bar if p is not None]
    return tuple(b - a for a, b in zip(ps, ps[1:])), tuple(d for _, d in bar)


def repetition(g: Genome) -> float:
    bars = [_bar_contour(b) for b in g.bar_genes()]
    c = Counter(bars)
    repeated = sum(v for v in c.values() if v > 1) / len(bars)   # fraction of bars that have a twin
    return _band(repeated, 0.25, 0.75, 0.5)


_GOOD = {("V", "I"), ("V7", "I"), ("IV", "I"), ("IV", "V"), ("ii", "V"), ("ii7", "V7"), ("I", "IV"), ("I", "V"),
         ("vi", "IV"), ("V", "vi"), ("I", "vi"), ("vi", "ii"), ("iii", "vi"), ("bVII", "I"), ("bVI", "bVII"),
         ("II", "V"), ("III", "vi"), ("Vsus4", "V"), ("Isus4", "I"), ("iv", "I"), ("IVmaj7", "V"), ("Imaj7", "IV"),
         ("vi7", "ii7"), ("I", "ii"), ("IV", "vi"), ("V", "IV")}


def harmony(g: Genome) -> float:
    names = [CHORDS[c].name for c in g.chords]
    pairs = list(zip(names, names[1:]))
    good = sum((a, b) in _GOOD for a, b in pairs if a != b)
    moves = sum(a != b for a, b in pairs) or 1
    stuck = max((len(list(grp)) for _, grp in __import__("itertools").groupby(names)), default=1)
    dim = names.count("vii°") / len(names)
    return 0.6 * good / moves + 0.2 * _band(moves / len(pairs), 0.4, 0.8, 0.5) + 0.1 * (stuck <= 2) + 0.1 * (1 - dim)


def default_rules() -> Composite:
    return (Composite()
            .add("scale", 1.5, scale_consistency)
            .add("chord_tones", 1.5, chord_tones_on_strong_beats)
            .add("intervals", 1.5, interval_profile)
            .add("range", 0.5, pitch_range)
            .add("rhythm", 1.0, rhythm)
            .add("rests", 0.5, rests)
            .add("cadence", 1.0, cadence)
            .add("repetition", 1.0, repetition)
            .add("harmony", 1.5, harmony))
