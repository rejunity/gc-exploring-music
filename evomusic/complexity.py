"""Complexity measures for a Genome ("produced complexity" of the output).

- entropy_rate: block-entropy rate h = H(n+1) - H(n) of the melody token
  sequence (interval, duration). For a stationary symbolic process this
  converges to the Kolmogorov-Sinai entropy under that partition. Melodies
  are short, so n is small (2) and the estimate is biased low; use it
  comparatively (evolved vs real), not as an absolute number.
- compression: zlib(compressed) / raw length of the token bytes; a
  Kolmogorov-complexity proxy (lower = more repetitive structure).
- lyapunov: largest Lyapunov exponent (Rosenstein) of the pitch series via a
  delay embedding. Experimental: series of 30-60 notes are very short.
"""
from __future__ import annotations

import math
import zlib
from collections import Counter

import numpy as np

from evomusic.genome import Genome


def melody_symbols(g: Genome) -> list[tuple]:
    out, prev = [], None
    for p, d in g.melody:
        if p is None:
            out.append(("R", d))
        else:
            out.append((0 if prev is None else p - prev, d))
            prev = p
    return out


def block_entropy(seq: list, n: int) -> float:
    if len(seq) < n or n == 0:
        return 0.0
    c = Counter(tuple(seq[i:i + n]) for i in range(len(seq) - n + 1))
    tot = sum(c.values())
    return -sum(v / tot * math.log2(v / tot) for v in c.values())


def entropy_rate(g: Genome, n: int = 2) -> float:
    s = melody_symbols(g)
    return block_entropy(s, n + 1) - block_entropy(s, n)


def symbol_entropy(g: Genome) -> float:
    return block_entropy(melody_symbols(g), 1)


def compression(g: Genome) -> float:
    raw = bytes(((p if p is not None else 0) & 0x7F) for p, d in g.melody for _ in range(d))   # step grid
    raw += bytes(c for c in g.chords)
    return len(zlib.compress(raw, 9)) / max(1, len(raw))


def lyapunov(g: Genome, emb: int = 3, lag: int = 1, horizon: int = 6) -> float:
    """Rosenstein's method on the step-sampled pitch series (rests hold the previous pitch)."""
    series, last = [], None
    for p, d in g.melody:
        last = p if p is not None else last
        series += [last if last is not None else 60] * d
    x = np.asarray(series[::2], dtype=float)           # 8th-note sampling
    N = len(x) - (emb - 1) * lag
    if N < horizon + 10:
        return float("nan")
    Y = np.stack([x[i * lag:i * lag + N] for i in range(emb)], 1)
    D = np.linalg.norm(Y[:, None] - Y[None], axis=-1)
    for i in range(N):
        D[i, max(0, i - 4):i + 5] = np.inf               # Theiler window
    nn_ = D.argmin(1)
    div = []
    for k in range(1, horizon + 1):
        ok = [(i, j) for i, j in enumerate(nn_) if i + k < N and j + k < N]
        d = [np.linalg.norm(Y[i + k] - Y[j + k]) for i, j in ok]
        d = [v for v in d if v > 0]
        div.append(np.mean(np.log(d)) if d else np.nan)
    div = np.array(div)
    ks = np.arange(1, horizon + 1)
    m = ~np.isnan(div)
    return float(np.polyfit(ks[m], div[m], 1)[0]) if m.sum() >= 3 else float("nan")


def descriptors(g: Genome) -> dict[str, float]:
    ps = g.pitches()
    iv = [abs(b - a) for a, b in zip(ps, ps[1:])]
    return {
        "notes_per_bar": len(ps) / g.bars,
        "pitch_range": (max(ps) - min(ps)) if ps else 0,
        "step_fraction": sum(i <= 2 for i in iv) / len(iv) if iv else 0,
        "rest_fraction": sum(d for p, d in g.melody if p is None) / g.n_steps,
        "distinct_chords": len(set(g.chords)),
        "chord_changes": sum(a != b for a, b in zip(g.chords, g.chords[1:])) / max(1, len(g.chords) - 1),
        "symbol_entropy": symbol_entropy(g),
        "entropy_rate": entropy_rate(g),
        "compression": compression(g),
        "lyapunov": lyapunov(g),
    }
