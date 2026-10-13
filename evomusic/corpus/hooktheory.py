"""Hooktheory (Sheet Sage release) -> Genome.

    uv run python -m evomusic.corpus.hooktheory            # writes data/processed/hooktheory.jsonl + stats

Keeps segments that: have MELODY and HARMONY tags, no TEMPO/METER/KEY changes,
are in 4/4, a major or natural-minor key, and whose melody quantises cleanly to
16ths. Transposes to C major (or A minor for minor keys: same pitch-class set,
so the chord vocabulary still applies). Chords are mapped to the closest entry
of evomusic.genome.CHORDS by pitch-class set; unmatched chords are dropped from
the segment's statistics but the segment is kept (chord gene = nearest match).
"""
from __future__ import annotations

import gzip
import json
import math
from collections import Counter
from pathlib import Path

from evomusic.genome import CHORDS, CHORD_STEPS, STEPS_PER_BAR, Genome

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "data" / "raw" / "hooktheory" / "Hooktheory.json.gz"
OUT = ROOT / "data" / "processed" / "hooktheory.jsonl"

MAJOR_INTERVALS = [2, 2, 1, 2, 2, 2]
MINOR_INTERVALS = [2, 1, 2, 2, 1, 2]
_CHORD_BY_PCS = {frozenset(c.pitch_classes()): i for i, c in enumerate(CHORDS)}


def _q(x: float) -> int | None:
    """Beats -> 16th steps, None if not on the grid (tolerance 1/32)."""
    s = x * 4
    return round(s) if abs(s - round(s)) < 0.13 else None


def _chord_index(root_pc: int, intervals: list[int], tonic: int) -> int:
    pcs, p = [(root_pc - tonic) % 12], root_pc - tonic
    for i in intervals:
        p += i
        pcs.append(p % 12)
    key = frozenset(pcs)
    if key in _CHORD_BY_PCS:
        return _CHORD_BY_PCS[key]
    # nearest by root then triad quality (drop 7ths / extensions)
    tri = frozenset(pcs[:3])
    if tri in _CHORD_BY_PCS:
        return _CHORD_BY_PCS[tri]
    root = pcs[0]
    for i, c in enumerate(CHORDS):
        if c.root == root:
            return i
    return 0


def convert(seg: dict) -> tuple[Genome | None, str]:
    tags = set(seg["tags"])
    if not {"MELODY", "HARMONY"} <= tags or tags & {"TEMPO_CHANGES", "METER_CHANGES", "KEY_CHANGES", "SWING_CHANGES"}:
        return None, "tags"
    ann = seg["annotations"]
    m = ann["meters"][0]
    if (m["beats_per_bar"], m["beat_unit"]) != (4, 4):
        return None, "meter"
    k = ann["keys"][0]
    if k["scale_degree_intervals"] == MAJOR_INTERVALS:
        tonic, mode = k["tonic_pitch_class"], "major"
    elif k["scale_degree_intervals"] == MINOR_INTERVALS:
        tonic, mode = (k["tonic_pitch_class"] + 3) % 12, "minor"   # relative major's tonic
    else:
        return None, "mode"
    n_beats = ann["num_beats"]
    bars = n_beats / 4
    if bars != int(bars) or not (2 <= bars <= 32):
        return None, "length"
    bars = int(bars)
    total = bars * STEPS_PER_BAR

    # melody -> dense step grid
    grid: list[int | None] = [None] * total
    for n in ann["melody"]:
        a, b = _q(n["onset"]), _q(n["offset"])
        if a is None or b is None or b <= a or b > total:
            return None, "grid"
        pitch = 60 + 12 * n["octave"] + ((n["pitch_class"] - tonic) % 12)
        # keep melody in a sensible register: Hooktheory octave 0 = C4..B4 relative to the key
        for s in range(a, b):
            grid[s] = pitch
    # grid -> genes, splitting at bar lines; consecutive identical pitches within a note stay one gene
    genes: list[tuple[int | None, int]] = []
    s = 0
    starts = set()
    for n in ann["melody"]:
        starts.add(_q(n["onset"]))
    while s < total:
        p = grid[s]
        e = s + 1
        while e < total and e % STEPS_PER_BAR != 0 and grid[e] == p and e not in starts:
            e += 1
        genes.append((p, e - s))
        s = e

    chords = [0] * (total // CHORD_STEPS)
    filled = [False] * len(chords)
    for h in ann["harmony"]:
        a, b = _q(h["onset"]), _q(h["offset"])
        if a is None or b is None:
            return None, "grid"
        ci = _chord_index(h["root_pitch_class"], h["root_position_intervals"], tonic)
        for slot in range(a // CHORD_STEPS, min(len(chords), math.ceil(b / CHORD_STEPS))):
            if not filled[slot]:
                chords[slot], filled[slot] = ci, True
    if sum(filled) < len(chords) * 0.75:
        return None, "sparse_harmony"
    for i in range(len(chords)):           # carry forward over gaps
        if not filled[i] and i > 0:
            chords[i] = chords[i - 1]

    g = Genome(genes, chords, bars, {"id": seg["hooktheory"]["id"], "artist": seg["hooktheory"]["artist"],
                                     "song": seg["hooktheory"]["song"], "mode": mode, "split": seg["split"]})
    g.check()
    return g, "ok"


def main() -> None:
    data = json.load(gzip.open(SRC))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    reasons = Counter()
    n = 0
    with OUT.open("w") as f:
        for key, seg in data.items():
            g, why = convert(seg)
            reasons[why] += 1
            if g is None:
                continue
            f.write(json.dumps({"melody": g.melody, "chords": g.chords, "bars": g.bars, **g.meta}) + "\n")
            n += 1
    print(f"kept {n} / {len(data)} segments -> {OUT.relative_to(ROOT)}")
    print("drop reasons:", dict(reasons))


if __name__ == "__main__":
    main()
