"""Read back MIDI files written by Genome.to_midi (melody track + chords track)."""
from __future__ import annotations

from pathlib import Path

import mido

from evomusic.genome import CHORD_STEPS, CHORDS, STEP, STEPS_PER_BAR, Genome

_BY_PCS = {frozenset(c.pitch_classes()): i for i, c in enumerate(CHORDS)}


def _notes(track) -> list[tuple[int, int, int]]:
    t, on, out = 0, {}, []
    for m in track:
        t += m.time
        if m.type == "note_on" and m.velocity > 0:
            on[m.note] = t
        elif m.type in ("note_off", "note_on") and m.note in on:
            s = on.pop(m.note); out.append((s, t, m.note))
    return sorted(out)


def genome_from_midi(path: Path, bars: int = 8) -> Genome:
    mid = mido.MidiFile(path)
    tracks = {next((m.name for m in tr if m.type == "track_name"), ""): tr for tr in mid.tracks}
    scale = STEP * mid.ticks_per_beat / 480
    total = bars * STEPS_PER_BAR
    grid: list[int | None] = [None] * total
    starts = set()
    for s, e, n in _notes(tracks["melody"]):
        a, b = round(s / scale), round((e + 1) / scale)
        starts.add(a)
        for k in range(a, min(b, total)):
            grid[k] = n
    genes, s = [], 0
    while s < total:
        p, e = grid[s], s + 1
        while e < total and e % STEPS_PER_BAR and grid[e] == p and e not in starts:
            e += 1
        genes.append((p, e - s)); s = e
    chords = []
    cn = _notes(tracks["chords"])
    for i in range(total // CHORD_STEPS):
        t0 = i * CHORD_STEPS * scale
        pcs = frozenset(n % 12 for s, e, n in cn if abs(s - t0) < scale)
        bass = min((n for s, e, n in cn if abs(s - t0) < scale), default=0) % 12
        idx = _BY_PCS.get(pcs, _BY_PCS.get(pcs - {bass} | {bass}, None))
        if idx is None:
            idx = next((k for k, c in enumerate(CHORDS) if set(c.pitch_classes()) <= pcs), 0)
        chords.append(idx)
    return Genome.from_bars(Genome(genes, chords, bars).bar_genes(), chords, {"file": str(path)})
