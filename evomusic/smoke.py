"""Smoke test: write a random monophonic 8-bar melody to out/smoke.mid.

This is the "generation 0" individual of the planned genetic algorithm: a
sequence of (pitch, duration) genes with no musical knowledge at all.
"""
from __future__ import annotations

import random
from pathlib import Path

import mido

PPQ = 480  # ticks per quarter note
DURATIONS = [PPQ // 4, PPQ // 2, PPQ, PPQ * 2]  # 16th, 8th, quarter, half


def random_genome(n_bars: int = 8, lo: int = 55, hi: int = 84, seed: int | None = 0):
    rng = random.Random(seed)
    total = n_bars * 4 * PPQ
    t, genes = 0, []
    while t < total:
        dur = min(rng.choice(DURATIONS), total - t)
        pitch = rng.randint(lo, hi)
        rest = rng.random() < 0.1
        genes.append((None if rest else pitch, dur))
        t += dur
    return genes


def genome_to_midi(genes, bpm: int = 110) -> mido.MidiFile:
    mid = mido.MidiFile(ticks_per_beat=PPQ)
    track = mido.MidiTrack()
    mid.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm)))
    track.append(mido.Message("program_change", program=0, time=0))
    pending = 0  # accumulated delta for rests
    for pitch, dur in genes:
        if pitch is None:
            pending += dur
            continue
        track.append(mido.Message("note_on", note=pitch, velocity=80, time=pending))
        track.append(mido.Message("note_off", note=pitch, velocity=0, time=dur))
        pending = 0
    track.append(mido.MetaMessage("end_of_track", time=pending))
    return mid


def main() -> None:
    out = Path("out")
    out.mkdir(exist_ok=True)
    path = out / "smoke.mid"
    genome_to_midi(random_genome()).save(path)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
