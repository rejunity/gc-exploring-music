"""Genome = melody genes + chord genes, and its phenotype as a MIDI file.

Time unit everywhere is the 16th note ("step"). A bar of 4/4 has 16 steps,
a chord slot is half a bar (8 steps). Key is fixed to C major / A minor for
evolution; transposition is a rendering option.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import mido

PPQ = 480
STEP = PPQ // 4            # ticks per 16th
STEPS_PER_BAR = 16
CHORD_STEPS = 8            # one chord per half bar
MAJOR = [0, 2, 4, 5, 7, 9, 11]
DEFAULT_DURS = [1, 2, 2, 3, 4, 4, 6, 8]   # 16ths, weighted towards 8ths/quarters


@dataclass(frozen=True)
class Chord:
    name: str
    root: int              # pitch class relative to tonic
    intervals: tuple[int, ...]   # stacked intervals from root, e.g. (4, 3) = major triad

    def pitch_classes(self) -> list[int]:
        pcs, p = [self.root % 12], self.root
        for i in self.intervals:
            p += i
            pcs.append(p % 12)
        return pcs


# Vocabulary relative to C major. Index = chord gene value.
CHORDS: list[Chord] = [
    Chord("I", 0, (4, 3)), Chord("ii", 2, (3, 4)), Chord("iii", 4, (3, 4)),
    Chord("IV", 5, (4, 3)), Chord("V", 7, (4, 3)), Chord("vi", 9, (3, 4)),
    Chord("vii°", 11, (3, 3)),
    Chord("Imaj7", 0, (4, 3, 4)), Chord("ii7", 2, (3, 4, 3)), Chord("IVmaj7", 5, (4, 3, 4)),
    Chord("V7", 7, (4, 3, 3)), Chord("vi7", 9, (3, 4, 3)),
    Chord("bVII", 10, (4, 3)), Chord("iv", 5, (3, 4)), Chord("bVI", 8, (4, 3)),
    Chord("III", 4, (4, 3)),   # V/vi
    Chord("II", 2, (4, 3)),    # V/V
    Chord("Isus4", 0, (5, 2)), Chord("Vsus4", 7, (5, 2)),
]
CHORD_INDEX = {c.name: i for i, c in enumerate(CHORDS)}


@dataclass
class Genome:
    melody: list[tuple[int | None, int]]   # (midi pitch or None for rest, duration in steps)
    chords: list[int]                      # one CHORDS index per half bar
    bars: int = 8
    meta: dict = field(default_factory=dict)

    # ---- invariants -------------------------------------------------------
    @property
    def n_steps(self) -> int:
        return self.bars * STEPS_PER_BAR

    def check(self) -> None:
        assert sum(d for _, d in self.melody) == self.n_steps, "melody length != bars*16"
        assert all(d > 0 for _, d in self.melody), "non-positive duration"
        assert len(self.chords) == self.bars * STEPS_PER_BAR // CHORD_STEPS, "wrong chord count"
        assert all(0 <= c < len(CHORDS) for c in self.chords)

    # ---- derived views ----------------------------------------------------
    def onsets(self) -> list[tuple[int, int | None, int]]:
        """(start_step, pitch, dur) for every gene."""
        out, t = [], 0
        for p, d in self.melody:
            out.append((t, p, d))
            t += d
        return out

    def chord_at(self, step: int) -> Chord:
        return CHORDS[self.chords[step // CHORD_STEPS]]

    def pitches(self) -> list[int]:
        return [p for p, _ in self.melody if p is not None]

    def bar_genes(self) -> list[list[tuple[int | None, int]]]:
        """Melody split per bar. Genes never cross bar lines (enforced by random() and mutations)."""
        out, cur, t = [], [], 0
        for g in self.melody:
            cur.append(g); t += g[1]
            if t % STEPS_PER_BAR == 0:
                out.append(cur); cur = []
        assert not cur, "gene crosses bar line"
        return out

    @classmethod
    def from_bars(cls, bars: list[list[tuple[int | None, int]]], chords: list[int], meta: dict | None = None) -> "Genome":
        g = cls([x for b in bars for x in b], list(chords), len(bars), meta or {})
        g.check()
        return g

    def copy(self) -> "Genome":
        return Genome(list(self.melody), list(self.chords), self.bars, dict(self.meta))

    # ---- construction -----------------------------------------------------
    @classmethod
    def random(cls, bars: int = 8, lo: int = 60, hi: int = 84, rest_p: float = 0.08,
               in_scale: bool = False, rng: random.Random | None = None) -> "Genome":
        """Fully random (default) or weakly constrained (in_scale=True) individual."""
        rng = rng or random.Random()
        total, t, genes = bars * STEPS_PER_BAR, 0, []
        while t < total:
            d = min(rng.choice(DEFAULT_DURS), STEPS_PER_BAR - t % STEPS_PER_BAR)  # never cross a bar line
            if rng.random() < rest_p:
                genes.append((None, d))
            else:
                p = rng.randint(lo, hi)
                if in_scale:
                    while p % 12 not in MAJOR:
                        p = rng.randint(lo, hi)
                genes.append((p, d))
            t += d
        n_chords = total // CHORD_STEPS
        chords = [rng.randrange(7 if in_scale else len(CHORDS)) for _ in range(n_chords)]
        g = cls(genes, chords, bars)
        g.check()
        return g

    # ---- phenotype --------------------------------------------------------
    def to_midi(self, bpm: int = 110, transpose: int = 0, melody_program: int = 0,
                chord_program: int = 0, melody_velocity: int = 88, chord_velocity: int = 56,
                chord_octave: int = 4, bass: bool = True) -> mido.MidiFile:
        mid = mido.MidiFile(ticks_per_beat=PPQ)
        meta = mido.MidiTrack(); mid.tracks.append(meta)
        meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm)))
        meta.append(mido.MetaMessage("time_signature", numerator=4, denominator=4))
        meta.append(mido.MetaMessage("key_signature", key="C"))
        meta.append(mido.MetaMessage("end_of_track", time=self.n_steps * STEP))

        # melody: absolute-time events then convert to deltas
        events: list[tuple[int, int, mido.Message]] = []   # (tick, order, msg)
        for start, p, d in self.onsets():
            if p is None:
                continue
            events.append((start * STEP, 1, mido.Message("note_on", note=p + transpose, velocity=melody_velocity, channel=0)))
            events.append(((start + d) * STEP - 1, 0, mido.Message("note_off", note=p + transpose, velocity=0, channel=0)))
        mid.tracks.append(_track("melody", melody_program, 0, events, self.n_steps * STEP))

        events = []
        for i, ci in enumerate(self.chords):
            ch = CHORDS[ci]
            start, end = i * CHORD_STEPS * STEP, (i + 1) * CHORD_STEPS * STEP
            notes = _voice(ch, chord_octave)
            if bass:
                notes.append(12 * (chord_octave - 2) + ch.root)
            for n in notes:
                events.append((start, 1, mido.Message("note_on", note=n + transpose, velocity=chord_velocity, channel=1)))
                events.append((end - 1, 0, mido.Message("note_off", note=n + transpose, velocity=0, channel=1)))
        mid.tracks.append(_track("chords", chord_program, 1, events, self.n_steps * STEP))
        return mid

    def chord_names(self) -> str:
        return " ".join(CHORDS[c].name for c in self.chords)


def _voice(ch: Chord, octave: int) -> list[int]:
    """Close voicing around middle C: root at `octave`, upper notes within an octave."""
    base = 12 * octave + ch.root
    notes, p = [base], base
    for i in ch.intervals:
        p += i
        notes.append(p)
    # keep the top note below ~C5 so the melody sits above
    while max(notes) > 12 * (octave + 1) + 4:
        notes = [n - 12 for n in notes]
    return notes


def _track(name: str, program: int, channel: int, events, end_tick: int) -> mido.MidiTrack:
    tr = mido.MidiTrack()
    tr.append(mido.MetaMessage("track_name", name=name))
    tr.append(mido.Message("program_change", program=program, channel=channel, time=0))
    t = 0
    for tick, _, msg in sorted(events, key=lambda e: (e[0], e[1])):
        msg.time = tick - t
        t = tick
        tr.append(msg)
    tr.append(mido.MetaMessage("end_of_track", time=max(0, end_tick - t)))
    return tr
