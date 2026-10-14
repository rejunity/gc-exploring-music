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
# Duration prior (16ths) from Hooktheory 8-bar windows: 8ths 47 %, 16ths 18 %, quarters 18 %, ...
DUR_PRIOR = {1: 0.18, 2: 0.47, 3: 0.05, 4: 0.18, 6: 0.04, 8: 0.04, 12: 0.01, 16: 0.02}
DEFAULT_DURS = list(DUR_PRIOR)
MELODY_LO, MELODY_HI = 55, 79      # init register: corpus melody pitch p10..p90 is ~55..79 (median 67)


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

    def canonical(self) -> "Genome":
        """Merge consecutive rests within a bar (how the corpus encodes silence). Inaudible
        difference, but fragmented rest tokens would otherwise be a giveaway to the critic."""
        bars = []
        for b in self.bar_genes():
            nb = []
            for p, d in b:
                if p is None and nb and nb[-1][0] is None:
                    nb[-1] = (None, nb[-1][1] + d)
                else:
                    nb.append((p, d))
            bars.append(nb)
        return Genome.from_bars(bars, self.chords, dict(self.meta))

    def copy(self) -> "Genome":
        return Genome(list(self.melody), list(self.chords), self.bars, dict(self.meta))

    # ---- construction -----------------------------------------------------
    @classmethod
    def random(cls, bars: int = 8, lo: int = MELODY_LO, hi: int = MELODY_HI, rest_p: float = 0.1,
               in_scale: bool = False, rng: random.Random | None = None) -> "Genome":
        """Fully random (default) or weakly constrained (in_scale=True) individual."""
        rng = rng or random.Random()
        total, t, genes = bars * STEPS_PER_BAR, 0, []
        while t < total:
            d = min(rng.choices(DEFAULT_DURS, weights=list(DUR_PRIOR.values()))[0], STEPS_PER_BAR - t % STEPS_PER_BAR)  # never cross a bar line
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
    def to_midi(self, bpm: int = 110, transpose: int = 0, style: str = "pop", melody_program: int = 0,
                chord_program: int | None = None, bass_program: int = 33, melody_velocity: int = 96,
                chord_velocity: int = 52, bass_velocity: int = 72, chord_octave: int = 4, bass: bool = True) -> mido.MidiFile:
        """Render melody + accompaniment.

        style="pop"  : lead piano on top; electric-piano chord stabs on beats 2 and 4 voiced
                       below the melody; finger bass on the root on beats 1 and 3 (separate tracks,
                       so "rhythm" and "lead" are audibly separated).
        style="block": the original sustained block chords + bass in one piano track.
        """
        mid = mido.MidiFile(ticks_per_beat=PPQ)
        meta = mido.MidiTrack(); mid.tracks.append(meta)
        meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(bpm)))
        meta.append(mido.MetaMessage("time_signature", numerator=4, denominator=4))
        meta.append(mido.MetaMessage("key_signature", key="C"))
        meta.append(mido.MetaMessage("end_of_track", time=self.n_steps * STEP))
        end = self.n_steps * STEP

        def note(events, start, stop, n, vel, ch):
            events.append((start, 1, mido.Message("note_on", note=n + transpose, velocity=vel, channel=ch)))
            events.append((stop - 1, 0, mido.Message("note_off", note=n + transpose, velocity=0, channel=ch)))

        events: list = []
        for start, p, d in self.onsets():
            if p is not None:
                note(events, start * STEP, (start + d) * STEP, p, melody_velocity, 0)
        mid.tracks.append(_track("melody", melody_program, 0, events, end))

        if style == "block":
            events = []
            for i, ci in enumerate(self.chords):
                ch = CHORDS[ci]
                a, b = i * CHORD_STEPS * STEP, (i + 1) * CHORD_STEPS * STEP
                notes = _voice(ch, chord_octave) + ([12 * (chord_octave - 2) + ch.root] if bass else [])
                for n in notes:
                    note(events, a, b, n, chord_velocity, 1)
            mid.tracks.append(_track("chords", chord_program or 0, 1, events, end))
            return mid

        # style == "pop": comping below the lead + separate bass
        lowest = min(self.pitches(), default=60)
        events, bass_ev = [], []
        for i, ci in enumerate(self.chords):
            ch = CHORDS[ci]
            slot = i * CHORD_STEPS * STEP
            voicing = _voice(ch, chord_octave - 1)
            while max(voicing) >= lowest - 2 and min(voicing) > 40:     # keep the comp under the melody
                voicing = [n - 12 for n in voicing]
            for beat in (1, 3) if CHORD_STEPS == 8 else (1,):             # beats 2 and 4 of the bar
                if beat * 4 < CHORD_STEPS:
                    a = slot + beat * 4 * STEP
                    for n in voicing:
                        note(events, a, a + 3 * STEP, n, chord_velocity, 1)
            if bass:
                note(bass_ev, slot, slot + CHORD_STEPS * STEP - STEP, 36 + ch.root if ch.root < 8 else 24 + ch.root, bass_velocity, 2)
        mid.tracks.append(_track("chords", 4 if chord_program is None else chord_program, 1, events, end))
        if bass:
            mid.tracks.append(_track("bass", bass_program, 2, bass_ev, end))
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
