"""REMI-style tokenizer for Genome (melody + chords, 4/4, 16th grid).

Per bar:  BAR, then events ordered by position:
          POS_p  [CH_c if a chord slot starts at p]  [PITCH_x | REST  DUR_d  if a gene starts at p]
Sequence: <bos> bar* <eos>
"""
from __future__ import annotations

import random

from evomusic.genome import CHORD_STEPS, CHORDS, STEPS_PER_BAR, Genome

PITCH_LO, PITCH_HI = 36, 96
SPECIAL = ["<pad>", "<bos>", "<eos>", "BAR", "REST"]
VOCAB = (SPECIAL + [f"POS_{i}" for i in range(STEPS_PER_BAR)] + [f"CH_{i}" for i in range(len(CHORDS))]
         + [f"P_{p}" for p in range(PITCH_LO, PITCH_HI + 1)] + [f"D_{d}" for d in range(1, STEPS_PER_BAR + 1)])
TOK = {t: i for i, t in enumerate(VOCAB)}
PAD, BOS, EOS = TOK["<pad>"], TOK["<bos>"], TOK["<eos>"]


def encode(g: Genome) -> list[int]:
    g = g.canonical()
    out = [BOS]
    starts = {}
    for s, p, d in g.onsets():
        starts[s] = (p, d)
    for b in range(g.bars):
        out.append(TOK["BAR"])
        for pos in range(STEPS_PER_BAR):
            step = b * STEPS_PER_BAR + pos
            chord = pos % CHORD_STEPS == 0
            note = step in starts
            if not (chord or note):
                continue
            out.append(TOK[f"POS_{pos}"])
            if chord:
                out.append(TOK[f"CH_{g.chords[step // CHORD_STEPS]}"])
            if note:
                p, d = starts[step]
                out.append(TOK["REST"] if p is None else TOK[f"P_{min(PITCH_HI, max(PITCH_LO, p))}"])
                out.append(TOK[f"D_{min(16, d)}"])
    out.append(EOS)
    return out


def windows(g: Genome, bars: int = 8, hop: int = 4) -> list[Genome]:
    """Bar-aligned windows of `bars` bars (the GA's phrase length)."""
    if g.bars <= bars:
        return [g]
    gb = g.bar_genes()
    cpb = len(g.chords) // g.bars
    out = []
    for i in range(0, g.bars - bars + 1, hop):
        out.append(Genome.from_bars(gb[i:i + bars], g.chords[i * cpb:(i + bars) * cpb], dict(g.meta)))
    return out
