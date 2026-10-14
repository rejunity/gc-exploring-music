"""Independent grader: Anticipatory Music Transformer (Thickstun et al. 2023,
stanford-crfm/music-medium-800k, trained on Lakh MIDI). It has never seen
Hooktheory, so it is a held-out judge. Never optimise against it.

Score = mean next-token NLL of the AMT event sequence (lower = more like
Lakh music). Genomes are rendered exactly like the site clips (110 bpm, piano).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from evomusic.genome import Genome

MODEL = "stanford-crfm/music-medium-800k"


class AMTGrader:
    def __init__(self, name: str = MODEL, device: str | None = None):
        from transformers import AutoModelForCausalLM
        from anticipation.vocab import AUTOREGRESS
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(name).to(self.device).eval()
        self.start = AUTOREGRESS

    def events(self, g: Genome) -> list[int]:
        from anticipation.convert import midi_to_events
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.mid"
            g.to_midi().save(p)
            return midi_to_events(str(p))

    @torch.no_grad()
    def nll(self, gs: list[Genome], bs: int = 4) -> list[float]:
        seqs = [[self.start] + self.events(g)[:1020] for g in gs]
        out = []
        for i in range(0, len(seqs), bs):
            chunk = seqs[i:i + bs]
            T = max(map(len, chunk))
            x = torch.zeros(len(chunk), T, dtype=torch.long)
            m = torch.zeros(len(chunk), T, dtype=torch.bool)
            for j, s in enumerate(chunk):
                x[j, :len(s)] = torch.tensor(s); m[j, :len(s)] = True
            x, m = x.to(self.device), m.to(self.device)
            logits = self.model(x, attention_mask=m.long()).logits[:, :-1]
            tgt, tm = x[:, 1:], m[:, 1:]
            nll = torch.nn.functional.cross_entropy(logits.transpose(1, 2), tgt, reduction="none")
            out += ((nll * tm).sum(1) / tm.sum(1)).tolist()
        return out
