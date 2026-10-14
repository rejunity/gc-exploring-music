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
        with tempfile.TemporaryDirectory(dir="/dev/shm" if Path("/dev/shm").exists() else None) as d:
            p = Path(d) / "x.mid"
            g.canonical().to_midi().save(p)
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


DENSITY_BUCKETS = [(0, 2.75), (2.75, 3.5), (3.5, 5), (5, 7), (7, 99)]


def _density(g: Genome) -> float:
    return len(g.pitches()) / g.bars


class AMTTerm:
    """Fitness term from AMT likelihood, density-matched and band-scored.

    residual = NLL - median real NLL of the same density bucket (calibrated on VALID windows).
    Score 1 inside the real residual p25..p75, linear decay to 0 at p5 / p95: rewards being as
    natural as typical real music, not the single most predictable continuation.
    """

    def __init__(self, grader: AMTGrader, calib: list[Genome]):
        import statistics as st
        self.g = grader
        nll = grader.nll(calib)
        self.med = {}
        for lo, hi in DENSITY_BUCKETS:
            xs = [x for c, x in zip(calib, nll) if lo <= _density(c) < hi]
            self.med[(lo, hi)] = st.median(xs) if xs else st.median(nll)
        res = [x - self._m(c) for c, x in zip(calib, nll)]
        q = st.quantiles(res, n=20)
        self.q = (q[0], q[4], q[14], q[18])

    def _m(self, g: Genome) -> float:
        d = _density(g)
        return next(v for (lo, hi), v in self.med.items() if lo <= d < hi)

    def _score(self, r: float) -> float:
        q5, q25, q75, q95 = self.q
        if q25 <= r <= q75:
            return 1.0
        if r < q25:
            return max(0.0, (r - q5) / (q25 - q5))
        return max(0.0, (q95 - r) / (q95 - q75))

    def batch(self, gs: list[Genome]) -> list[float]:
        return [self._score(x - self._m(g)) for g, x in zip(gs, self.g.nll(gs))]

    def __call__(self, g: Genome) -> float:
        return self.batch([g])[0]
