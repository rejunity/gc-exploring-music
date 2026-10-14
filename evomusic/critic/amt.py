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
    def nll(self, gs: list[Genome], bs: int = 4, novel_only: bool = False) -> list[float]:
        """Mean token NLL. novel_only: only events in bars that are not exact repeats of an
        earlier bar count (see Critic.nll). AMT events are (time, duration, note) triplets with
        time in 10 ms units; pieces are rendered at 110 bpm, so a bar is 2400 units * 4 / 110..."""
        from anticipation.config import TIME_OFFSET
        from evomusic.critic.tokens import novel_bars
        bar_units = 4 * 60 / 110 * 100                     # 10 ms units per bar at 110 bpm
        seqs, masks = [], []
        for g in gs:
            ev = self.events(g)[:1020]
            seqs.append([self.start] + ev)
            if novel_only:
                nov = novel_bars(g)
                keep = [False]
                for k in range(0, len(ev), 3):
                    bar = int((ev[k] - TIME_OFFSET) // bar_units)
                    ok = 0 <= bar < len(nov) and nov[bar]
                    keep += [ok] * min(3, len(ev) - k)
                if not any(keep):
                    keep = [True] * len(keep)
                masks.append(keep)
        out = []
        for i in range(0, len(seqs), bs):
            chunk = seqs[i:i + bs]
            T = max(map(len, chunk))
            x = torch.zeros(len(chunk), T, dtype=torch.long)
            m = torch.zeros(len(chunk), T, dtype=torch.bool)
            w = torch.zeros(len(chunk), T, dtype=torch.bool)
            for j, s in enumerate(chunk):
                x[j, :len(s)] = torch.tensor(s); m[j, :len(s)] = True
                w[j, :len(s)] = torch.tensor(masks[i + j]) if novel_only else True
            x, m, w = x.to(self.device), m.to(self.device), w.to(self.device)
            logits = self.model(x, attention_mask=m.long()).logits[:, :-1]
            tgt, tm = x[:, 1:], (m & w)[:, 1:]
            nll = torch.nn.functional.cross_entropy(logits.transpose(1, 2), tgt, reduction="none")
            out += ((nll * tm).sum(1) / tm.sum(1).clamp(min=1)).tolist()
        return out


DENSITY_BUCKETS = [(0, 2.75), (2.75, 3.5), (3.5, 5), (5, 7), (7, 99)]


def _density(g: Genome) -> float:
    return len(g.pitches()) / g.bars


class AMTTerm:
    """Fitness term from AMT likelihood, density-matched and band-scored.

    residual = NLL - median real NLL of the same density bucket (calibrated on VALID windows).
    Score peaks at the real residual median, 0.85 at p25 / p75, 0 at p5 / p95: rewards being
    as natural as typical real music, not the single most predictable continuation.
    """

    def __init__(self, grader: AMTGrader, calib: list[Genome], novel_only: bool = False):
        import statistics as st
        self.g = grader
        self.novel_only = novel_only
        nll = grader.nll(calib, novel_only=novel_only)
        self.med = {}
        for lo, hi in DENSITY_BUCKETS:
            xs = [x for c, x in zip(calib, nll) if lo <= _density(c) < hi]
            self.med[(lo, hi)] = st.median(xs) if xs else st.median(nll)
        from evomusic.fitness import quantiles5
        res = [x - self._m(c) for c, x in zip(calib, nll)]
        self.q = quantiles5(res)

    def _m(self, g: Genome) -> float:
        d = _density(g)
        return next(v for (lo, hi), v in self.med.items() if lo <= d < hi)

    def _score(self, r: float) -> float:
        from evomusic.fitness import band_peak
        return band_peak(r, *self.q)

    def batch(self, gs: list[Genome]) -> list[float]:
        return [self._score(x - self._m(g)) for g, x in zip(gs, self.g.nll(gs, novel_only=self.novel_only))]

    def __call__(self, g: Genome) -> float:
        return self.batch([g])[0]
