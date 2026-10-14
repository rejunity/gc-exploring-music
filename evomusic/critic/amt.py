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
    def _token_nll(self, gs: list[Genome], bs: int = 4):
        """Per piece: (per-target-token NLL list, bar index of each target token or -1).
        AMT events are (time, duration, note) triplets, time in 1/TIME_RESOLUTION s; pieces are
        rendered at 110 bpm, so bar = onset // bar_units."""
        from anticipation.config import TIME_RESOLUTION
        from anticipation.vocab import TIME_OFFSET
        bar_units = 4 * 60 / 110 * TIME_RESOLUTION
        seqs, bars = [], []
        for g in gs:
            ev = self.events(g)[:1020]
            seqs.append([self.start] + ev)
            b = []
            for k in range(0, len(ev), 3):
                bar = int((ev[k] - TIME_OFFSET) // bar_units)
                b += [bar] * min(3, len(ev) - k)
            bars.append(b)                                       # aligned with targets (= seq[1:])
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
            nll = torch.nn.functional.cross_entropy(logits.transpose(1, 2), x[:, 1:], reduction="none").cpu()
            for j, s in enumerate(chunk):
                n = len(s) - 1
                out.append((nll[j, :n].tolist(), bars[i + j][:n]))
        return out

    def nll(self, gs: list[Genome], bs: int = 4, novel_only: bool = False) -> list[float]:
        """Mean token NLL per piece; novel_only restricts the mean to bars that are not exact repeats."""
        from evomusic.critic.tokens import novel_bars
        out = []
        for g, (tok, bar) in zip(gs, self._token_nll(gs, bs)):
            keep = [True] * len(tok)
            if novel_only:
                nov = novel_bars(g)
                keep = [0 <= b < len(nov) and nov[b] for b in bar]
                if not any(keep):
                    keep = [True] * len(tok)
            vals = [t for t, k in zip(tok, keep) if k]
            out.append(sum(vals) / max(1, len(vals)))
        return out

    def bar_nlls(self, gs: list[Genome], bs: int = 4, n_bars: int = 8) -> list[list[float | None]]:
        """Per piece, the mean token NLL of each bar (None for bars without events)."""
        out = []
        for tok, bar in self._token_nll(gs, bs):
            sums, counts = [0.0] * n_bars, [0] * n_bars
            for t, b in zip(tok, bar):
                if 0 <= b < n_bars:
                    sums[b] += t; counts[b] += 1
            out.append([sums[b] / counts[b] if counts[b] else None for b in range(n_bars)])
        return out


DENSITY_BUCKETS = [(0, 2.75), (2.75, 3.5), (3.5, 5), (5, 7), (7, 99)]


def _density(g: Genome) -> float:
    return len(g.pitches()) / g.bars


class AMTTerm:
    """Fitness term from AMT likelihood, band-scored against real music (fitness.band_peak).

    mode "piece": residual = piece NLL - median real NLL of the same density bucket.
    mode "novel_pos": only bars that are not exact repeats count, and each bar's NLL is
        compared with the real-pop median for *that bar position* (bar 1 is always expensive:
        no context yet), so a piece whose repeats come late is not blamed for its early bars;
        then density-corrected like "piece". Repetition cannot lower this score; it is judged
        by the form axes and the discriminators instead.
    """

    def __init__(self, grader: AMTGrader, calib: list[Genome], novel_only: bool = False, mode: str | None = None):
        import statistics as st
        from evomusic.fitness import quantiles5
        self.g = grader
        self.mode = mode or ("novel_pos" if novel_only else "piece")
        if self.mode == "novel_pos":
            from evomusic.critic.tokens import novel_bars
            bn = grader.bar_nlls(calib)
            per_pos = [[] for _ in range(8)]
            for g, bars in zip(calib, bn):
                for p, (v, nov) in enumerate(zip(bars, novel_bars(g))):
                    if v is not None and nov:
                        per_pos[p].append(v)
            self.pos_med = [st.median(v) if v else 0.0 for v in per_pos]
            raw = [self._raw_from_bars(g, bars) for g, bars in zip(calib, bn)]
        else:
            raw = grader.nll(calib)
        self.med = {}
        for lo, hi in DENSITY_BUCKETS:
            xs = [x for c, x in zip(calib, raw) if lo <= _density(c) < hi]
            self.med[(lo, hi)] = st.median(xs) if xs else st.median(raw)
        self.q = quantiles5([x - self._m(c) for c, x in zip(calib, raw)])

    def _raw_from_bars(self, g: Genome, bars) -> float:
        from evomusic.critic.tokens import novel_bars
        nov = novel_bars(g)
        vals = [v - self.pos_med[p] for p, (v, n) in enumerate(zip(bars, nov)) if v is not None and n]
        if not vals:
            vals = [v - self.pos_med[p] for p, v in enumerate(bars) if v is not None]
        return sum(vals) / max(1, len(vals))

    def _m(self, g: Genome) -> float:
        d = _density(g)
        return next(v for (lo, hi), v in self.med.items() if lo <= d < hi)

    def _score(self, r: float) -> float:
        from evomusic.fitness import band_peak
        return band_peak(r, *self.q)

    def batch(self, gs: list[Genome]) -> list[float]:
        if self.mode == "novel_pos":
            raw = [self._raw_from_bars(g, bars) for g, bars in zip(gs, self.g.bar_nlls(gs))]
        else:
            raw = self.g.nll(gs)
        return [self._score(x - self._m(g)) for g, x in zip(gs, raw)]

    def __call__(self, g: Genome) -> float:
        return self.batch([g])[0]
