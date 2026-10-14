"""Small GPT-style backbone with a language-model head and a real/fake head.

`Critic` wraps a trained model for batched scoring of Genomes inside the GA.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from evomusic.critic.tokens import EOS, PAD, VOCAB, encode
from evomusic.genome import Genome


class Backbone(nn.Module):
    def __init__(self, vocab: int = len(VOCAB), d: int = 256, layers: int = 6, heads: int = 8, ctx: int = 512, drop: float = 0.1):
        super().__init__()
        self.cfg = dict(vocab=vocab, d=d, layers=layers, heads=heads, ctx=ctx, drop=drop)
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(ctx, d)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, drop, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.lm = nn.Linear(d, vocab, bias=False)
        self.lm.weight = self.tok.weight
        self.disc = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Dropout(drop), nn.Linear(d, 1))

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        h = self.tok(x) + self.pos(torch.arange(T, device=x.device))
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h = self.enc(h, mask=mask, is_causal=True, src_key_padding_mask=(x == PAD))
        return self.norm(h)

    def forward_lm(self, x: torch.Tensor) -> torch.Tensor:
        return self.lm(self.hidden(x))

    def forward_disc(self, x: torch.Tensor) -> torch.Tensor:
        h = self.hidden(x)
        last = (x != PAD).sum(1) - 1                      # index of <eos>: sees the whole piece (causal)
        return self.disc(h[torch.arange(len(x)), last]).squeeze(-1)


def pad_batch(seqs: list[list[int]], ctx: int, device) -> torch.Tensor:
    seqs = [s[:ctx - 1] + [EOS] if len(s) > ctx else s for s in seqs]
    T = max(map(len, seqs))
    x = torch.full((len(seqs), T), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, :len(s)] = torch.tensor(s)
    return x.to(device)


def lm_nll(model: Backbone, x: torch.Tensor) -> torch.Tensor:
    """Mean next-token NLL per sequence."""
    logits = model.forward_lm(x[:, :-1])
    tgt = x[:, 1:]
    nll = F.cross_entropy(logits.transpose(1, 2), tgt, reduction="none", ignore_index=PAD)
    n = (tgt != PAD).sum(1).clamp(min=1)
    return nll.sum(1) / n


def save(model: Backbone, path: Path, **extra) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"cfg": model.cfg, "state": model.state_dict(), **extra}, path)


def load(path: Path, device="cpu") -> tuple[Backbone, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    m = Backbone(**ck["cfg"]).to(device)
    m.load_state_dict(ck["state"])
    m.eval()
    return m, ck


class Critic:
    """Batched Genome scorer. Exposes fitness terms with a `.batch` method for the GA."""

    def __init__(self, lm_path: Path | None = None, disc_path: Path | list[Path] | None = None, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.lm, self.lm_meta = load(lm_path, self.device) if lm_path else (None, {})
        paths = [] if disc_path is None else (disc_path if isinstance(disc_path, (list, tuple)) else [disc_path])
        self.discs = [load(p, self.device)[0] for p in paths]
        self.disc = self.discs[0] if self.discs else None

    @torch.no_grad()
    def nll(self, gs: list[Genome], bs: int = 128) -> list[float]:
        out = []
        for i in range(0, len(gs), bs):
            x = pad_batch([encode(g) for g in gs[i:i + bs]], self.lm.cfg["ctx"], self.device)
            out += lm_nll(self.lm, x).tolist()
        return out

    @torch.no_grad()
    def p_real(self, gs: list[Genome], bs: int = 128) -> list[float]:
        """Ensemble score: geometric mean of members' p_real (any member rejecting pulls it down)."""
        out = []
        for i in range(0, len(gs), bs):
            x = pad_batch([encode(g) for g in gs[i:i + bs]], self.disc.cfg["ctx"], self.device)
            logp = torch.stack([F.logsigmoid(d.forward_disc(x)) for d in self.discs]).mean(0)
            out += logp.exp().tolist()
        return out

    # ---- fitness terms -----------------------------------------------------
    def lm_term(self):
        if "quantiles" in self.lm_meta:                    # p5, p25, p75, p95 of held-out real NLL
            q5, q25, q75, q95 = self.lm_meta["quantiles"]
        else:
            lo, hi = self.lm_meta["band"]; q5, q25, q75, q95 = lo - (hi - lo) / 4, lo, hi, hi + (hi - lo) / 4
        def score(x: float) -> float:
            if q25 <= x <= q75:
                return 1.0
            if x < q25:
                return max(0.0, (x - q5) / (q25 - q5))
            return max(0.0, (q95 - x) / (q95 - q75))
        f = lambda g: score(self.nll([g])[0])
        f.batch = lambda gs: [score(x) for x in self.nll(gs)]
        f.__name__ = "critic_lm"
        return f

    def disc_term(self):
        f = lambda g: self.p_real([g])[0]
        f.batch = self.p_real
        f.__name__ = "critic_disc"
        return f
