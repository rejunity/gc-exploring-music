"""Corpus-statistics fitness: interpolated n-gram models over melody and chord
token streams, fitted on a processed corpus (see evomusic.corpus.*).

    uv run python -m evomusic.fitness.ngram fit data/processed/hooktheory.jsonl data/processed/ngram_hooktheory.json
    uv run python -m evomusic.fitness.ngram score data/processed/ngram_hooktheory.json out/run/star_*.mid   # (genomes, not midi: see score_jsonl)

Three streams:
  melody : token = (interval class from previous note | REST | START, duration)   n = 4
  chords : token = chord index                                                       n = 3
  chord_tone : token = (pitch class relative to current chord root, strong beat?) n = 1
Fitness = mean per-token log-likelihood, mapped to [0, 1] by the corpus
distribution (corpus median -> 0.5 ... corpus p95 -> 1, random -> ~0), so it
is comparable with the rule terms.
"""
from __future__ import annotations

import json
import math
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

from evomusic.genome import CHORDS, Genome

_D = 0.75  # absolute discount


class NGram:
    """Interpolated absolute-discounting n-gram LM over hashable tokens."""

    def __init__(self, n: int):
        self.n = n
        self.counts: list[dict[tuple, Counter]] = [defaultdict(Counter) for _ in range(n)]   # order k: ctx(len k) -> Counter
        self.vocab: set = set()

    def fit(self, seqs) -> "NGram":
        for s in seqs:
            s = ["<s>"] * (self.n - 1) + list(s)
            for i in range(self.n - 1, len(s)):
                self.vocab.add(s[i])
                for k in range(self.n):
                    self.counts[k][tuple(s[i - k:i])][s[i]] += 1
        return self

    def prob(self, ctx: tuple, tok) -> float:
        p = 1.0 / (len(self.vocab) + 1)       # uniform floor incl. unseen
        for k in range(self.n):
            c = self.counts[k].get(tuple(ctx[len(ctx) - k:]) if k else ())
            if not c:
                continue
            tot = sum(c.values())
            p = max(c.get(tok, 0) - _D, 0) / tot + (_D * len(c) / tot) * p
        return p

    def logprob(self, seq) -> float:
        s = ["<s>"] * (self.n - 1) + list(seq)
        return sum(math.log(self.prob(tuple(s[i - self.n + 1:i]), s[i])) for i in range(self.n - 1, len(s))) / max(1, len(seq))

    def to_json(self) -> dict:
        return {"n": self.n, "vocab": [list(v) if isinstance(v, tuple) else v for v in self.vocab],
                "counts": [[[list(ctx), [[list(t) if isinstance(t, tuple) else t, c] for t, c in cnt.items()]]
                            for ctx, cnt in level.items()] for level in self.counts]}

    @classmethod
    def from_json(cls, d: dict) -> "NGram":
        m = cls(d["n"])
        key = lambda v: tuple(v) if isinstance(v, list) else v
        m.vocab = {key(v) for v in d["vocab"]}
        for k, level in enumerate(d["counts"]):
            for ctx, items in level:
                m.counts[k][tuple(key(c) for c in ctx)] = Counter({key(t): c for t, c in items})
        return m


# ----------------------------------------------------------------------------- tokenisers

def melody_tokens(g: Genome) -> list[tuple]:
    out, prev = [], None
    for p, d in g.melody:
        if p is None:
            out.append(("R", d))
        else:
            iv = "S" if prev is None else max(-12, min(12, p - prev))
            out.append((iv, d))
            prev = p
    return out


def chord_tokens(g: Genome) -> list[int]:
    return list(g.chords)


def chord_tone_tokens(g: Genome) -> list[tuple]:
    out = []
    for start, p, _ in g.onsets():
        if p is None:
            continue
        ch = g.chord_at(start)
        out.append(((p - ch.root) % 12, start % 4 == 0))
    return out


STREAMS = {"melody": (melody_tokens, 4), "chords": (chord_tokens, 3), "chord_tone": (chord_tone_tokens, 1)}


class CorpusModel:
    def __init__(self, models: dict[str, NGram], scale: dict[str, tuple[float, float]]):
        self.models, self.scale = models, scale     # scale: stream -> (corpus p50, corpus p95) of logprob

    @classmethod
    def fit(cls, genomes: list[Genome], calib: list[Genome] | None = None) -> "CorpusModel":
        """Fit on `genomes`; derive the [0,1] scale from `calib` (held-out, so the
        training set's memorised n-grams don't inflate the target)."""
        models = {name: NGram(n).fit(fn(g) for g in genomes) for name, (fn, n) in STREAMS.items()}
        scale = {}
        for name, (fn, _) in STREAMS.items():
            lp = [models[name].logprob(fn(g)) for g in (calib or genomes)]
            q = st.quantiles(lp, n=20)
            scale[name] = (q[9], q[18])
        return cls(models, scale)

    def raw(self, g: Genome) -> dict[str, float]:
        return {name: self.models[name].logprob(fn(g)) for name, (fn, _) in STREAMS.items()}

    def term(self, name: str):
        """A [0,1] fitness term for one stream: p50 -> 0.5, p95 -> 1.0, linear, clipped."""
        fn = STREAMS[name][0]
        lo, hi = self.scale[name]
        def f(g: Genome) -> float:
            x = self.models[name].logprob(fn(g))
            return max(0.0, min(1.0, 0.5 + 0.5 * (x - lo) / (hi - lo)))
        f.__name__ = f"ngram_{name}"
        return f

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"models": {k: m.to_json() for k, m in self.models.items()}, "scale": self.scale}))

    @classmethod
    def load(cls, path: Path) -> "CorpusModel":
        d = json.loads(path.read_text())
        return cls({k: NGram.from_json(v) for k, v in d["models"].items()}, {k: tuple(v) for k, v in d["scale"].items()})


def main(argv: list[str]) -> None:
    from evomusic.corpus.stats import load
    cmd = argv[0]
    if cmd == "fit":
        gs = load(Path(argv[1]))
        train = [g for g in gs if g.meta.get("split") == "TRAIN"]
        valid = [g for g in gs if g.meta.get("split") == "VALID"] or None
        m = CorpusModel.fit(train, valid)
        m.save(Path(argv[2]))
        print(f"fitted on {len(train)} genomes, scale from {len(valid or [])} held-out; (p50, p95):",
              {k: tuple(round(x, 2) for x in v) for k, v in m.scale.items()})
    elif cmd == "score":
        m = CorpusModel.load(Path(argv[1]))
        gs = load(Path(argv[2]), 500)
        for name in STREAMS:
            t = m.term(name)
            print(name, "mean term on corpus sample:", round(st.mean(t(g) for g in gs), 3))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
