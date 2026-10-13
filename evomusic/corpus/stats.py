"""Corpus descriptors + how the hand-written rules score real music (calibration).

    uv run python -m evomusic.corpus.stats data/processed/hooktheory.jsonl [--n 3000]
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
from collections import Counter
from pathlib import Path

from evomusic.fitness.rules import default_rules
from evomusic.genome import CHORDS, Genome


def load(path: Path, n: int | None = None, seed: int = 0) -> list[Genome]:
    rows = [json.loads(l) for l in path.open()]
    if n and n < len(rows):
        rows = random.Random(seed).sample(rows, n)
    out = []
    for r in rows:
        meta = {k: v for k, v in r.items() if k not in ("melody", "chords", "bars")}
        out.append(Genome([tuple(x) for x in r["melody"]], r["chords"], r["bars"], meta))
    return out


def describe(gs: list[Genome]) -> dict:
    d = {}
    ivs, ranges, dens, rest, durs, chords = [], [], [], [], Counter(), Counter()
    for g in gs:
        ps = g.pitches()
        if len(ps) > 1:
            iv = [abs(b - a) for a, b in zip(ps, ps[1:])]
            ivs.append(sum(i <= 2 for i in iv) / len(iv))
            ranges.append(max(ps) - min(ps))
        dens.append(len(ps) / g.bars)
        rest.append(sum(d for p, d in g.melody if p is None) / g.n_steps)
        durs.update(d for p, d in g.melody if p is not None)
        chords.update(CHORDS[c].name for c in g.chords)
    q = lambda xs: tuple(round(x, 2) for x in st.quantiles(xs, n=10)[::3]) if xs else ()   # 10/40/70 th pct
    d["n"] = len(gs)
    d["bars"] = Counter(g.bars for g in gs).most_common(5)
    d["step_fraction p10/40/70"] = q(ivs)
    d["range_semitones p10/40/70"] = q(ranges)
    d["notes_per_bar p10/40/70"] = q(dens)
    d["rest_fraction p10/40/70"] = q(rest)
    tot = sum(durs.values())
    d["duration_16ths"] = {k: round(v / tot, 2) for k, v in sorted(durs.items()) if v / tot > 0.01}
    tot = sum(chords.values())
    d["chords"] = {k: round(v / tot, 3) for k, v in chords.most_common(12)}
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", type=Path)
    ap.add_argument("--n", type=int, default=3000)
    a = ap.parse_args()
    gs = load(a.path, a.n)
    for k, v in describe(gs).items():
        print(f"{k:28s} {v}")
    rules = default_rules()
    scores = [rules(g) for g in gs]
    print(f"\nrules fitness on real segments: mean {st.mean(scores):.3f}  p10 {st.quantiles(scores, n=10)[0]:.3f}  p90 {st.quantiles(scores, n=10)[-1]:.3f}")
    terms = {k: [] for k in rules.terms}
    for g in gs:
        for k, v in rules.breakdown(g).items():
            terms[k].append(v)
    print("per term mean:", {k: round(st.mean(v), 2) for k, v in terms.items()})


if __name__ == "__main__":
    main()
