"""Plain generational GA over Genome (v1). MAP-Elites comes later.

    uv run python -m evomusic.evolve --gens 300 --pop 300 --out out/run1
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

from evomusic.fitness import Composite
from evomusic.fitness.rules import default_rules
from evomusic.genome import CHORDS, MAJOR, STEPS_PER_BAR, Genome

# ----------------------------------------------------------------------------- mutation

def _scale_step(p: int, k: int) -> int:
    """Move pitch p by k scale degrees of C major (snapping to scale first)."""
    pcs = sorted(MAJOR)
    octave, pc = divmod(p, 12)
    idx = min(range(7), key=lambda i: abs(pcs[i] - pc))
    idx += k
    octave += idx // 7
    return 12 * octave + pcs[idx % 7]


def mutate(g: Genome, rng: random.Random, rate: float = 1.0, lo: int = 45, hi: int = 88) -> Genome:
    """Apply ~rate random operators. Each op keeps the invariants."""
    bars = g.bar_genes()
    chords = list(g.chords)
    n_ops = max(1, round(rng.expovariate(1 / rate)))
    for _ in range(n_ops):
        op = rng.choices(["pitch", "degree", "transpose_bar", "split", "merge", "rest", "repeat_bar",
                          "retrograde", "invert", "chord", "chord_copy", "cadence", "octave"],
                         weights=[4, 4, 1, 2, 2, 1, 1.5, 0.5, 0.5, 3, 1, 0.5, 0.5])[0]
        bi = rng.randrange(len(bars)); bar = bars[bi]
        gi = rng.randrange(len(bar)); p, d = bar[gi]
        if op == "pitch" and p is not None:
            bar[gi] = (min(hi, max(lo, p + rng.choice([-4, -3, -2, -1, 1, 2, 3, 4]))), d)
        elif op == "degree" and p is not None:
            bar[gi] = (min(hi, max(lo, _scale_step(p, rng.choice([-2, -1, 1, 2])))), d)
        elif op == "transpose_bar":
            k = rng.choice([-2, -1, 1, 2])
            bars[bi] = [(None if q is None else min(hi, max(lo, _scale_step(q, k))), e) for q, e in bar]
        elif op == "split" and d >= 2:
            a = rng.randint(1, d - 1)
            q = p if p is None or rng.random() < 0.5 else min(hi, max(lo, _scale_step(p, rng.choice([-1, 1]))))
            bar[gi:gi + 1] = [(p, a), (q, d - a)]
        elif op == "merge" and gi + 1 < len(bar):
            bar[gi:gi + 2] = [(p if p is not None else bar[gi + 1][0], d + bar[gi + 1][1])]
        elif op == "rest":
            bar[gi] = (None if p is not None else rng.randint(lo, hi), d)
        elif op == "repeat_bar":
            bj = rng.randrange(len(bars))
            bars[bj] = list(bar)
        elif op == "retrograde":
            bars[bi] = list(reversed(bar))
        elif op == "invert":
            ps = [q for q, _ in bar if q is not None]
            if ps:
                c = ps[0]
                bars[bi] = [(None if q is None else min(hi, max(lo, 2 * c - q)), e) for q, e in bar]
        elif op == "chord":
            chords[rng.randrange(len(chords))] = rng.randrange(len(CHORDS))
        elif op == "chord_copy":
            i = rng.randrange(len(chords)); j = rng.randrange(len(chords))
            chords[j] = chords[i]
        elif op == "octave":                         # whole melody up/down an octave (register search)
            k = rng.choice([-12, 12])
            flat = [q for b in bars for q, _ in b if q is not None]
            if flat and lo <= min(flat) + k and max(flat) + k <= hi:
                bars = [[(None if q is None else q + k, e) for q, e in b] for b in bars]
        elif op == "cadence":
            chords[-2], chords[-1] = CHORDS.index(next(c for c in CHORDS if c.name == "V")), 0
    return Genome.from_bars(bars, chords)


def crossover(a: Genome, b: Genome, rng: random.Random) -> Genome:
    """Bar-aligned two-point crossover on melody; chords follow the same cut."""
    ba, bb = a.bar_genes(), b.bar_genes()
    n = len(ba)
    i, j = sorted(rng.sample(range(n + 1), 2))
    bars = ba[:i] + bb[i:j] + ba[j:]
    cpb = len(a.chords) // n
    chords = a.chords[:i * cpb] + b.chords[i * cpb:j * cpb] + a.chords[j * cpb:]
    return Genome.from_bars(bars, chords)


# ----------------------------------------------------------------------------- loop

def evolve(fitness: Composite, pop_size: int = 300, gens: int = 300, bars: int = 8, elite: float = 0.05,
           tournament: int = 3, p_cross: float = 0.7, mut_rate: float = 1.5, seed: int = 0,
           in_scale: bool = False, log: Path | None = None, verbose: bool = True) -> tuple[list[Genome], list[float]]:
    rng = random.Random(seed)
    pop = [Genome.random(bars=bars, in_scale=in_scale, rng=rng) for _ in range(pop_size)]
    fit = fitness.score_all(pop)
    rows = []
    n_elite = max(1, int(elite * pop_size))
    t0 = time.time()
    for gen in range(gens):
        order = sorted(range(pop_size), key=lambda i: -fit[i])
        new = [pop[i] for i in order[:n_elite]]
        while len(new) < pop_size:
            def pick():
                cands = rng.sample(range(pop_size), tournament)
                return pop[max(cands, key=lambda i: fit[i])]
            child = crossover(pick(), pick(), rng) if rng.random() < p_cross else pick().copy()
            new.append(mutate(child, rng, mut_rate))
        pop = new
        fit = fitness.score_all(pop)
        best = max(fit)
        rows.append({"gen": gen, "best": best, "mean": sum(fit) / pop_size, "t": time.time() - t0})
        if verbose and (gen % 20 == 0 or gen == gens - 1):
            print(f"gen {gen:4d}  best {best:.3f}  mean {rows[-1]['mean']:.3f}  {rows[-1]['t']:.0f}s")
    if log:
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    return pop, fit


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gens", type=int, default=300)
    ap.add_argument("--pop", type=int, default=300)
    ap.add_argument("--bars", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mut", type=float, default=1.5, help="mean number of mutation ops per child (novelty knob)")
    ap.add_argument("--in-scale", action="store_true", help="weakly constrained initial population")
    ap.add_argument("--out", type=Path, default=Path("out/run"))
    ap.add_argument("--top", type=int, default=5, help="how many distinct top individuals to save")
    ap.add_argument("--corpus", type=Path, help="n-gram model json (evomusic.fitness.ngram fit); adds corpus terms")
    ap.add_argument("--corpus-weight", type=float, default=3.0, help="weight of each corpus stream term")
    a = ap.parse_args(argv)

    fitness = default_rules()
    if a.corpus:
        from evomusic.fitness.ngram import STREAMS, CorpusModel
        model = CorpusModel.load(a.corpus)
        for name in STREAMS:
            fitness.add(f"ngram_{name}", a.corpus_weight, model.term(name))
    pop, fit = evolve(fitness, a.pop, a.gens, a.bars, mut_rate=a.mut, seed=a.seed, in_scale=a.in_scale,
                      log=a.out / "log.csv")
    a.out.mkdir(parents=True, exist_ok=True)
    seen, saved = set(), []
    for i in sorted(range(len(pop)), key=lambda i: -fit[i]):
        key = (tuple(pop[i].melody), tuple(pop[i].chords))
        if key in seen:
            continue
        seen.add(key)
        rank = len(saved)
        name = f"star_{fit[i]:.3f}" if rank == 0 else f"top{rank}_{fit[i]:.3f}"
        pop[i].to_midi().save(a.out / f"{name}.mid")
        saved.append({"file": f"{name}.mid", "fitness": fit[i], "chords": pop[i].chord_names(),
                      "breakdown": fitness.breakdown(pop[i])})
        if len(saved) >= a.top:
            break
    (a.out / "summary.json").write_text(json.dumps({"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}, "saved": saved}, indent=2))
    print("best:", saved[0]["file"], saved[0]["chords"])
    print("breakdown:", {k: round(v, 2) for k, v in saved[0]["breakdown"].items()})


if __name__ == "__main__":
    main()
