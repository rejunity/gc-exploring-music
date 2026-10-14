"""MAP-Elites archive over complexity descriptors, with a starred global best.

    uv run python -m evomusic.mapelites --lm models/critic_lm.pt --disc models/critic_disc.pt --iters 400 --out out/me1

Axes (the "knobs"): notes per bar (density) x melodic symbol entropy (variety).
Every filled cell keeps its best melody; the global best is exported as star_*.mid.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from evomusic.complexity import descriptors
from evomusic.evolve import crossover, mutate
from evomusic.fitness.factory import build
from evomusic.genome import Genome

AXES = {
    "notes_per_bar": (2.0, 8.0),
    "symbol_entropy": (2.5, 5.0),
}


def cell(g: Genome, bins: int) -> tuple[int, int]:
    d = descriptors_fast(g)
    out = []
    for k, (lo, hi) in AXES.items():
        x = (d[k] - lo) / (hi - lo)
        out.append(min(bins - 1, max(0, int(x * bins))))
    return tuple(out)


def descriptors_fast(g: Genome) -> dict:
    from evomusic.complexity import symbol_entropy
    return {"notes_per_bar": len(g.pitches()) / g.bars, "symbol_entropy": symbol_entropy(g)}


def run(fitness, iters: int, batch: int, bins: int, seed: int, init: int, log=lambda *a: print(*a, flush=True), seeds: list[Genome] | None = None):
    rng = random.Random(seed)
    archive: dict[tuple, tuple[float, Genome]] = {}

    def insert(gs):
        fs = fitness.score_all(gs)
        for g, f in zip(gs, fs):
            c = cell(g, bins)
            if c not in archive or f > archive[c][0]:
                archive[c] = (f, g)

    insert((seeds or []) + [Genome.random(rng=rng, in_scale=rng.random() < 0.5) for _ in range(init)])
    t0 = time.time()
    for it in range(iters):
        elites = [g for _, g in archive.values()]
        kids = []
        for _ in range(batch):
            a = rng.choice(elites)
            if rng.random() < 0.5:
                a = crossover(a, rng.choice(elites), rng)
            kids.append(mutate(a, rng, rng.choice([0.7, 1.5, 3.0])))
        insert(kids)
        if it % 50 == 0 or it == iters - 1:
            fs = [f for f, _ in archive.values()]
            log(f"iter {it:4d}  cells {len(archive):3d}/{bins * bins}  best {max(fs):.3f}  mean {sum(fs) / len(fs):.3f}  {time.time() - t0:.0f}s")
    return archive


def export(archive, fitness, out: Path, bins: int) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    best_c = max(archive, key=lambda c: archive[c][0])
    cells = []
    for c, (f, g) in sorted(archive.items()):
        name = (f"star_{c[0]}_{c[1]}_{f:.3f}" if c == best_c else f"cell_{c[0]}_{c[1]}_{f:.3f}")
        g.to_midi().save(out / f"{name}.mid")
        cells.append({"file": f"{name}.mid", "cell": c, "fitness": f, "chords": g.chord_names(),
                      "descriptors": {k: round(v, 3) for k, v in descriptors(g).items()},
                      "melody": g.melody, "chord_ids": g.chords})
    bd = fitness.breakdown(archive[best_c][1])
    summary = {"axes": AXES, "bins": bins, "star": cells[[x["cell"] for x in cells].index(best_c)]["file"],
               "star_breakdown": bd, "saved": cells}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=list))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=400)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--bins", type=int, default=8)
    ap.add_argument("--init", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rules", type=float, default=1.0)
    ap.add_argument("--ngram", type=Path)
    ap.add_argument("--lm", type=Path)
    ap.add_argument("--disc", type=Path, nargs="*")
    ap.add_argument("--disc-w", type=float, default=4.0)
    ap.add_argument("--lm-w", type=float, default=1.0)
    ap.add_argument("--desc", type=Path, help="descriptor bands json (fitness.descriptors fit)")
    ap.add_argument("--desc-w", type=float, default=2.0)
    ap.add_argument("--amt-w", type=float, default=0.0, help="Anticipatory Music Transformer term weight (GPU, ~15 genomes/s)")
    ap.add_argument("--out", type=Path, default=Path("out/me"))
    ap.add_argument("--resume", type=Path, nargs="*", default=[], help="summary.json files whose pieces seed the archive (continue a lineage)")
    a = ap.parse_args()
    fitness = build(a.rules, a.ngram, 0.5, a.lm, a.lm_w, a.disc or None, a.disc_w, a.desc, a.desc_w, a.amt_w)
    seeds = []
    for path in a.resume:
        if path.exists():
            for c in json.loads(path.read_text()).get("saved", []):
                if "melody" in c and "chord_ids" in c:
                    seeds.append(Genome([tuple(x) for x in c["melody"]], c["chord_ids"], 8))
    if seeds:
        print(f"resuming from {len(seeds)} pieces", flush=True)
    archive = run(fitness, a.iters, a.batch, a.bins, a.seed, a.init, seeds=seeds)
    s = export(archive, fitness, a.out, a.bins)
    print("star:", s["star"], {k: round(v, 3) for k, v in s["star_breakdown"].items()})


if __name__ == "__main__":
    main()
