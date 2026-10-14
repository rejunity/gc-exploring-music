"""MAP-Elites archive with a starred global best.

    uv run python -m evomusic.mapelites --genome form --cvt data/cvt_hooktheory.json \
        --lm models/critic_lm.pt --disc models/critic_disc_r9.pt --desc data/processed/desc_bands.json --iters 300 --out out/me1

Genomes ("search spaces"):
  notes : flat melody + chords (evomusic.genome.Genome)
  form  : motifs + phrase plan + transformations (evomusic.form.FormGenome)

Archives:
  grid  : the original 2-D knobs, notes per bar x melodic entropy (--bins)
  cvt   : 128 cells placed by k-means on real pop phrases in an 8-D behaviour
          space (evomusic.behaviour); cell weights = share of real pop there

Every filled cell keeps its best piece; the global best is exported as star_*.mid.
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


# ----------------------------------------------------------------------------- search spaces

class NoteSpace:
    name = "notes"

    def random(self, rng):
        return Genome.random(rng=rng, in_scale=rng.random() < 0.5)

    def mutate(self, x, rng, rate):
        return mutate(x, rng, rate)

    def crossover(self, a, b, rng):
        return crossover(a, b, rng)

    def pheno(self, x) -> Genome:
        return x

    def dump(self, x) -> dict:
        return {}

    def load(self, entry: dict):
        return Genome([tuple(t) for t in entry["melody"]], entry["chord_ids"], 8)


class FormSpace:
    name = "form"

    def __init__(self):
        from evomusic.form import load_stats
        self.stats = load_stats()

    def random(self, rng):
        from evomusic.form import random_form
        return random_form(rng, self.stats)

    def mutate(self, x, rng, rate):
        from evomusic.form import mutate_form
        return mutate_form(x, rng, rate, self.stats)

    def crossover(self, a, b, rng):
        from evomusic.form import crossover_form
        return crossover_form(a, b, rng)

    def pheno(self, x) -> Genome:
        return x.to_genome()

    def dump(self, x) -> dict:
        return {"form": x.to_json()}

    def load(self, entry: dict):
        from evomusic.form import FormGenome, from_genome
        if "form" in entry:
            return FormGenome.from_json(entry["form"])
        return from_genome(Genome([tuple(t) for t in entry["melody"]], entry["chord_ids"], 8))[0]


SPACES = {"notes": NoteSpace, "form": FormSpace}


# ----------------------------------------------------------------------------- the algorithm

def run(fitness, iters: int, batch: int, bins: int, seed: int, init: int,
        log=lambda *a: print(*a, flush=True), seeds: list | None = None, space=None, cell_fn=None,
        cvt=None, centre_w: float = 0.0):
    """Returns {cell: (fitness, genotype)}. Defaults (notes genome, 2-D grid) keep the old behaviour.

    With a CVT archive and centre_w > 0, pieces compete within a cell on
    fitness - centre_w * distance-to-centroid: elites are pulled toward where real pop
    is densest instead of drifting to whichever corner of the cell a biased judge likes."""
    space = space or NoteSpace()
    if cvt is not None:
        cell_fn = cvt.cell
    cell_fn = cell_fn or (lambda g: cell(g, bins))
    rng = random.Random(seed)
    archive: dict = {}
    keyed: dict = {}                                  # cell -> competition key

    def insert(xs):
        ph = [space.pheno(x) for x in xs]
        fs = fitness.score_all(ph)
        for x, g, f in zip(xs, ph, fs):
            if cvt is not None and centre_w:
                c, d = cvt.cell_and_distance(g)
                key = f - centre_w * d
            else:
                c, key = cell_fn(g), f
            if c not in keyed or key > keyed[c]:
                keyed[c] = key
                archive[c] = (f, x)

    insert((seeds or []) + [space.random(rng) for _ in range(init)])
    t0 = time.time()
    for it in range(iters):
        elites = [x for _, x in archive.values()]
        kids = []
        for _ in range(batch):
            a = rng.choice(elites)
            if rng.random() < 0.5:
                a = space.crossover(a, rng.choice(elites), rng)
            kids.append(space.mutate(a, rng, rng.choice([0.7, 1.5, 3.0])))
        insert(kids)
        if it % 50 == 0 or it == iters - 1:
            fs = [f for f, _ in archive.values()]
            log(f"iter {it:4d}  cells {len(archive):3d}  best {max(fs):.3f}  mean {sum(fs) / len(fs):.3f}  {time.time() - t0:.0f}s")
    return archive


def _cell_name(c) -> str:
    return f"{c[0]}_{c[1]}" if isinstance(c, tuple) else f"{c:03d}"


def export(archive, fitness, out: Path, bins: int, space=None, cvt=None) -> dict:
    from evomusic.form import from_genome
    space = space or NoteSpace()
    out.mkdir(parents=True, exist_ok=True)
    # the star: best fitness among pieces inside real pop's p10..p90 on every behaviour axis
    # (the global best tends to sit where a biased judge pushed it); fallback: global best
    typical = [c for c, (f, x) in archive.items() if cvt is None or cvt.typical(space.pheno(x))]
    best_c = max(typical or archive, key=lambda c: archive[c][0])
    cells = []
    for c, (f, x) in sorted(archive.items()):
        g = space.pheno(x)
        name = (f"star_{_cell_name(c)}_{f:.3f}" if c == best_c else f"cell_{_cell_name(c)}_{f:.3f}")
        g.to_midi().save(out / f"{name}.mid")
        entry = {"file": f"{name}.mid", "cell": c, "fitness": f, "chords": g.chord_names(),
                 "descriptors": {k: round(v, 3) for k, v in descriptors(g).items()},
                 "plan": " ".join(from_genome(g)[1]),
                 "melody": g.melody, "chord_ids": g.chords, **space.dump(x)}
        if cvt is not None:
            from evomusic.behaviour import behaviour
            entry["behaviour"] = {k: round(v, 3) for k, v in behaviour(g).items()}
            entry["cell_weight"] = cvt.weights[c]
        cells.append(entry)
    bd = fitness.breakdown(space.pheno(archive[best_c][1]))
    summary = {"genome": space.name, "archive": "cvt" if cvt is not None else "grid",
               "axes": cvt.dims if cvt is not None else AXES, "bins": len(cvt) if cvt is not None else bins,
               "star": next(e["file"] for e in cells if e["cell"] == best_c),
               "star_breakdown": bd, "saved": cells}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=list))
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--genome", choices=list(SPACES), default="notes")
    ap.add_argument("--cvt", type=Path, help="behaviour-space centroids (evomusic.behaviour fit); replaces the 2-D grid")
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
    ap.add_argument("--form-w", type=float, default=0.0, help="deprecated: form-likelihood term (penalises the most common real plan)")
    ap.add_argument("--amt-w", type=float, default=0.0, help="Anticipatory Music Transformer term weight (GPU, ~15 genomes/s)")
    ap.add_argument("--out", type=Path, default=Path("out/me"))
    ap.add_argument("--resume", type=Path, nargs="*", default=[], help="summary.json files whose pieces seed the archive (continue a lineage)")
    ap.add_argument("--centre-w", type=float, default=0.1, help="pull toward each cell's real-pop centre (CVT archives only)")
    ap.add_argument("--novel-only", action="store_true", help="likelihood judges score only non-repeated bars")
    ap.add_argument("--agg", choices=["geo", "sum"], default="geo", help="how judges combine: geometric mean (soft AND) or weighted sum")
    a = ap.parse_args()

    space = SPACES[a.genome]()
    cvt = None
    if a.cvt:
        from evomusic.behaviour import CVT
        cvt = CVT(a.cvt)
    # behaviour dimensions are the archive's axes, not targets: keep them out of the descriptor-band term
    exclude = tuple(cvt.dims) + ("parallel_rhythm", "parallel_contour", "distinct_chords") if cvt is not None else ()
    fitness = build(a.rules, a.ngram, 0.5, a.lm, a.lm_w, a.disc or None, a.disc_w, a.desc, a.desc_w, a.amt_w, desc_exclude=exclude,
                    form_w=a.form_w, agg=a.agg, novel_only=a.novel_only)

    seeds = []
    for path in a.resume:
        if path.exists():
            for e in json.loads(path.read_text()).get("saved", []):
                if "melody" in e and "chord_ids" in e:
                    seeds.append(space.load(e))
    if seeds:
        print(f"resuming from {len(seeds)} pieces", flush=True)
    archive = run(fitness, a.iters, a.batch, a.bins, a.seed, a.init, seeds=seeds, space=space,
                  cvt=cvt, centre_w=a.centre_w)
    s = export(archive, fitness, a.out, a.bins, space, cvt)
    print("star:", s["star"], {k: round(v, 3) for k, v in s["star_breakdown"].items()})


if __name__ == "__main__":
    main()
