"""Behaviour space shaped by real pop (CVT-MAP-Elites).

Each piece gets a 9-D behaviour vector:

  notes_per_bar    density
  symbol_entropy   melodic variety
  period_echo      how much bars 5-6 echo bars 1-2 (question / answer)
  climax_pos       where the highest note sits (0 = first bar, 1 = last)
  distinct_motifs  how many different bar-motifs the form parser finds
  exact_share      share of bars that literally repeat an earlier bar
  four_chord_share share of half-bars on I / IV / V / vi (and their 7th / sus forms)
  minor_share      share of half-bars on minor chords (mood)
  chord_changes    harmonic rhythm

`fit` places k centroids with k-means on real pop phrases, so the archive's
cells sit where real music lives, denser where real music is denser. Each
centroid remembers its share of real phrases ("weight"): publishing clips by
sampling cells in proportion to that weight gives an evolved set with the same
mix of forms, densities and harmonic colours as real pop, instead of 64 copies
of the single best recipe.

    uv run python -m evomusic.behaviour fit            # -> data/cvt_hooktheory.json
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

from evomusic.genome import CHORDS, Genome

ROOT = Path(__file__).resolve().parent.parent
CVT_PATH = ROOT / "data" / "cvt_hooktheory.json"
DIMS = ["notes_per_bar", "symbol_entropy", "period_echo", "climax_pos", "distinct_motifs", "exact_share",
        "four_chord_share", "minor_share", "chord_changes"]
FOUR = {"I", "IV", "V", "vi", "Imaj7", "IVmaj7", "V7", "vi7", "Isus4", "Vsus4"}
MINOR = {"ii", "iii", "vi", "ii7", "vi7", "iv"}


def behaviour(g: Genome) -> dict[str, float]:
    from evomusic.complexity import symbol_entropy
    from evomusic.fitness.descriptors import form
    from evomusic.form import from_genome
    f = form(g)
    fg, toks = from_genome(g)
    names = [CHORDS[c].name for c in g.chords]
    return {
        "notes_per_bar": len(g.pitches()) / g.bars,
        "symbol_entropy": symbol_entropy(g),
        "period_echo": f["period_echo"],
        "climax_pos": f["climax_pos"],
        "distinct_motifs": float(len(fg.motifs)),
        "exact_share": sum(t.endswith("=") for t in toks) / len(toks),
        "four_chord_share": sum(n in FOUR for n in names) / len(names),
        "minor_share": sum(n in MINOR for n in names) / len(names),
        "chord_changes": sum(a != b for a, b in zip(g.chords, g.chords[1:])) / (len(g.chords) - 1),
    }


class CVT:
    def __init__(self, path: Path = CVT_PATH):
        d = json.loads(Path(path).read_text())
        self.dims = d["dims"]
        self.mean, self.std = np.array(d["mean"]), np.array(d["std"])
        self.centroids, self.weights = np.array(d["centroids"]), d["weights"]

    def vec(self, g: Genome) -> np.ndarray:
        b = behaviour(g)
        return (np.array([b[k] for k in self.dims]) - self.mean) / self.std

    def cell(self, g: Genome) -> int:
        return int(np.argmin(((self.centroids - self.vec(g)) ** 2).sum(1)))

    def __len__(self) -> int:
        return len(self.centroids)


def kmeans(X: np.ndarray, k: int, seed: int = 0, iters: int = 50) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    C = [X[rng.integers(len(X))]]
    d2 = ((X - C[0]) ** 2).sum(1)
    for _ in range(1, k):                                   # k-means++ initialisation
        C.append(X[rng.choice(len(X), p=d2 / d2.sum())])
        d2 = np.minimum(d2, ((X - C[-1]) ** 2).sum(1))
    C = np.array(C)
    for _ in range(iters):
        lab = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(-1), 1)
        for j in range(k):
            if (lab == j).any():
                C[j] = X[lab == j].mean(0)
    lab = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(-1), 1)
    return C, np.bincount(lab, minlength=k)


def fit(k: int = 128, out: Path = CVT_PATH) -> dict:
    from evomusic.critic.train_disc import real_windows
    real = real_windows("TRAIN", 4)
    B = np.array([[behaviour(g)[d] for d in DIMS] for g in real])
    mean, std = B.mean(0), B.std(0) + 1e-9
    C, counts = kmeans((B - mean) / std, k)
    d = {"dims": DIMS, "mean": mean.tolist(), "std": std.tolist(), "centroids": C.tolist(),
         "weights": (counts / counts.sum()).tolist(), "n_real": len(real)}
    out.write_text(json.dumps(d))
    return d


def pick(saved: list[dict], n: int, seed: int = 0) -> list[dict]:
    """Choose n archive entries: by real-pop weight of their cell when known (CVT archives),
    otherwise spread evenly over the grid."""
    rng = random.Random(seed)
    pool = list(saved)
    if pool and "cell_weight" in pool[0]:
        out = []
        while pool and len(out) < n:
            w = [max(1e-6, e["cell_weight"]) for e in pool]
            e = rng.choices(pool, weights=w)[0]
            out.append(e); pool.remove(e)
        return out
    pool.sort(key=lambda e: tuple(e["cell"]) if isinstance(e["cell"], list) else (e["cell"],))
    return [pool[int(i * len(pool) / n)] for i in range(min(n, len(pool)))]


def main(argv: list[str]) -> None:
    if argv[:1] == ["fit"]:
        k = int(argv[1]) if len(argv) > 1 else 128
        d = fit(k)
        w = np.array(d["weights"])
        print(f"{k} centroids from {d['n_real']} real 8-bar windows; cell weight min {w.min():.4f} median {np.median(w):.4f} max {w.max():.4f}")
        print("dims:", ", ".join(f"{n} mean {m:.2f} sd {s:.2f}" for n, m, s in zip(DIMS, d["mean"], d["std"])))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
