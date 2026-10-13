"""Train the real-vs-fake discriminator, initialised from the critic LM.

    uv run python -m evomusic.critic.train_disc --lm models/critic_lm.pt --out models/critic_disc.pt \
        [--pool data/processed/negatives/*.jsonl]

Real  : Hooktheory 8-bar windows (TRAIN; VALID for evaluation).
Fake  : synthetic corruptions of real windows (GA mutation operators at several
        strengths, pitch noise, bar shuffle, chord shuffle, melody/chord swap
        between songs), random genomes, and GA outputs from adversarial rounds
        (--pool). Each fake family is reported separately on validation.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from evomusic.corpus.stats import load as load_corpus
from evomusic.critic.model import Backbone, load, pad_batch, save
from evomusic.critic.tokens import encode, windows
from evomusic.evolve import mutate
from evomusic.genome import CHORDS, Genome

ROOT = Path(__file__).resolve().parent.parent.parent


def real_windows(split: str, hop: int) -> list[Genome]:
    gs = [g for g in load_corpus(ROOT / "data/processed/hooktheory.jsonl") if g.meta["split"] == split]
    return [w for g in gs for w in windows(g, 8, hop) if w.bars == 8]


def load_pool(paths: list[Path]) -> list[Genome]:
    out = []
    for p in paths:
        for line in p.open():
            r = json.loads(line)
            out.append(Genome([tuple(x) for x in r["melody"]], r["chords"], r["bars"], {"pool": p.stem}))
    return out


# ---- synthetic negatives ------------------------------------------------------

def _pitch_noise(g, rng):
    return Genome.from_bars([[(None if p is None else p + (rng.choice([-2, -1, 1, 2, 3, -3]) if rng.random() < 0.35 else 0), d)
                              for p, d in b] for b in g.bar_genes()], g.chords)


def _bar_shuffle(g, rng):
    bars = g.bar_genes(); idx = list(range(len(bars))); rng.shuffle(idx)
    return Genome.from_bars([bars[i] for i in idx], [g.chords[2 * i + k] for i in idx for k in (0, 1)])


def _chord_shuffle(g, rng):
    ch = list(g.chords)
    for i in range(len(ch)):
        if rng.random() < 0.5:
            ch[i] = rng.randrange(len(CHORDS))
    return Genome.from_bars(g.bar_genes(), ch)


def _swap(g, other, rng):
    return Genome.from_bars(g.bar_genes(), other.chords)


FAMILIES = ["mut1", "mut3", "mut8", "pitch_noise", "bar_shuffle", "chord_shuffle", "swap", "random", "random_scale"]


def make_fake(kind: str, reals: list[Genome], rng: random.Random) -> Genome:
    g = rng.choice(reals)
    if kind == "mut1":
        return mutate(g, rng, 1.0)
    if kind == "mut3":
        return mutate(g, rng, 3.0)
    if kind == "mut8":
        h = g
        for _ in range(3):
            h = mutate(h, rng, 3.0)
        return h
    if kind == "pitch_noise":
        return _pitch_noise(g, rng)
    if kind == "bar_shuffle":
        return _bar_shuffle(g, rng)
    if kind == "chord_shuffle":
        return _chord_shuffle(g, rng)
    if kind == "swap":
        return _swap(g, rng.choice(reals), rng)
    if kind == "random":
        return Genome.random(rng=rng)
    if kind == "random_scale":
        return Genome.random(in_scale=True, rng=rng)
    raise ValueError(kind)


def auc(pos: list[float], neg: list[float]) -> float:
    s = sorted([(x, 1) for x in pos] + [(x, 0) for x in neg])
    rank_sum = sum(i + 1 for i, (_, y) in enumerate(s) if y == 1)
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


@torch.no_grad()
def scores(model, gs, ctx, dev, bs=256) -> list[float]:
    model.eval(); out = []
    for i in range(0, len(gs), bs):
        out += torch.sigmoid(model.forward_disc(pad_batch([encode(g) for g in gs[i:i + bs]], ctx, dev))).tolist()
    model.train()
    return out


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm", type=Path, default=ROOT / "models/critic_lm.pt")
    ap.add_argument("--init", type=Path, help="continue from an existing discriminator instead of the LM")
    ap.add_argument("--pool", type=Path, nargs="*", default=[])
    ap.add_argument("--pool-frac", type=float, default=0.4, help="share of fakes drawn from the GA pool")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", type=Path, default=ROOT / "models/critic_disc.pt")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    rng = random.Random(a.seed); torch.manual_seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    model, _ = load(a.init or a.lm, dev)
    model.train()
    ctx = model.cfg["ctx"]
    train_real, valid_real = real_windows("TRAIN", 4), real_windows("VALID", 8)
    pool = load_pool(a.pool)
    rng.shuffle(pool)
    n_pool_valid = len(pool) // 10
    pool_valid, pool_train = pool[:n_pool_valid], pool[n_pool_valid:]
    print(f"real train {len(train_real)} valid {len(valid_real)}; GA pool train {len(pool_train)} valid {len(pool_valid)}", flush=True)

    # fixed validation fakes per family
    vrng = random.Random(123)
    vfakes = {k: [make_fake(k, valid_real, vrng) for _ in range(400)] for k in FAMILIES}
    if pool_valid:
        vfakes["ga_pool"] = pool_valid

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    t0 = time.time()
    for ep in range(a.epochs):
        reals = list(train_real); rng.shuffle(reals)
        tot = n = 0
        for i in range(0, len(reals), a.bs // 2):
            rb = reals[i:i + a.bs // 2]
            fb = []
            for _ in range(len(rb)):
                if pool_train and rng.random() < a.pool_frac:
                    fb.append(rng.choice(pool_train))
                else:
                    fb.append(make_fake(rng.choice(FAMILIES), train_real, rng))
            x = pad_batch([encode(g) for g in rb + fb], ctx, dev)
            y = torch.tensor([1.0] * len(rb) + [0.0] * len(fb), device=dev)
            y = y * 0.95 + 0.025                                   # label smoothing
            loss = F.binary_cross_entropy_with_logits(model.forward_disc(x), y)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item(); n += 1
        print(f"epoch {ep}  loss {tot / n:.3f}  {time.time() - t0:.0f}s", flush=True)

    pr = scores(model, valid_real, ctx, dev)
    report = {"real_valid_p_real_median": st.median(pr)}
    for k, fs in vfakes.items():
        pf = scores(model, fs, ctx, dev)
        report[f"auc_{k}"] = auc(pr, pf)
    print("validation:", {k: round(v, 3) for k, v in report.items()}, flush=True)
    save(model, a.out, report=report, pool=[str(p) for p in a.pool])
    return report


if __name__ == "__main__":
    main()
