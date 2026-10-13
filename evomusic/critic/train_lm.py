"""Train the critic's language model on Hooktheory 8-bar windows.

    uv run python -m evomusic.critic.train_lm --epochs 20 --out models/critic_lm.pt
"""
from __future__ import annotations

import argparse
import math
import random
import statistics as st
import time
from pathlib import Path

import torch

from evomusic.corpus.stats import load as load_corpus
from evomusic.critic.model import Backbone, lm_nll, pad_batch, save
from evomusic.critic.tokens import encode, windows

ROOT = Path(__file__).resolve().parent.parent.parent


def corpus_windows(split: str, hop: int = 4) -> list[list[int]]:
    gs = [g for g in load_corpus(ROOT / "data/processed/hooktheory.jsonl") if g.meta["split"] == split]
    return [encode(w) for g in gs for w in windows(g, 8, hop)]


@torch.no_grad()
def evaluate(model, seqs, ctx, device, bs=128) -> list[float]:
    model.eval()
    out = []
    for i in range(0, len(seqs), bs):
        out += lm_nll(model, pad_batch(seqs[i:i + bs], ctx, device)).tolist()
    model.train()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--ctx", type=int, default=384)
    ap.add_argument("--out", type=Path, default=ROOT / "models/critic_lm.pt")
    a = ap.parse_args()
    torch.manual_seed(0); random.seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    train, valid, test = corpus_windows("TRAIN", 2), corpus_windows("VALID", 4), corpus_windows("TEST", 4)
    lens = [len(s) for s in train]
    print(f"windows train {len(train)} valid {len(valid)} test {len(test)}; tokens/window p50 {st.median(lens)} max {max(lens)}")

    model = Backbone(d=a.d, layers=a.layers, heads=8, ctx=a.ctx).to(dev)
    print(f"params {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M on {dev}")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05, betas=(0.9, 0.98))
    steps = a.epochs * math.ceil(len(train) / a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.05)
    best, t0 = 1e9, time.time()
    for ep in range(a.epochs):
        random.shuffle(train)
        # bucket by length for speed
        order = sorted(range(len(train)), key=lambda i: len(train[i]) + random.random() * 40)
        batches = [order[i:i + a.bs] for i in range(0, len(order), a.bs)]
        random.shuffle(batches)
        tot = n = 0
        for b in batches:
            x = pad_batch([train[i] for i in b], a.ctx, dev)
            loss = lm_nll(model, x).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tot += loss.item(); n += 1
        v = st.mean(evaluate(model, valid, a.ctx, dev))
        print(f"epoch {ep:2d}  train {tot / n:.3f}  valid {v:.3f}  {time.time() - t0:.0f}s", flush=True)
        if v < best:
            best = v
            save(model, a.out, epoch=ep, valid_nll=v)
    # band from held-out real windows, using the best checkpoint
    from evomusic.critic.model import load
    model, _ = load(a.out, dev)
    vn = evaluate(model, valid, a.ctx, dev)
    q = st.quantiles(vn, n=10)
    tn = evaluate(model, test, a.ctx, dev)
    save(model, a.out, valid_nll=st.mean(vn), test_nll=st.mean(tn), band=(q[0], q[8]))
    print(f"best valid {st.mean(vn):.3f}  test {st.mean(tn):.3f}  band p10..p90 = {q[0]:.3f}..{q[8]:.3f}")


if __name__ == "__main__":
    main()
