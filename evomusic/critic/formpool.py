"""Adversarial negatives from the hook-and-form genome: several short MAP-Elites runs
against the current critic; every archive elite goes into a fake pool with its run id
(run ids keep whole runs held out in train_disc's validation split).

    uv run python -m evomusic.critic.formpool --disc models/critic_disc_r7.pt ... --out data/processed/negatives/me_form_round1.jsonl
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

from evomusic.behaviour import CVT
from evomusic.fitness.factory import build
from evomusic.mapelites import FormSpace, run

ROOT = Path(__file__).resolve().parent.parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--lm", type=Path, default=ROOT / "models/critic_lm.pt")
    ap.add_argument("--disc", type=Path, nargs="+", required=True)
    ap.add_argument("--amt-w", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    cvt = CVT()
    excl = tuple(cvt.dims) + ("parallel_rhythm", "parallel_contour", "distinct_chords")
    fitness = build(0.5, None, 0.5, a.lm, 1.0, a.disc, 2.0, ROOT / "data/processed/desc_bands.json", 2.5, a.amt_w, desc_exclude=excl)
    space = FormSpace()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with a.out.open("w") as f:
        for s in range(a.runs):
            arch = run(fitness, a.iters, 128, 8, seed=a.seed * 100 + s, init=512, log=lambda *_: None, space=space, cvt=cvt, centre_w=0.1)
            fs = [x for x, _ in arch.values()]
            print(f"run {s}: {len(arch)} cells, best {max(fs):.3f}, mean {st.mean(fs):.3f}", flush=True)
            for fit, x in arch.values():
                g = space.pheno(x)
                f.write(json.dumps({"melody": g.melody, "chords": g.chords, "bars": g.bars, "run": s, "fitness": fit}) + "\n")
                n += 1
    print(f"{n} form-genome negatives -> {a.out}")


if __name__ == "__main__":
    main()
