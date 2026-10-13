"""Adversarial rounds v2: diverse MAP-Elites fakes, run-level held-out validation,
ensemble critic.

    uv run python -m evomusic.critic.adversarial --rounds 6 --runs 8

Each round r:
  1. fitness = rules + LM band + ensemble(last `--ensemble` discriminators)
  2. `--runs` short MAP-Elites runs against it; every archive elite (diverse by
     construction) goes to data/processed/negatives/me_round{r}.jsonl with its run id
  3. a new discriminator continues from the previous one, trained on real vs
     synthetic fakes + all pools; validated on *held-out runs* only
  4. honest exploit check: elites of held-out runs scored by the old ensemble
     (what the GA achieved) vs the new discriminator (did it learn to spot them?)
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

from evomusic.critic import train_disc
from evomusic.critic.model import Critic
from evomusic.fitness.factory import build
from evomusic.mapelites import run as me_run

ROOT = Path(__file__).resolve().parent.parent.parent
POOL = ROOT / "data/processed/negatives"
MODELS = ROOT / "models"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--start", type=int, default=2, help="latest existing discriminator round (models/critic_disc_r{start}.pt)")
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--bins", type=int, default=6)
    ap.add_argument("--ensemble", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--real-per-epoch", type=int, default=12000)
    ap.add_argument("--lm", type=Path, default=MODELS / "critic_lm.pt")
    a = ap.parse_args()
    POOL.mkdir(parents=True, exist_ok=True)
    log = (ROOT / "logs/adversarial_v2.jsonl").open("a")

    for r in range(a.start, a.start + a.rounds):
        members = [MODELS / f"critic_disc_r{k}.pt" for k in range(max(0, r - a.ensemble + 1), r + 1)]
        members = [m for m in members if m.exists()]
        print(f"== round {r + 1}: MAP-Elites against ensemble {[m.stem for m in members]}", flush=True)
        fitness = build(rules=1.0, lm=a.lm, lm_w=1.0, disc=members, disc_w=4.0)
        elites = []
        path = POOL / f"me_round{r + 1}.jsonl"
        with path.open("w") as f:
            for s in range(a.runs):
                arch = me_run(fitness, a.iters, a.batch, a.bins, seed=1000 * (r + 1) + s, init=512, log=lambda *_: None)
                fs = [x for x, _ in arch.values()]
                print(f"   run {s}: {len(arch)} cells, best {max(fs):.3f}, mean {st.mean(fs):.3f}", flush=True)
                for fit, g in arch.values():
                    f.write(json.dumps({"melody": g.melody, "chords": g.chords, "bars": g.bars, "run": s, "fitness": fit}) + "\n")
                    elites.append((s, g))
        # held-out runs in this file are those with index % 4 == 3 (same rule as train_disc.split_by_run)
        held = [g for s, g in elites if s % 4 == 3]
        old = Critic(disc_path=members)
        p_old = old.p_real(held)
        new = MODELS / f"critic_disc_r{r + 1}.pt"
        pools = sorted(POOL.glob("*.jsonl"))
        rep = train_disc.main(["--lm", str(a.lm), "--init", str(members[-1]), "--out", str(new),
                               "--epochs", str(a.epochs), "--real-per-epoch", str(a.real_per_epoch), "--seed", str(r + 1),
                               "--pool", *map(str, pools)])
        p_new = Critic(disc_path=[new]).p_real(held)
        row = {"round": r + 1, "members": [m.stem for m in members], "heldout_elites": len(held),
               "heldout_p_real_old_ensemble": st.mean(p_old), "heldout_p_real_new_disc": st.mean(p_new), "train_report": rep}
        print(f"   held-out elites p_real: old ensemble {st.mean(p_old):.3f} -> new disc {st.mean(p_new):.3f}; "
              f"real valid median {rep['real_valid_p_real_median']:.3f}; AUC held-out latest {rep.get('auc_ga_heldout_latest', float('nan')):.3f}",
              flush=True)
        log.write(json.dumps(row) + "\n"); log.flush()


if __name__ == "__main__":
    main()
