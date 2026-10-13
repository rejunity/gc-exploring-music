"""Adversarial rounds: evolve against the current critic, add the GA's output to
the discriminator's fake pool, retrain, repeat. The critic gets harder to game
each round; the GA has to move closer to real music to keep scoring.

    uv run python -m evomusic.critic.adversarial --rounds 4 --runs 4

Per round it logs, on held-out data:
  - p_real of held-out real windows (critic should still accept real music),
  - p_real of this round's GA winners under the critic they were evolved against,
  - p_real of the same winners under the *retrained* critic (exploit closed?),
  - AUC real vs each fake family.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

from evomusic.critic import train_disc
from evomusic.critic.model import Critic
from evomusic.evolve import evolve
from evomusic.fitness.factory import build

ROOT = Path(__file__).resolve().parent.parent.parent
POOL = ROOT / "data/processed/negatives"
MODELS = ROOT / "models"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--start", type=int, default=0, help="resume at this round (expects models/critic_disc_r{start}.pt)")
    ap.add_argument("--runs", type=int, default=4)
    ap.add_argument("--pop", type=int, default=256)
    ap.add_argument("--gens", type=int, default=150)
    ap.add_argument("--keep", type=int, default=128, help="distinct individuals per run added to the pool")
    ap.add_argument("--lm", type=Path, default=MODELS / "critic_lm.pt")
    a = ap.parse_args()
    POOL.mkdir(parents=True, exist_ok=True)
    log = (ROOT / "logs/adversarial.jsonl").open("a")

    disc = MODELS / f"critic_disc_r{a.start}.pt"
    if a.start == 0 and not disc.exists():
        print("== round 0: discriminator on synthetic fakes only", flush=True)
        rep = train_disc.main(["--lm", str(a.lm), "--out", str(disc), "--epochs", "3"])
        log.write(json.dumps({"round": 0, "train_report": rep}) + "\n"); log.flush()

    for r in range(a.start, a.start + a.rounds):
        print(f"== round {r + 1}: evolve against critic_disc_r{r}", flush=True)
        fitness = build(rules=1.0, lm=a.lm, lm_w=1.0, disc=disc, disc_w=4.0)
        winners, pool = [], []
        for s in range(a.runs):
            pop, fit = evolve(fitness, a.pop, a.gens, seed=1000 * r + s, verbose=False)
            seen, kept = set(), []
            for i in sorted(range(len(pop)), key=lambda i: -fit[i]):
                k = (tuple(pop[i].melody), tuple(pop[i].chords))
                if k not in seen:
                    seen.add(k); kept.append(pop[i])
                if len(kept) >= a.keep:
                    break
            winners.append(kept[0]); pool += kept
            print(f"   run {s}: best fitness {max(fit):.3f}, kept {len(kept)}", flush=True)
        path = POOL / f"round{r + 1}.jsonl"
        with path.open("w") as f:
            for g in pool:
                f.write(json.dumps({"melody": g.melody, "chords": g.chords, "bars": g.bars}) + "\n")

        old = Critic(disc_path=disc)
        p_old = old.p_real(winners)
        new_disc = MODELS / f"critic_disc_r{r + 1}.pt"
        rep = train_disc.main(["--lm", str(a.lm), "--init", str(disc), "--out", str(new_disc), "--epochs", "2",
                               "--pool", *map(str, sorted(POOL.glob("round*.jsonl")))])
        p_new = Critic(disc_path=new_disc).p_real(winners)
        row = {"round": r + 1, "winners_p_real_old": p_old, "winners_p_real_new": p_new, "train_report": rep}
        print(f"   winners p_real: under r{r} critic {st.mean(p_old):.3f} -> retrained r{r + 1} {st.mean(p_new):.3f}; "
              f"real valid median {rep['real_valid_p_real_median']:.3f}", flush=True)
        log.write(json.dumps(row) + "\n"); log.flush()
        disc = new_disc
    print("final critic:", disc)


if __name__ == "__main__":
    main()
