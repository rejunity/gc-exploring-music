#!/usr/bin/env bash
# Final stage after adversarial rounds: independent grading critic, final MAP-Elites archive
# against the last ensemble, on-par report, publish batch.  Usage: bash scripts_final.sh <last_round>
set -euo pipefail
R=${1:?last round}
export PATH=$HOME/.local/bin:$PATH
M=models
ENS="$M/critic_disc_r$((R-2)).pt $M/critic_disc_r$((R-1)).pt $M/critic_disc_r$R.pt"

# 1. independent grader: fresh from the LM (not from the GA-facing chain), other seed, all pools so far
uv run python -m evomusic.critic.train_disc --lm $M/critic_lm.pt --out $M/critic_grader.pt --epochs 3 \
    --real-per-epoch 16000 --seed 4242 --pool data/processed/negatives/*.jsonl

# 2. final archive: longer MAP-Elites against the last ensemble
uv run python -m evomusic.mapelites --lm $M/critic_lm.pt --disc $ENS --iters 600 --batch 128 --bins 8 --init 1024 \
    --seed 7 --out out/final_r$R

# 3. on-par report (train ensemble vs independent grader)
uv run python -m evomusic.evaluate out/final_r$R/summary.json --disc $ENS --eval-disc $M/critic_grader.pt \
    --json out/final_r$R/onpar.json
