#!/usr/bin/env bash
# Structure-aware final: MAP-Elites vs ensemble [r7 r8 r9] + LM + descriptor bands (incl. phrase-parallel),
# graded by grader2 (fresh, structure fakes), AMT (independent, Lakh), and agreement with human ratings.
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH
M=models; OUT=out/final_v3
uv run python -m evomusic.mapelites --lm $M/critic_lm.pt --lm-w 2 --disc $M/critic_disc_r7.pt $M/critic_disc_r8.pt $M/critic_disc_r9.pt \
    --desc data/processed/desc_bands.json --desc-w 2 --iters 600 --batch 128 --bins 8 --init 1024 --seed 11 --out $OUT
uv run python -m evomusic.evaluate $OUT/summary.json --disc $M/critic_disc_r7.pt $M/critic_disc_r8.pt $M/critic_disc_r9.pt \
    --eval-disc $M/critic_grader2.pt --amt --human --n-real 400 --json $OUT/onpar.json
echo FINAL_V3_DONE
