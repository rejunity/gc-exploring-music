#!/usr/bin/env bash
# Form-aware + AMT-guided final. AMT is now a fitness term, so the judges are grader 2 and humans.
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH
M=models; OUT=out/final_v4
uv run python -m evomusic.mapelites --rules 0.5 --lm $M/critic_lm.pt --lm-w 1 \
    --disc $M/critic_disc_r7.pt $M/critic_disc_r8.pt $M/critic_disc_r9.pt --disc-w 2 \
    --desc data/processed/desc_bands.json --desc-w 2.5 --amt-w 3 \
    --iters 300 --batch 128 --bins 8 --init 1024 --seed 21 --out $OUT
uv run python -m evomusic.evaluate $OUT/summary.json --disc $M/critic_disc_r7.pt $M/critic_disc_r8.pt $M/critic_disc_r9.pt \
    --eval-disc $M/critic_grader2.pt --amt --human --n-real 400 --json $OUT/onpar.json
echo FINAL_V4_DONE
