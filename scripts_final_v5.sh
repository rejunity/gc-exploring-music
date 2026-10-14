#!/usr/bin/env bash
# Step 2 (real-pop-shaped CVT archive) and step 3 (hook-and-form genome), same full fitness, then grading.
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH
M=models
FIT="--rules 0.5 --lm $M/critic_lm.pt --lm-w 1 --disc $M/critic_disc_r7.pt $M/critic_disc_r8.pt $M/critic_disc_r9.pt --disc-w 2 \
     --desc data/processed/desc_bands.json --desc-w 2.5 --amt-w 3 --cvt data/cvt_hooktheory.json --iters 300 --batch 128 --init 1024"
for G in form notes; do
  OUT=out/final_v5_$G
  uv run python -m evomusic.mapelites --genome $G $FIT --seed 41 --out $OUT
  uv run python -m evomusic.evaluate $OUT/summary.json --disc $M/critic_disc_r7.pt $M/critic_disc_r8.pt $M/critic_disc_r9.pt \
      --eval-disc $M/critic_grader2.pt --amt --human --n-real 400 --json $OUT/onpar.json
  echo "DONE_$G"
done
echo FINAL_V5_DONE
