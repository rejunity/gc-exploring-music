#!/usr/bin/env bash
# Form genome v2: + form-likelihood term, 9-D real-pop archive (exact-repeat share), group-wise anchor moves.
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH
until grep -q FINAL_V5_DONE logs/final_v5.log; do sleep 30; done
M=models; OUT=out/final_v6_form
uv run python -m evomusic.mapelites --genome form --cvt data/cvt_hooktheory.json --rules 0.5 --lm $M/critic_lm.pt --lm-w 1 \
    --disc $M/critic_disc_r7.pt $M/critic_disc_r8.pt $M/critic_disc_r9.pt --disc-w 2 \
    --desc data/processed/desc_bands.json --desc-w 2.5 --form-w 1.5 --amt-w 3 \
    --iters 300 --batch 128 --init 1024 --seed 51 --out $OUT
uv run python -m evomusic.evaluate $OUT/summary.json --disc $M/critic_disc_r7.pt $M/critic_disc_r8.pt $M/critic_disc_r9.pt \
    --eval-disc $M/critic_grader2.pt --amt --human --n-real 400 --json $OUT/onpar.json
echo FINAL_V6_DONE
