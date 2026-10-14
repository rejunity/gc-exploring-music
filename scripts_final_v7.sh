#!/usr/bin/env bash
# Form genome v3: new critic ensemble [r8 r9 r10], peaked bands, geometric mean, 10-D real-pop archive
# with centroid pull, corpus-sampled motifs. Graded by grader3 (fresh, repetition-aware), AMT, humans; audited.
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH
M=models; OUT=out/final_v7_form
DISC="$M/critic_disc_r8.pt $M/critic_disc_r9.pt $M/critic_disc_r10.pt"
uv run python -m evomusic.mapelites --genome form --cvt data/cvt_hooktheory.json --rules 0.5 --lm $M/critic_lm.pt --lm-w 1 \
    --disc $DISC --disc-w 2 --desc data/processed/desc_bands.json --desc-w 2.5 --amt-w 3 --agg geo --centre-w 0.1 \
    --iters 300 --batch 128 --init 1024 --seed 61 --out $OUT
uv run python -m evomusic.evaluate $OUT/summary.json --disc $DISC --eval-disc $M/critic_grader3.pt --amt --human --n-real 400 --json $OUT/onpar.json
echo FINAL_V7_DONE
