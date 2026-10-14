#!/usr/bin/env bash
# v8 = v7 + novelty-masked likelihood judges (LM, AMT score only non-repeated bars). Then the audit with AMT.
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH
M=models; OUT=out/final_v8_form
DISC="$M/critic_disc_r8.pt $M/critic_disc_r9.pt $M/critic_disc_r10.pt"
uv run python -m evomusic.mapelites --genome form --cvt data/cvt_hooktheory.json --rules 0.5 --lm $M/critic_lm.pt --lm-w 1 \
    --disc $DISC --disc-w 2 --desc data/processed/desc_bands.json --desc-w 2.5 --amt-w 3 --agg geo --centre-w 0.1 --novel-only \
    --iters 300 --batch 128 --init 1024 --seed 61 --out $OUT
uv run python -m evomusic.evaluate $OUT/summary.json --disc $DISC --eval-disc $M/critic_grader3.pt --amt --human --n-real 400 --json $OUT/onpar.json
echo "== audit: ensemble + AMT, novelty-masked"
uv run python -m evomusic.critic.audit --n 80 --lm $M/critic_lm.pt --disc $DISC --desc data/processed/desc_bands.json \
    --ngram data/processed/ngram_hooktheory.json --amt-w 3 --novel-only --json out/audit/critic_v2_novel_amt.json
echo FINAL_V8_DONE
