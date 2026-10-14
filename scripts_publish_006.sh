#!/usr/bin/env bash
# Batch 006: hook-and-form genome vs flat notes genome (both on the real-pop-shaped archive) vs real pop.
# 4 clips each: the starred best + 3 cells sampled in proportion to how common their region is in real pop.
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH
for G in form notes; do uv run python -m evomusic.pick out/final_v5_$G --n 3 --seed 6 >/dev/null; done
uv run python -m evomusic.publish 006-form-genome --title "Hook-and-form genome vs flat notes vs real pop (blind test)" \
  --notes "Step 2: archive cells placed where real pop lives (8 dimensions incl. phrase echo, climax position, number of motifs, harmony), clips sampled by how common their region is in real pop. Step 3: 'form' pieces are built from motifs + a phrase plan + transformations (repeat, sequence, new ending, same rhythm, fragment), plans mined from 17k real pop phrases; 'notes' is the old flat genome on the same archive. Same fitness for both (rules, corpus bands, small critic, AMT). 'reference' = mainstream pop excerpts. Each clip shows its phrase plan." \
  --credits out/reference5/credits.json \
  --label form $(cat out/final_v5_form/chosen.txt) \
  --label notes $(cat out/final_v5_notes/chosen.txt) \
  --label reference out/reference5/ref_*.mid
