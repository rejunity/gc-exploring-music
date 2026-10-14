#!/usr/bin/env bash
# Critic v2: close the repetition shortcut.
#  1. form-genome adversarial pool against the current ensemble (what the GA actually exploits)
#  2. critic_disc_r10 continued from r9 with over-repetition fakes + all pools
#  3. grader3 fresh from the LM (independent judge), other seed
#  4. judge audit of the new ensemble [r8 r9 r10] with AMT, and of grader3 alone
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH
M=models; NEG=data/processed/negatives
uv run python -m evomusic.critic.formpool --disc $M/critic_disc_r7.pt $M/critic_disc_r8.pt $M/critic_disc_r9.pt --runs 8 --iters 100 --seed 7 --out $NEG/me_form_round1.jsonl
uv run python -m evomusic.critic.train_disc --lm $M/critic_lm.pt --init $M/critic_disc_r9.pt --out $M/critic_disc_r10.pt --epochs 3 --real-per-epoch 16000 --seed 10 --pool $NEG/*.jsonl
uv run python -m evomusic.critic.train_disc --lm $M/critic_lm.pt --out $M/critic_grader3.pt --epochs 4 --real-per-epoch 16000 --seed 4244 --pool $NEG/*.jsonl
echo "== audit: GA-facing ensemble r8 r9 r10 + AMT"
uv run python -m evomusic.critic.audit --n 80 --lm $M/critic_lm.pt --disc $M/critic_disc_r8.pt $M/critic_disc_r9.pt $M/critic_disc_r10.pt \
    --desc data/processed/desc_bands.json --ngram data/processed/ngram_hooktheory.json --amt-w 3 --json out/audit/critic_v2_ensemble.json
echo "== audit: grader3 alone"
uv run python -m evomusic.critic.audit --n 80 --rules 0 --disc $M/critic_grader3.pt --disc-w 1 --json out/audit/critic_v2_grader3.json
echo CRITIC_V2_DONE
