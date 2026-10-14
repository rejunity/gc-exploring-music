#!/usr/bin/env bash
# Batch 007: form genome v3 (repetition fix: retrained critic, peaked bands, geometric mean, 10-D archive,
# corpus-sampled motifs) vs real pop. 8 evolved (star + 7 picks by real-pop weight) + 4 references.
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH
RUN=out/final_v7_form
uv run python -m evomusic.pick $RUN --n 7 --seed 7 >/dev/null
# grader3 verdict on the references, for the site
uv run python - <<'PY'
import json
from pathlib import Path
from evomusic.critic.model import Critic
from evomusic.evaluate import load_evolved
ref = Path("out/reference5/summary.json"); r = json.loads(ref.read_text())
gs = load_evolved([ref]); ps = Critic(disc_path=Path("models/critic_grader3.pt")).p_real(gs)
for c, p in zip(r["saved"], ps): c["p_real"] = p
ref.write_text(json.dumps(r, indent=1, default=list))
PY
uv run python -m evomusic.publish 007-repetition-fix --title "Repetition fix: form genome v3 vs real pop (blind test)" \
  --notes "Response to 'random but formulaic / one trick repeated': the critics were found to reward repetition without limit (an audit of real phrases with injected loops), so they were retrained with over-repetition fakes and the GA's own output; likelihood judges now prefer typical real values instead of a flat band; judges combine as a geometric mean (all must agree); the archive tracks rhythm repetition and pulls toward where real pop is densest; new bars are sampled from corpus statistics so new material is as cheap as a repeat. 'Sounds real' = independent grader 3 (never optimised against). 'reference' = mainstream pop excerpts." \
  --credits out/reference5/credits.json \
  --label evolved $(cat $RUN/chosen.txt) \
  --label reference out/reference5/ref_*.mid
