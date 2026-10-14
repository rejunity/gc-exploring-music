# External critics (independent of our training data)

Tried 2026-10-14 on 24 clips from site batches 001-003 (random, rules-evolved,
corpus-evolved, critic-evolved, real Hooktheory references). Each lives in its
own venv because their dependencies clash with ours.

```bash
uv venv -p 3.11 amt && VIRTUAL_ENV=amt uv pip install "anticipation @ git+https://github.com/jthickstun/anticipation.git" \
    transformers mido numpy tqdm "torch>=2.6" --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu126
uv venv -p 3.11 aes && VIRTUAL_ENV=aes uv pip install audiobox_aesthetics requests soundfile "torch>=2.6" \
    --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cu126
# CLIPS dir: <name>.mid (+ <name>.wav rendered with fluidsynth) and index.json [{name, label}]
amt/bin/python amt_score.py CLIPS     # Anticipatory Music Transformer (Lakh MIDI) NLL per token
aes/bin/python aes_score.py CLIPS     # Meta Audiobox Aesthetics on rendered audio
```

| group | AMT NLL (lower = more like Lakh music) | Audiobox content enjoyment |
|---|---|---|
| random | 1.22 | 7.11 |
| rules-only GA | 0.85 | 6.82 |
| rules + n-gram GA | 0.67 | 6.53 |
| critic-guided archive (batch 003) | 0.71 | 6.50 |
| real references | 0.67 | 6.64 |

- **AMT is useful**: ranks random < rules < corpus/critic ≈ real, fully
  independent of Hooktheory. It disagrees with our grader on the two sparsest
  archive cells (cell_0_0, cell_0_7: AMT NLL ~1.07, our grader 0.81-0.82), a
  blind spot worth fixing. Use as a held-out judge, not a fitness term.
- **Audiobox Aesthetics is not a composition critic** for this setup: random
  notes score highest on enjoyment; all groups are within 0.6 points because
  every clip uses the same piano rendering. It judges audio production.
