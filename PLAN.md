# PLAN — Evolutionary MIDI melody generation

Status: draft v2, 2026-10-13 (v1 questions answered by Renaldas, see section 11). Source: Renaldas' "Evolutionary Music Generation
approach" doc (Palaiseau discussion) plus a survey of datasets and tools done today.

## 1. Goal

Produce `.mid` files by **evolving** melodies rather than hand-writing rules or
training a next-note model. A genetic algorithm (GA) maintains a population of
melodies and pushes it toward high fitness, where fitness blends:

1. hand-written music-theory rules,
2. statistical similarity to a corpus,
3. a learned critic network,
4. occasional human feedback,

with explicit **knobs** for novelty (mutation strength) and target complexity.

Why this is interesting versus the two baselines from the doc:

| approach | explores broadly? | sounds "human"? | can absorb feedback? |
|---|---|---|---|
| procedural rules | only as far as the rules allow | mechanical unless probabilistic | awkward |
| corpus-trained model | stays inside the corpus | yes | via fine-tuning only |
| **evolution + composite fitness** | yes, search is open-ended | as good as the critic | naturally, feedback is just another fitness term |

The honest risk: a GA is only as good as its fitness function. Most of the
work in this project is *designing and validating fitness*, not the GA loop.

## 2. Scope decisions (proposed, please push back)

- **Melody + chords from day one.** One melody line over a chord track
  (block chords or a simple comping pattern), fixed key, 4/4, 8–16 bars.
  Polyphonic counterpoint and expressive timing come later (section 9).
- **Discrete genome.** A melody is a list of `(pitch | rest, duration)` genes on
  a 16th-note grid plus a chord chromosome, one chord per half-bar. Simple,
  every mutation stays valid, trivially serialises to MIDI.
- **Style order: pop → folk → baroque → jazz.** Pop first means the primary
  corpora are POP909 and the Hooktheory Lead Sheet Dataset (both melody +
  chords). Folk (Nottingham, Essen) is the clean fallback for calibrating rules.
- **Quality-diversity archive, with a starred winner.** Each run produces a
  MAP-Elites archive of diverse melodies; the single best-fitness individual is
  exported with a `★` / `star` prefix in its filename.
- **Code first, notebooks for exploration only.** Everything reusable lives in
  the `evomusic` package; notebooks import from it.

## 3. Datasets

### 3.1 Pop lead-sheet corpora (primary: melody + chords)

| dataset | size | format | license / access | why |
|---|---|---|---|---|
| **POP909** | 909 pop songs (Chinese pop, piano arrangements) | MIDI with melody / bridge / piano tracks, chord + phrase labels, key | research use, github.com/music-x-lab/POP909-Dataset | melody + chord labels + phrase structure; first corpus |
| **Hooktheory via Sheet Sage** (Donahue et al. 2022) | 26,175 TheoryTab segments, 23,833 with melody, ~50 h | gzipped JSON: per segment key (tonic + scale intervals), meter, melody `(onset, offset, octave, pitch_class)` in beats, harmony `(root, intervals, inversion)`; plus MIDI tarballs, 8:1:1 split | CC BY-NC-SA 3.0, github.com/chrisdonahue/sheetsage-data | **primary Western pop corpus**: melody + functional harmony, already key-normalisable |
| Hooktheory Lead Sheet Dataset (HLSD, wayne391) | 11,329 segments | JSON + MIDI | academic use only; full dump link dead (2026-10-13) | superseded by the Sheet Sage release above; repo kept for its crawler |
| **OpenEWLD / WikiMusicText** | ~1,000–2,000 lead sheets | MusicXML / ABC | public domain subset of Wikifonia (closed 2013) | pop/standards lead sheets |
| **Lakh MIDI "clean_midi"** | ~17k files named by artist/title | multi-track MIDI | CC-BY 4.0 | real pop songs, but melody track must be extracted heuristically |

### 3.2 Folk / monophonic corpora (second style; also used to calibrate rules)

| dataset | size | format | license / access | why |
|---|---|---|---|---|
| **Essen Folksong Collection** | ~8,500 tunes (5,365 German, 2,246 Chinese) | `**kern` (KernScores), ABC, also bundled inside `music21` corpus | legal status "unclear" per music21 notes, universally used in research | the standard monophonic corpus; phrase boundaries annotated |
| **Nottingham Music Database** | ~1,200 British/American folk tunes | ABC (with chords), MIDI | free, maintained by Seymour Shlien at ifdo.ca | melody + chord symbols, ideal first corpus |
| **Meertens Tune Collections (MTC)** via `MTCFeatures` | 8k+ Dutch/German melodies as feature sequences | Python package, Zenodo | open | ready-made features (intervals, metric weights, phrase ends) |
| **Weimar Jazz Database** | 456 transcribed jazz solos | SQLite + MIDI export | free download, jazzomat.hfm-weimar.de | later: a "jazz" style target for the critic |
| **MeloSol corpus** | ~780 solfege-style melodies | kern | open | small, very clean |

### 3.3 Baroque and jazz (third and fourth style), plus large corpora for a stronger critic

| dataset | size | notes |
|---|---|---|
| Bach chorales (music21 corpus, `bach/bwv*`) | 371 chorales | baroque four-part harmony, soprano = melody |
| KernScores classical (Haydn, Mozart, Beethoven quartets, Bach inventions) | thousands of movements | kern, open |
| Weimar Jazz Database | 456 transcribed solos with chords | jazz target, free download |

Large polyphonic MIDI corpora (critic pre-training, later):

| dataset | size | notes |
|---|---|---|
| Lakh MIDI (LMD), "clean_midi" subset | 176k files total, ~17k named by artist/title | CC-BY 4.0; noisy, sequenced not performed |
| MAESTRO v3 | 1,282 performances, ~200 h | classical piano, expressive timing |
| GiantMIDI-Piano | 10,854 pieces | transcribed from audio |
| Aria-MIDI (ICLR 2025) | 1.19M files, ~100k h piano | huggingface `loubb/aria-midi`; "Unique" subset is 32k files |
| GigaMIDI (TISMIR 2025) | 1.4M files, all GM instruments | HF access gated to research use |
| PDMX | 250k public-domain MusicXML scores | Zenodo |

### 3.4 Data plan

1. `uv run python -m evomusic.data download` fetches every core corpus into
   `data/raw/` with sha256 verification (done 2026-10-13; see README).
2. Write `evomusic/corpus.py`: parse ABC / kern / MIDI into the canonical
   genome representation (section 5): melody + chord-per-half-bar. Normalise:
   transpose to C major / A minor, quantise to 16ths, reduce chords to a
   vocabulary of ~24 (diatonic triads + 7ths + a few borrowed), drop tunes with
   unusual metres for v1.
3. Produce `data/processed/*.parquet` with one row per tune plus per-tune
   descriptors (section 6.5). Commit a handful of tiny samples under
   `data/samples/` so tests run without downloads.
4. Exploration notebook: interval histograms, duration histograms, pitch range,
   phrase lengths, n-gram entropy. These numbers directly calibrate the rules.

## 4. Tools

Environment already set up (commit 1): `uv` + `pyproject.toml`, Python 3.11.

| tool | role | status |
|---|---|---|
| `uv` | env + deps | installed (`~/.local/bin/uv`) |
| `mido` | write MIDI, inspect messages | installed |
| `symusic` | fast C++ MIDI/ABC parsing (100× mido), used by MidiTok 3 | installed |
| `music21` | kern/ABC parsing, key finding, bundled Essen corpus, score rendering | optional extra `analysis` |
| `numpy`, `pandas`, `matplotlib`, `jupyterlab` | analysis | optional extra `analysis` |
| `torch`, `miditok` | critic network, tokenisation | optional extra `critic` (GTX 1080, 8 GB: fine for a small transformer) |
| `pyribs` | MAP-Elites / quality-diversity archive | add when we reach section 7.3 |
| `nolds` | Lyapunov / entropy estimators on time series | add when we reach section 6.5 |
| `fluidsynth` + `fluid-soundfont-gm` + `ffmpeg` | render `.mid` → `.wav` → `.mp3` (`evomusic/render.py`); 8 bars ≈ 240 KB mp3 at 96 kbps | installed 2026-10-13 |
| `lilypond` or MuseScore | render notation PNGs for the notebook | optional, `apt install lilypond` |

Deliberately *not* used: DEAP (last release 2020), musicaiz (AGPL, stale).
The GA loop is ~100 lines of numpy; owning it keeps mutation operators
musical rather than generic.

## 5. Genome and phenotype

```
Genome    = (melody: list[Gene], chords: list[Chord])
Gene      = (pitch: int | None, dur: int)      # dur in 16ths
Chord     = index into a fixed vocabulary       # one per half-bar
Phenotype = MIDI file, track 1 melody, track 2 chords (block or comping pattern)
```

Invariants: durations sum to `bars * 16`; pitches within a configurable range;
`len(chords) == bars * 2`.

Chord mutations: substitute (diatonic neighbour, relative minor/major,
secondary dominant), copy a bar's chord, re-harmonise a bar to fit its melody
notes. Melody and chords can also be evolved in alternation (fix one, evolve
the other) which keeps the search space manageable.
Mutation operators (each with its own probability knob):

- `pitch_shift` — move one note by ±1..±4 semitones (or ±1 scale degree);
- `transpose_phrase` — shift a whole bar/phrase;
- `invert`, `retrograde` of a phrase (classical motivic operations);
- `split` / `merge` — change rhythm while keeping total duration;
- `swap_rest`;
- `repeat_motif` — copy a bar elsewhere (cheap way to get structure).

Crossover: bar-aligned one-point and two-point (phrase 1 + phrase 4 as in the doc).

## 6. Fitness

Fitness is a weighted sum (v1) or a Pareto front (v2) over the terms below.
Every term is a function `genome -> float` in `evomusic/fitness/`, unit-tested
on a few known-good tunes and a few known-bad ones.

### 6.1 Music-theory rules (`rules.py`), available immediately
- scale consistency: fraction of notes in the key;
- interval profile: penalise leaps > octave, reward step motion, reward leap
  followed by step in the opposite direction (gap-fill);
- range: penalise tessitura > 1.5 octaves;
- rhythmic sanity: notes on strong beats, no pathological syncopation;
- cadence: last note on tonic/dominant, phrase ends on long notes;
- **melody–chord fit**: notes on strong beats are chord tones, non-chord tones
  resolve by step; chord progression follows functional harmony (reward V→I,
  ii→V, penalise retrogressions unless the corpus says pop does them);
- repetition / self-similarity: at least one repeated bar, but not all bars equal.

### 6.2 Corpus statistics (`ngram.py`), after section 3.3
- n-gram models (n = 2..5) over (interval, duration) pairs and over chord
  transitions, fitted on the corpus (HLSD Roman numerals are perfect for this);
- fitness = mean log-likelihood of the genome under the model, blended with a
  short-term model fitted on the genome itself. This is a cheap re-implementation
  of IDyOM's long-term + short-term "information content". IDyOM's surprise
  correlates with human expectancy ratings, so this is already a decent critic.

### 6.3 Learned critic (`critic.py`)
Two options, in order of effort:
1. **Likelihood critic**: a small transformer language model (MidiTok REMI
   tokens) trained on the corpus; fitness = NLL. Same spirit as 6.2 but learns
   longer context. ~1–2 hours on the GTX 1080 for folk-scale data.
2. **Discriminator critic**: classifier real-vs-evolved, retrained every N
   generations (GA as generator, critic as discriminator). More "interesting",
   but adversarial training is notoriously unstable. Try only after (1).

### 6.4 Human feedback (`human.py`)
- Pairwise comparisons: play A and B, pick one. Fit a Bradley-Terry score
  per individual; then train a tiny reward model on genome descriptors so
  feedback generalises to unrated individuals.
- Budget: ~50 comparisons per session should be enough to steer.
- **Interface: a static web page on GitHub Pages** (`docs/` folder of this
  repo, zero hosting cost). Each run uploads a batch of mp3s plus a
  `pairs.json`; the page shows two players and "A is better / B is better /
  can't tell". **Ratings are submitted as GitHub issues** (decided
  2026-10-13): a static page cannot call the GitHub API without a token, so the
  page accumulates choices in `localStorage` and, at the end of a batch, opens
  a prefilled `issues/new?title=ratings:<batch>&labels=ratings&body=<json>` URL;
  the rater (needs a GitHub account) clicks "Submit new issue". Compact JSON
  for 20 pairs is ~1 KB, well under the ~8 KB URL limit. A
  `uv run evomusic pull-ratings` command runs `gh issue list --label ratings
  --json body`, parses the JSON, and feeds the Bradley-Terry fit; processed
  issues are closed. Fallback: "download my ratings as JSON". Keep batches to
  ~20 pairs so a session fits a coffee break.
- **Playback is from the `.mid` files directly in the browser** (decided
  2026-10-13 to keep the repo small): `html-midi-player` (Magenta) from the
  jsDelivr CDN, with Magenta's SGM Plus soundfont streamed from Google's CDN on
  first play. A clip is ~400 bytes. The trade-off is an external dependency at
  view time and a different piano than the local FluidSynth render; mp3 rendering
  stays available locally via `publish --mp3` / `evomusic.render`.

### 6.5 Complexity knobs (`complexity.py`)
The doc proposes Kolmogorov–Sinai (KS) entropy and Lyapunov exponents. Both
are dynamical-systems quantities and need an embedding of the melody as a
trajectory; honest proxies for a discrete note sequence:

- **block-entropy rate** of the interval sequence (h = lim H(n+1) − H(n));
  this *is* the KS entropy of the symbolic sequence under the pitch partition;
- **compression ratio** (zlib / LZ77 length ÷ raw length): a Kolmogorov
  complexity proxy;
- **largest Lyapunov exponent** via Rosenstein's method on a delay embedding of
  the pitch series (`nolds.lyap_r`): tests whether a melody "diverges"
  like a chaotic system. Melodies are short (64–256 symbols), so expect noisy
  estimates; treat as experimental.

These feed in two ways, matching the doc: as a **target** in fitness
(`−|complexity − target|`) and as a **report** on finished pieces.

## 7. Evolutionary engine (`evolve.py`)

1. **v1 simple GA**: population 200–500, tournament selection, elitism 5 %,
   mutation rate knob, bar-aligned crossover, 200–1000 generations. Save best
   individual per generation as MIDI, log fitness terms to a CSV.
2. **Diversity preservation**: fitness sharing or niching, plus "keep a few
   low-scorers" as the doc suggests.
3. **MAP-Elites (quality-diversity)**, agreed as the default: archive indexed
   by 2–3 behaviour descriptors (e.g. note density × interval entropy × pitch
   range). Produces a *map* of good melodies of different characters, which is
   exactly what the "knobs" want. `pyribs` does this out of the box. Export:
   every archive cell as `out/<run>/cell_<i>_<j>.mid`, and the global best as
   `out/<run>/star_<fitness>.mid` (+ mp3).
4. **Multi-objective** (pymoo NSGA-II) if weighting the fitness terms turns out
   to be a fight.

### 6.6 Calibration finding (2026-10-13)

Scoring 3,000 real Hooktheory segments with the v1 rules gives mean 0.70
(p10 0.59, p90 0.80) while the GA reaches 0.99 in 200 generations. The rules
are a floor, not a target: they are easy to satisfy perfectly and real pop
breaks them routinely. Corpus numbers to recalibrate against (p10 / p40 / p70):

| descriptor | Hooktheory | v1 rule assumed |
|---|---|---|
| fraction of stepwise intervals | 0.32 / 0.55 / 0.69 | band 0.55–0.80 |
| pitch range (semitones) | 9 / 17 / 23 | band 7–16 (fixed to 9–23) |
| notes per bar | 2.5 / 4.1 / 5.7 | band 3–9 |
| rest fraction | 0.00 / 0.10 / 0.22 | band 0.03–0.20 |
| note durations (16ths) | 2: 48 %, 1: 20 %, 4: 17 % | 16ths were penalised as "off-beat" |
| chord usage | I 20 %, vi 19 %, IV 15 %, V 13 % | — |
| cadence rule satisfied | 0.36 | segments are excerpts; cadence must not dominate |

Consequence: the corpus n-gram term (6.2) and the critic (6.3) are the real
fitness; rules should move to soft bands derived from these percentiles.

### 6.7 Critic as proxy for human evaluation (decided 2026-10-13)

Renaldas: "Build the best possible critic, use it as a proxy for human
evaluation and work without human intervention for now. Aim to create music
that matches the complexity and critical evaluation on par with the
professional music from the datasets."

Implementation (`evomusic/critic/`):

1. **Tokenizer**: REMI-style per bar `BAR, POS_p, CH_c, P_x|REST, D_d`; 8-bar
   windows (the GA's phrase length) from Hooktheory, hop 2-4 bars.
2. **Language model**: 6-layer GPT-style transformer, d=256, ~4.9 M params,
   trained on 37.7k TRAIN windows. Fitness term = mean NLL scored against the
   held-out p10..p90 band (as for the n-grams).
3. **Discriminator**: same backbone initialised from the LM, real vs fake head
   on the `<eos>` state. Fakes: GA mutation operators applied to real windows
   at three strengths, pitch noise, bar shuffle, chord shuffle, melody/chord
   swap between songs, random genomes, and GA output from adversarial rounds.
4. **Adversarial rounds** (`critic/adversarial.py`): evolve against the
   current critic, add the GA's top distinct individuals to the fake pool,
   retrain, repeat. Logged per round: does the critic still accept real
   music, and did retraining close the exploit the GA found?
5. **Grading** (`evomusic/evaluate.py`): an *independent* discriminator (other
   seed, never trained on the final archive) plus complexity descriptors.
   "On par" = evolved medians inside the real p10..p90 band on every metric and
   AUC(real vs evolved) close to 0.5 under the independent critic.
6. **MAP-Elites** (`evomusic/mapelites.py`) over density x melodic entropy
   produces the final archive; the global best is `star_*.mid`.

### 6.8 Results, 2026-10-14 (batch 003 on the site)

Adversarial v1 (GA, 128 near-identical individuals kept per run) only taught the
discriminator to memorise: a fresh GA run still scored 0.84 under the "retrained"
critic. v2 fixed this with diverse MAP-Elites fakes, validation on held-out
*runs*, and a geometric-mean ensemble of the last three discriminators. Over
rounds 5-8 fresh held-out GA output settled at ~0.80 p_real vs 0.82-0.85 for
real music under a newly trained critic.

The first final archive still exploited the ensemble it was optimised against
(0.88 vs real 0.84) and an independent grader (fresh from the LM, other seed,
never sees the final archive) separated it from real music with AUC 0.84: too
many chords, no rests, too many leaps, LM loss at the edge of the real band.
Adding the descriptor-matching term (`fitness/descriptors.py`, real p25..p75
targets) and a tighter LM target fixed it:

| measure | real median | evolved median | evolved inside real p10..p90 |
|---|---|---|---|
| independent grader p_real | 0.77 | 0.80 | 96 % (AUC 0.36) |
| LM NLL | 0.75 | 0.91 | 96 % |
| distinct chords / 8 bars | 4 | 3 | 98 % |
| rest fraction | 0.125 | 0.125 | 98 % |
| step fraction | 0.61 | 0.64 | 98 % |
| entropy rate | 0.17 | 0.19 | 98 % |
| compression | 0.42 | 0.44 | 98 % |

Caveats: the grader shares the LM backbone and the synthetic fake families with
the GA-facing critics, so shared blind spots are possible; the critic is weak
on subtle edits (AUC ~0.6 vs single mutations, ~0.67 vs bar shuffles); and the
Hooktheory corpus mixes pop with game and film music. Blind human A/B on batch
003 is the real test.

Reproduce: `evomusic.critic.train_lm` -> `evomusic.critic.adversarial` ->
`scripts_final.sh` (grader) -> `evomusic.fitness.descriptors fit` ->
`evomusic.mapelites --lm --lm-w 2 --disc r6 r7 r8 --desc ...` -> `evomusic.evaluate`.

## 8. Evaluation: how do we know it is working?

- Fitness curves per term over generations (should rise, and not saturate
  trivially by exploiting one rule — watch for reward hacking).
- Corpus-distance metrics (MusPy-style): pitch-class entropy, scale consistency,
  groove consistency, compared against held-out corpus tunes and against random.
- Listening sessions with Giorgia: blind A/B of gen-0, gen-N, and real folk tunes.
- Diversity: pairwise edit distance within the final population / archive.

## 9. Milestones

| # | milestone | deliverable | est. |
|---|---|---|---|
| M0 | scaffold (done) | repo, uv env, random melody → `out/smoke.mid` | done 2026-10-13 |
| M0b | render | `.mid` → mp3 via fluidsynth + ffmpeg (`evomusic/render.py`) | done 2026-10-13 |
| M1 | data | Hooktheory parsed to melody+chord genome (17,478 segments, `evomusic/corpus/hooktheory.py`); POP909 / Nottingham parsers still to do | Hooktheory done 2026-10-13 |
| M2 | rules + GA | `evomusic.evolve` with 9 hand-written rules (`fitness/rules.py`) produces 8-bar tunes with chords; first batch on the site | v1 done 2026-10-13 |
| M3 | corpus critic | `fitness/ngram.py`: interpolated n-grams over melody / chords / chord-tones, scaled by held-out percentiles; batch 002 on the site compares reference vs rules vs rules+corpus | v1 done 2026-10-13 |
| M4 | learned critic | transformer LM + 8 adversarial rounds + independent grader; archive on par with real pop on every measured axis (section 6.8) | done 2026-10-14 |
| M5 | human feedback | GitHub Pages site (`docs/`: listen + A/B rate, ratings as GitHub issues) done 2026-10-13; `evomusic.ratings pull/rank` (GitHub issues → Bradley-Terry) done; reward-model term still to do | 1 day |
| M6 | knobs | `complexity.py` (entropy rate, compression, Lyapunov) + MAP-Elites archive with starred best | code done 2026-10-13 |
| M7 | more styles | folk, then baroque (Bach chorales, counterpoint rules), then jazz (Weimar DB); expressive velocity | open |

## 10. Repo layout

```
evomusic/            package: corpus.py genome.py midi.py fitness/ evolve.py cli.py
data/raw/            downloads (git-ignored)
data/processed/      parquet (git-ignored)
data/samples/        a few tiny tunes, committed, for tests
notebooks/           exploration only
out/                 generated MIDI/WAV (git-ignored)
docs/                GitHub Pages site (rating page) + notes, listening-test logs
```

## 11. Decisions log

Answered by Renaldas on 2026-10-13:

1. **Style**: pop first, then folk, baroque, jazz (may change).
2. **Chords**: yes, chords underneath from the start.
3. **Feedback UI**: small web page, hosted on GitHub Pages.
4. **Output**: diversity archive, with the best melody marked with a star in its filename.
5. **Complexity knob**: entropy rate + compression ratio are the real knob; KS entropy and Lyapunov are side experiments.
6. **Audio tools**: fluidsynth, soundfont and ffmpeg installed; use mp3 for anything web-facing.
7. **Essen licence**: acceptable; noted in README for future consideration.

Answered later the same day:

8. **Ratings backend**: GitHub issues, prefilled from the static page (see 6.4). Revisit if raters without GitHub accounts appear.
9. **GitHub repo**: public, https://github.com/rejunity/gc-exploring-music (GPL-3.0), GitHub Pages from `docs/` → https://rejunity.github.io/gc-exploring-music/. Raw POP909 / HLSD data stay out of the repo; only our generated mp3s are committed.
10. **Critic target**: Western pop (Hooktheory via Sheet Sage) as the primary corpus; POP909 blended in with a lower weight (e.g. 70/30) for statistics and critic pre-training. Make the mix a config knob and A/B it in a listening test.

11. **Reference clips**: real Hooktheory segments may be published on the site with attribution (CC BY-NC-SA).
12. **Giorgia** has a GitHub account; Renaldas adds her once the site is polished.

Observations after the first corpus-guided runs (2026-10-13): the GA saturates
every n-gram term too (1.0 on melody / chord-tone) while real test segments sit
at ~0.5, because the terms reward *the most probable* continuation, not a
*plausible* one. Candidates to fix this: score against the corpus percentile
band (penalise being more predictable than p90 as well as less than p10), add a
diversity / novelty term, or move to the critic (6.3). The blind A/B results on
batch 002 will tell us whether listeners agree.

## 12. References found during the survey

- Datasets: [POP909](https://github.com/music-x-lab/POP909-Dataset), [Hooktheory Lead Sheet Dataset](https://github.com/wayne391/lead-sheet-dataset), [Lakh MIDI](https://colinraffel.com/projects/lmd/), [Aria-MIDI](https://huggingface.co/datasets/loubb/aria-midi) ([paper](https://arxiv.org/abs/2504.15071)), [GigaMIDI](https://arxiv.org/abs/2502.17726v1), [POP909](https://arxiv.org/pdf/2008.07142), [KernScores / Essen](https://kernscores.stanford.edu/help/data), [MTCFeatures](https://pypi.org/project/MTCFeatures), [Weimar Jazz DB](https://jazzomat.hfm-weimar.de/), [OpenEWLD](https://projects.dharc.unibo.it/musow/view-1645596444-6679177), [WikiMusicText](https://huggingface.co/datasets/sander-wood/wikimusictext), [PDMX](https://github.com/speechprojects/PDMX), [MeloSol](https://www.osu.tests.sfulib4.publicknowledgeproject.org/index.php/EMR/article/view/7645)
- Tools: [symusic](https://symusic.readthedocs.io/en/stable/introduction.html), [MidiTok](https://github.com/Natooz/MidiTok), [MusPy](https://pypi.org/project/muspy), [pymoo](https://arxiv.org/pdf/2002.04504), [LEAP](https://leap-gmu.readthedocs.io/en/stable/index.html)
- Evolutionary music: [Mycka et al. 2023, rule-based harmonisation fitness](https://pages.mini.pw.edu.pl/~zychowskia/Publications/Harmonization_JOCS2023.pdf), [Ponce de León et al., data-based multi-objective melody generation](https://grfia.dlsi.ua.es/repositori/grfia/pubs/344/Data%20based%20melody%20generation%20through%20multi%20objective%20evolutionary%20computation%20(post-print).pdf), [Murray et al. 2012 on interactive fitness](https://axon.cs.byu.edu/Dan/papers/temp/murray2012mume.pdf), [HarmonyMapper, MAP-Elites for chords](https://ceur-ws.org/Vol-3626/paper2.pdf), [QD in sound generation](https://arxiv.org/pdf/2512.02783)
- Expectation / complexity: [IDyOM overview](https://ppidyom.ccml.gtcmt.gatech.edu/articles/Overview.html), [computational features for symbolic melody analysis (2026)](https://arxiv.org/pdf/2608.19061), [KS entropy & Lyapunov intro](https://arxiv.org/pdf/nlin/0411023)
