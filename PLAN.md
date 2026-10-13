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
| **Hooktheory Lead Sheet Dataset (HLSD)** | 11,329 segments from TheoryTab, all 4/4 | JSON + MIDI, chords as symbols and Roman numerals | academic use only, github.com/wayne391/lead-sheet-dataset | Western pop hooks with functional harmony; ideal for chord n-grams |
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

1. Download POP909 and HLSD (pop), then Nottingham (ABC + MIDI; folk with
   chords) into `data/raw/`. Essen comes via `music21.corpus`. All < 200 MB.
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
  can't tell". Ratings are POSTed to a Google Apps Script web app that appends
  rows to a Google Sheet (free, no server, in Renaldas' Google account); the
  page also offers "download my ratings as JSON" as a fallback. A
  `uv run evomusic pull-ratings` command fetches the sheet as CSV and feeds the
  Bradley-Terry fit. Keep batches to ~20 pairs so a session fits a coffee break.
- Audio on the page is mp3 at 96 kbps, about 240 KB per 8-bar clip with
  sustain tails; a batch of 40 clips is under 10 MB, fine for a git repo. If the repo
  grows, move clips to GitHub Releases or git-lfs.

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
| M1 | data | POP909 + HLSD + Nottingham parsed to canonical melody+chord genome, stats notebook | 1–2 days |
| M2 | rules + GA | `evomusic evolve --fitness rules` produces listenable 8-bar pop tunes with chords | 2–3 days |
| M3 | corpus critic | n-gram fitness term; compare M2 vs M3 output in a listening test | 1–2 days |
| M4 | learned critic | transformer LM critic, NLL fitness | 3–5 days |
| M5 | human feedback | GitHub Pages A/B rating page + Apps Script sheet, Bradley-Terry, reward model term | 2–3 days |
| M6 | knobs | complexity measures as fitness targets + MAP-Elites archive | 3–4 days |
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

Still open:

- **Ratings backend**: Google Apps Script + Sheet is the proposed free option
  (needs a one-time setup in Renaldas' Google account). Alternatives: Supabase
  free tier, or ratings submitted as GitHub issues. Which?
- **GitHub repo**: public or private? (GitHub Pages on a private repo needs a
  paid plan; a public repo must not contain POP909/HLSD data, only our mp3s.)
- **POP909 is Chinese pop, HLSD is Western pop hooks.** Both are fine for
  statistics; which sound is the target for the critic?

## 12. References found during the survey

- Datasets: [POP909](https://github.com/music-x-lab/POP909-Dataset), [Hooktheory Lead Sheet Dataset](https://github.com/wayne391/lead-sheet-dataset), [Lakh MIDI](https://colinraffel.com/projects/lmd/), [Aria-MIDI](https://huggingface.co/datasets/loubb/aria-midi) ([paper](https://arxiv.org/abs/2504.15071)), [GigaMIDI](https://arxiv.org/abs/2502.17726v1), [POP909](https://arxiv.org/pdf/2008.07142), [KernScores / Essen](https://kernscores.stanford.edu/help/data), [MTCFeatures](https://pypi.org/project/MTCFeatures), [Weimar Jazz DB](https://jazzomat.hfm-weimar.de/), [OpenEWLD](https://projects.dharc.unibo.it/musow/view-1645596444-6679177), [WikiMusicText](https://huggingface.co/datasets/sander-wood/wikimusictext), [PDMX](https://github.com/speechprojects/PDMX), [MeloSol](https://www.osu.tests.sfulib4.publicknowledgeproject.org/index.php/EMR/article/view/7645)
- Tools: [symusic](https://symusic.readthedocs.io/en/stable/introduction.html), [MidiTok](https://github.com/Natooz/MidiTok), [MusPy](https://pypi.org/project/muspy), [pymoo](https://arxiv.org/pdf/2002.04504), [LEAP](https://leap-gmu.readthedocs.io/en/stable/index.html)
- Evolutionary music: [Mycka et al. 2023, rule-based harmonisation fitness](https://pages.mini.pw.edu.pl/~zychowskia/Publications/Harmonization_JOCS2023.pdf), [Ponce de León et al., data-based multi-objective melody generation](https://grfia.dlsi.ua.es/repositori/grfia/pubs/344/Data%20based%20melody%20generation%20through%20multi%20objective%20evolutionary%20computation%20(post-print).pdf), [Murray et al. 2012 on interactive fitness](https://axon.cs.byu.edu/Dan/papers/temp/murray2012mume.pdf), [HarmonyMapper, MAP-Elites for chords](https://ceur-ws.org/Vol-3626/paper2.pdf), [QD in sound generation](https://arxiv.org/pdf/2512.02783)
- Expectation / complexity: [IDyOM overview](https://ppidyom.ccml.gtcmt.gatech.edu/articles/Overview.html), [computational features for symbolic melody analysis (2026)](https://arxiv.org/pdf/2608.19061), [KS entropy & Lyapunov intro](https://arxiv.org/pdf/nlin/0411023)
