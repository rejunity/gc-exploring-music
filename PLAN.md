# PLAN — Evolutionary MIDI melody generation

Status: draft v1, 2026-10-13. Source: Renaldas' "Evolutionary Music Generation
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

- **Start monophonic.** One melody line, fixed key, 4/4, 8–16 bars. Harmony,
  polyphony and expressive timing come later (section 9).
- **Discrete genome.** A melody is a list of `(pitch | rest, duration)` genes on
  a 16th-note grid. Simple, every mutation stays valid, trivially serialises to MIDI.
- **Corpus = folk melodies first.** Clean, monophonic, metrically regular,
  tens of thousands of tunes, public. Pop/jazz/classical corpora are added once
  the pipeline works.
- **Code first, notebooks for exploration only.** Everything reusable lives in
  the `evomusic` package; notebooks import from it.

## 3. Datasets

### 3.1 Monophonic melody corpora (primary, for rules calibration + critic training)

| dataset | size | format | license / access | why |
|---|---|---|---|---|
| **Essen Folksong Collection** | ~8,500 tunes (5,365 German, 2,246 Chinese) | `**kern` (KernScores), ABC, also bundled inside `music21` corpus | legal status "unclear" per music21 notes, universally used in research | the standard monophonic corpus; phrase boundaries annotated |
| **Nottingham Music Database** | ~1,200 British/American folk tunes | ABC (with chords), MIDI | free, maintained by Seymour Shlien at ifdo.ca | melody + chord symbols, ideal first corpus |
| **Meertens Tune Collections (MTC)** via `MTCFeatures` | 8k+ Dutch/German melodies as feature sequences | Python package, Zenodo | open | ready-made features (intervals, metric weights, phrase ends) |
| **Weimar Jazz Database** | 456 transcribed jazz solos | SQLite + MIDI export | free download, jazzomat.hfm-weimar.de | later: a "jazz" style target for the critic |
| **OpenEWLD / WikiMusicText** | ~1,000–2,000 lead sheets | MusicXML / ABC | public domain subset of Wikifonia (Wikifonia itself closed in 2013) | melody + chords, pop/standards |
| **POP909** | 909 pop songs | MIDI with melody / bridge / piano tracks, chord + phrase labels | research use | pop melody track with chords |
| **MeloSol corpus** | ~780 solfege-style melodies | kern | open | small, very clean |

### 3.2 Large polyphonic MIDI corpora (later, for a stronger critic or harmony)

| dataset | size | notes |
|---|---|---|
| Lakh MIDI (LMD), "clean_midi" subset | 176k files total, ~17k named by artist/title | CC-BY 4.0; noisy, sequenced not performed |
| MAESTRO v3 | 1,282 performances, ~200 h | classical piano, expressive timing |
| GiantMIDI-Piano | 10,854 pieces | transcribed from audio |
| Aria-MIDI (ICLR 2025) | 1.19M files, ~100k h piano | huggingface `loubb/aria-midi`; "Unique" subset is 32k files |
| GigaMIDI (TISMIR 2025) | 1.4M files, all GM instruments | HF access gated to research use |
| PDMX | 250k public-domain MusicXML scores | Zenodo |

### 3.3 Data plan

1. Download Nottingham (ABC + MIDI) and Essen (kern via KernScores, or just
   `music21.corpus`) into `data/raw/`. Both are < 50 MB.
2. Write `evomusic/corpus.py`: parse ABC / kern / MIDI into the canonical
   genome representation (section 5). Normalise: transpose to C major / A minor,
   quantise to 16ths, drop tunes with unusual metres for v1.
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
| `fluidsynth` + `fluid-soundfont-gm` | render `.mid` to `.wav` to actually listen | `apt install fluidsynth fluid-soundfont-gm` (not yet run) |
| `lilypond` or MuseScore | render notation PNGs for the notebook | optional, `apt install lilypond` |

Deliberately *not* used: DEAP (last release 2020), musicaiz (AGPL, stale).
The GA loop is ~100 lines of numpy; owning it keeps mutation operators
musical rather than generic.

## 5. Genome and phenotype

```
Genome   = list[Gene];  Gene = (pitch: int | None, dur: int)   # dur in 16ths
Phenotype = MIDI file (+ optional chord track later)
```

Invariants: durations sum to `bars * 16`; pitches within a configurable range.
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
- repetition / self-similarity: at least one repeated bar, but not all bars equal.

### 6.2 Corpus statistics (`ngram.py`), after section 3.3
- n-gram models (n = 2..5) over (interval, duration) pairs fitted on the corpus;
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
- Interface: start with a CLI (`uv run evomusic rate`) that plays rendered
  WAVs; a small web page later if Giorgia wants to rate remotely.

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
3. **MAP-Elites (quality-diversity)**: archive indexed by 2–3 behaviour
   descriptors (e.g. note density × interval entropy × pitch range). Produces a
   *map* of good melodies of different characters instead of one winner, which
   is exactly what the "knobs" want. `pyribs` does this out of the box.
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
| M1 | data | Nottingham + Essen parsed to canonical genome, stats notebook | 1–2 days |
| M2 | rules + GA | `evomusic evolve --fitness rules` produces listenable 8-bar tunes; fluidsynth rendering | 2–3 days |
| M3 | corpus critic | n-gram fitness term; compare M2 vs M3 output in a listening test | 1–2 days |
| M4 | learned critic | transformer LM critic, NLL fitness | 3–5 days |
| M5 | human feedback | pairwise rating CLI, Bradley-Terry, reward model term | 2–3 days |
| M6 | knobs | complexity measures as fitness targets + MAP-Elites archive | 3–4 days |
| M7 | beyond monophony | chord track from Nottingham chords; two-voice counterpoint rules; expressive velocity | open |

## 10. Repo layout

```
evomusic/            package: corpus.py genome.py midi.py fitness/ evolve.py cli.py
data/raw/            downloads (git-ignored)
data/processed/      parquet (git-ignored)
data/samples/        a few tiny tunes, committed, for tests
notebooks/           exploration only
out/                 generated MIDI/WAV (git-ignored)
docs/                notes, listening-test logs
```

## 11. Open questions for Renaldas and Giorgia

1. **Style target.** Folk first because it is clean. Is there a style you
   actually want to hear (baroque, pop hooks, jazz lines, film cues)? It
   changes the corpus and the rules.
2. **Monophonic for how long?** Is melody-only output satisfying for the first
   few milestones, or do you want chords under it from day one (Nottingham has
   chord symbols, so that is cheap)?
3. **Who gives the human feedback and how?** Terminal A/B is fastest to build.
   A web page lets Giorgia rate from a phone. Which?
4. **Novelty vs. quality.** Would you rather have one great melody per run
   (plain GA) or a map of diverse decent ones (MAP-Elites)? I lean MAP-Elites
   because it makes the "knobs" first-class.
5. **Complexity measures.** KS entropy and Lyapunov are interesting but
   melodies are very short series. Fine to treat them as experiments and lean on
   entropy rate / compression ratio for the real knob?
6. **GPU budget.** The box has a GTX 1080 (8 GB). Enough for a small critic;
   not for training on Aria-MIDI-scale data. OK to keep the critic small?
7. **Data licensing.** Essen's legal status is "unclear" but it is used in
   every paper. OK for a personal research project? (Nottingham is unambiguous.)
8. **Playback.** May I `apt install fluidsynth fluid-soundfont-gm` on this
   machine so runs can be rendered to WAV, or do you prefer listening to the
   `.mid` files elsewhere (DAW, phone)?

## 12. References found during the survey

- Datasets: [Lakh MIDI](https://colinraffel.com/projects/lmd/), [Aria-MIDI](https://huggingface.co/datasets/loubb/aria-midi) ([paper](https://arxiv.org/abs/2504.15071)), [GigaMIDI](https://arxiv.org/abs/2502.17726v1), [POP909](https://arxiv.org/pdf/2008.07142), [KernScores / Essen](https://kernscores.stanford.edu/help/data), [MTCFeatures](https://pypi.org/project/MTCFeatures), [Weimar Jazz DB](https://jazzomat.hfm-weimar.de/), [OpenEWLD](https://projects.dharc.unibo.it/musow/view-1645596444-6679177), [WikiMusicText](https://huggingface.co/datasets/sander-wood/wikimusictext), [PDMX](https://github.com/speechprojects/PDMX), [MeloSol](https://www.osu.tests.sfulib4.publicknowledgeproject.org/index.php/EMR/article/view/7645)
- Tools: [symusic](https://symusic.readthedocs.io/en/stable/introduction.html), [MidiTok](https://github.com/Natooz/MidiTok), [MusPy](https://pypi.org/project/muspy), [pymoo](https://arxiv.org/pdf/2002.04504), [LEAP](https://leap-gmu.readthedocs.io/en/stable/index.html)
- Evolutionary music: [Mycka et al. 2023, rule-based harmonisation fitness](https://pages.mini.pw.edu.pl/~zychowskia/Publications/Harmonization_JOCS2023.pdf), [Ponce de León et al., data-based multi-objective melody generation](https://grfia.dlsi.ua.es/repositori/grfia/pubs/344/Data%20based%20melody%20generation%20through%20multi%20objective%20evolutionary%20computation%20(post-print).pdf), [Murray et al. 2012 on interactive fitness](https://axon.cs.byu.edu/Dan/papers/temp/murray2012mume.pdf), [HarmonyMapper, MAP-Elites for chords](https://ceur-ws.org/Vol-3626/paper2.pdf), [QD in sound generation](https://arxiv.org/pdf/2512.02783)
- Expectation / complexity: [IDyOM overview](https://ppidyom.ccml.gtcmt.gatech.edu/articles/Overview.html), [computational features for symbolic melody analysis (2026)](https://arxiv.org/pdf/2608.19061), [KS entropy & Lyapunov intro](https://arxiv.org/pdf/nlin/0411023)
