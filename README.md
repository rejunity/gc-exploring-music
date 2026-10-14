# 🎧 **[Listen and rate: https://rejunity.github.io/gc-exploring-music](https://rejunity.github.io/gc-exploring-music)**

# gc-exploring-music

Generative music via an evolutionary process. A population of random melodies is
evolved toward "musical" results under a fitness score that combines music-theory
rules, corpus statistics, a learned critic, and occasional human feedback. Output is
standard `.mid` files.

See [PLAN.md](PLAN.md) for the roadmap, datasets, and tooling.

## Quick start

```bash
uv sync                      # creates .venv with the core deps
uv run python -m evomusic.smoke   # writes out/smoke.mid (a random 8-bar melody)
```

## Data

All corpora are fetched by one script into `data/raw/` (git-ignored). The manifest
`data/manifest.json` pins URLs to commit SHAs / revisions where possible and
records a sha256 of every download plus a hash of the extracted file tree, so a
second machine can check it has byte-identical data.

```bash
uv run python -m evomusic.data list                 # what is in the manifest, what is on disk
uv run python -m evomusic.data download             # all "core" sets (~110 MB download, ~260 MB on disk)
uv run python -m evomusic.data download --tier all  # + Lakh clean_midi (234 MB download, 1.1 GB on disk)
uv run python -m evomusic.data verify               # re-hash disk vs manifest
uv run python -m evomusic.data download --record    # refresh hashes after a deliberate upstream update
```

| name | contents | style | licence |
|---|---|---|---|
| pop909 | 909 songs, melody/bridge/piano MIDI + chord/beat/key txt | pop (Chinese) | MIT |
| hooktheory | Sheet Sage release of TheoryTab: 26,175 segments (23,833 with melody) as JSON with key, meter, melody notes, chords; + 16k MIDI renderings in train/valid/test tarballs | pop (Western) | CC BY-NC-SA 3.0 |
| hlsd_sample | older Hooktheory lead-sheet repo snapshot: crawler + ~40 sample MIDI/JSON (full dump link is dead) | pop (Western) | academic only |
| openewld | 502 lead sheets, MusicXML with chords | pop / standards | public domain |
| wikimusictext | 1,010 lead sheets in ABC with metadata | pop / standards | CC BY-NC 4.0 |
| nottingham | ~1,200 folk tunes, ABC with chords | folk | free |
| essen | 8,473 monophonic tunes, **kern | folk | unclear, see below |
| bach_chorales | 370 four-part chorales, **kern | baroque | CC BY-NC-SA |
| weimar_jazz_db, weimar_jazz_midi | 456 transcribed solos, SQLite + MIDI | jazz | ODbL |
| lakh_clean_midi | 17,256 multi-track MIDI named by artist/title | pop/rock | CC BY 4.0 |

## Listening

```bash
uv run python -m evomusic.render out/smoke.mid   # -> out/smoke.mp3 (fluidsynth + ffmpeg)
```

Needs `apt install fluidsynth fluid-soundfont-gm ffmpeg`.

## Licensing

This repository, https://github.com/rejunity/gc-exploring-music, is licensed under GPL-3.0 (see LICENSE).

### Data licensing notes

- **Essen Folksong Collection**: the legal status is "unclear" (per the music21
  corpus notes; the data was entered by many contributors over decades). It is
  used universally in research. Accepted for this personal research project on
  2026-10-13; **reconsider before any public release of models or derived data.**
- **Hooktheory (Sheet Sage release)**: CC BY-NC-SA 3.0, derived from TheoryTab
  user contributions. Non-commercial, share-alike: anything trained on it or
  derived from it inherits that licence.
- **POP909** and the older **Hooktheory Lead Sheet Dataset** snapshot: research /
  academic use only. Do not commit the raw data to the repo.
- **Nottingham**, **Lakh MIDI (CC-BY 4.0)**, **OpenEWLD (public domain)**: unproblematic.
