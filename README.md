# exploring-music-with-giorgia

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

## Listening

```bash
uv run python -m evomusic.render out/smoke.mid   # -> out/smoke.mp3 (fluidsynth + ffmpeg)
```

Needs `apt install fluidsynth fluid-soundfont-gm ffmpeg`.

## Data licensing notes

- **Essen Folksong Collection**: the legal status is "unclear" (per the music21
  corpus notes; the data was entered by many contributors over decades). It is
  used universally in research. Accepted for this personal research project on
  2026-10-13; **reconsider before any public release of models or derived data.**
- **POP909** and **Hooktheory Lead Sheet Dataset**: research / academic use
  only. Do not commit the raw data to the repo.
- **Nottingham**, **Lakh MIDI (CC-BY 4.0)**, **OpenEWLD (public domain)**: unproblematic.
