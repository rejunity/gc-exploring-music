"""Render .mid -> .wav (fluidsynth) -> .mp3 (ffmpeg) so pieces can be listened to
locally or served from the GitHub Pages rating site.

Usage: uv run python -m evomusic.render out/smoke.mid [more.mid ...]
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

SOUNDFONT = Path("/usr/share/sounds/sf2/FluidR3_GM.sf2")


def midi_to_wav(mid: Path, wav: Path | None = None, soundfont: Path = SOUNDFONT) -> Path:
    wav = wav or mid.with_suffix(".wav")
    subprocess.run(
        ["fluidsynth", "-ni", "-g", "0.8", "-F", str(wav), "-r", "44100", str(soundfont), str(mid)],
        check=True, capture_output=True,
    )
    return wav


def wav_to_mp3(wav: Path, mp3: Path | None = None, bitrate: str = "96k") -> Path:
    mp3 = mp3 or wav.with_suffix(".mp3")
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-codec:a", "libmp3lame", "-b:a", bitrate, str(mp3)],
        check=True,
    )
    return mp3


def render(mid: Path, keep_wav: bool = False) -> Path:
    wav = midi_to_wav(mid)
    mp3 = wav_to_mp3(wav)
    if not keep_wav:
        wav.unlink()
    return mp3


def main(argv: list[str]) -> None:
    for tool in ("fluidsynth", "ffmpeg"):
        if shutil.which(tool) is None:
            sys.exit(f"{tool} not found; apt install fluidsynth fluid-soundfont-gm ffmpeg")
    for arg in argv:
        print("wrote", render(Path(arg)))


if __name__ == "__main__":
    main(sys.argv[1:])
