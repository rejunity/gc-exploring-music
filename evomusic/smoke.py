"""Smoke test: write generation-0 individuals (random melody + random chords) to out/.

    uv run python -m evomusic.smoke            # out/smoke.mid, fully random
    uv run python -m evomusic.smoke --scale    # out/smoke_scale.mid, weakly constrained (C major)
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

from evomusic.genome import Genome


def main(argv: list[str]) -> None:
    in_scale = "--scale" in argv
    out = Path("out"); out.mkdir(exist_ok=True)
    g = Genome.random(in_scale=in_scale, rng=random.Random(0))
    path = out / ("smoke_scale.mid" if in_scale else "smoke.mid")
    g.to_midi().save(path)
    print(f"wrote {path}  chords: {g.chord_names()}")


if __name__ == "__main__":
    main(sys.argv[1:])
