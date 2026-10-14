"""Choose which archive pieces to publish: the starred best plus n others.

    python -m evomusic.pick RUN_DIR [--n 7] [--seed 0]     # -> RUN_DIR/chosen.txt

CVT archives: cells are sampled in proportion to how much real pop lives there,
so the published set has real pop's mix of forms and harmonic colours.
Grid archives: evenly spread over the grid (old behaviour).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evomusic.behaviour import pick


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--n", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    s = json.loads((a.run / "summary.json").read_text())
    others = [e for e in s["saved"] if e["file"] != s["star"]]
    chosen = [s["star"]] + [e["file"] for e in pick(others, a.n, a.seed)]
    (a.run / "chosen.txt").write_text("\n".join(str(a.run / f) for f in chosen))
    print("\n".join(chosen))


if __name__ == "__main__":
    main()
