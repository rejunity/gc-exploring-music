"""Pull A/B ratings filed as GitHub issues (label `ratings`) and rank clips with
a Bradley-Terry model.

    uv run python -m evomusic.ratings pull                 # gh issue list -> data/ratings/*.json (needs gh auth)
    uv run python -m evomusic.ratings rank [--batch ID]    # fit Bradley-Terry over everything in data/ratings/
    uv run python -m evomusic.ratings rank --file ratings-giorgia.json   # a downloaded JSON instead of issues

Rating record (as written by docs/app.js):
    {"t": iso-time, "rater": str, "batch": id, "a": clip id, "b": clip id, "choice": "A" | "B" | "tie"}
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIR = ROOT / "data" / "ratings"
REPO = json.loads((ROOT / "docs" / "batches.json").read_text())["repo"]


def pull() -> int:
    out = subprocess.run(["gh", "issue", "list", "--repo", REPO, "--label", "ratings", "--state", "all",
                          "--limit", "500", "--json", "number,body,author,createdAt"], check=True, capture_output=True, text=True)
    DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for issue in json.loads(out.stdout):
        m = re.search(r"```json\s*(.*?)\s*```", issue["body"], re.S)
        if not m:
            continue
        try:
            ratings = json.loads(m.group(1))
        except json.JSONDecodeError:
            print(f"issue #{issue['number']}: bad JSON, skipped"); continue
        path = DIR / f"issue-{issue['number']}.json"
        path.write_text(json.dumps({"issue": issue["number"], "author": issue["author"]["login"],
                                    "createdAt": issue["createdAt"], "ratings": ratings}, indent=1))
        n += len(ratings)
    print(f"{n} ratings in {DIR.relative_to(ROOT)}")
    return n


def load(files: list[Path] | None = None) -> list[dict]:
    files = files or sorted(DIR.glob("*.json"))
    out = []
    for f in files:
        d = json.loads(f.read_text())
        out.extend(d["ratings"] if isinstance(d, dict) else d)
    return out


def bradley_terry(ratings: list[dict], iters: int = 200) -> dict[str, float]:
    """MM algorithm (Hunter 2004). Ties count as half a win each way."""
    wins: dict[str, float] = defaultdict(float)
    games: dict[tuple[str, str], float] = defaultdict(float)
    items = set()
    for r in ratings:
        a, b = r["a"], r["b"]
        items |= {a, b}
        key = tuple(sorted((a, b)))
        games[key] += 1
        if r["choice"] == "A":
            wins[a] += 1
        elif r["choice"] == "B":
            wins[b] += 1
        else:
            wins[a] += 0.5; wins[b] += 0.5
    p = {i: 1.0 for i in items}
    for _ in range(iters):
        new = {}
        for i in items:
            denom = sum(n / (p[i] + p[j]) for (x, y), n in games.items() for j in ((y,) if x == i else (x,) if y == i else ()))
            new[i] = (wins[i] + 0.01) / denom if denom else p[i]   # +0.01 keeps undefeated items finite
        s = sum(new.values()) / len(new)
        p = {i: v / s for i, v in new.items()}
    return p


def rank(ratings: list[dict], batch: str | None) -> None:
    if batch:
        ratings = [r for r in ratings if r["batch"] == batch]
    if not ratings:
        print("no ratings"); return
    p = bradley_terry(ratings)
    n = defaultdict(int)
    for r in ratings:
        n[r["a"]] += 1; n[r["b"]] += 1
    print(f"{len(ratings)} ratings, {len(p)} clips, raters: {sorted({r['rater'] for r in ratings})}")
    print(f"{'clip':22s} {'strength':>8s} {'log':>6s} {'n':>3s}")
    for clip, v in sorted(p.items(), key=lambda kv: -kv[1]):
        print(f"{clip:22s} {v:8.3f} {math.log(v):6.2f} {n[clip]:3d}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pull")
    rk = sub.add_parser("rank")
    rk.add_argument("--batch")
    rk.add_argument("--file", type=Path, action="append")
    a = ap.parse_args()
    if a.cmd == "pull":
        pull()
    else:
        rank(load(a.file), a.batch)


if __name__ == "__main__":
    main()
