"""Publish a batch of .mid files to the GitHub Pages site under docs/.

    uv run python -m evomusic.publish BATCH_ID --title "..." [--notes "..."] \
        [--label random out/a.mid out/b.mid] [--label evolved out/run/star_*.mid]

Each --label starts a group; the .mid paths that follow belong to it. Files are
copied to docs/audio/<batch>/ and appended to docs/batches.json; the site plays
the .mid files directly in the browser (html-midi-player), so no audio is rendered.
If a file's directory has a summary.json (written by evomusic.evolve) its fitness
and chord progression are attached to the clip.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import date
from pathlib import Path

from evomusic.render import midi_to_wav, wav_to_mp3

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
BATCHES = DOCS / "batches.json"


def set_docs(path: Path) -> None:
    global DOCS, BATCHES
    DOCS, BATCHES = path, path / "batches.json"


def load() -> dict:
    if BATCHES.exists():
        return json.loads(BATCHES.read_text())
    return {"repo": "rejunity/gc-exploring-music", "batches": []}


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    # split "--label name files..." groups by hand so argparse stays simple
    groups: list[tuple[str, list[Path]]] = []
    rest: list[str] = []
    i = 0
    while i < len(argv):
        if argv[i] == "--label":
            groups.append((argv[i + 1], [])); i += 2
        elif groups and not argv[i].startswith("--"):
            groups[-1][1].append(Path(argv[i])); i += 1
        else:
            rest.append(argv[i]); i += 1
    ap = argparse.ArgumentParser()
    ap.add_argument("batch")
    ap.add_argument("--title", required=True)
    ap.add_argument("--notes", default="")
    ap.add_argument("--replace", action="store_true", help="overwrite a batch with the same id")
    ap.add_argument("--docs", type=Path, help="site folder to publish into (default: this repo's docs/)")
    ap.add_argument("--mp3", action="store_true", help="also render mp3 (local listening; not needed for the site)")
    ap.add_argument("--credits", type=Path, help="json: {source filename: {text, url}} attached to clips as `credit`")
    a = ap.parse_args(rest)
    if not groups:
        sys.exit("give at least one --label NAME file.mid ...")

    if a.docs:
        set_docs(a.docs)
    credits = json.loads(a.credits.read_text()) if a.credits else {}
    data = load()
    if any(b["id"] == a.batch for b in data["batches"]):
        if not a.replace:
            sys.exit(f"batch {a.batch} exists; use --replace")
        data["batches"] = [b for b in data["batches"] if b["id"] != a.batch]
        shutil.rmtree(DOCS / "audio" / a.batch, ignore_errors=True)

    out = DOCS / "audio" / a.batch
    out.mkdir(parents=True, exist_ok=True)
    clips = []
    for label, files in groups:
        for src in files:
            summary = src.parent / "summary.json"
            info = {}
            if summary.exists():
                for s in json.loads(summary.read_text()).get("saved", []):
                    if s["file"] == src.name:
                        info = {"fitness": None if s.get("fitness") is None else round(s["fitness"], 3), "chords": s["chords"]}
                        if "descriptors" in s:
                            d = s["descriptors"]
                            info["desc"] = {"density": round(d["notes_per_bar"], 2), "entropy": round(d["symbol_entropy"], 2),
                                            "range": d["pitch_range"], "compression": round(d["compression"], 2)}
                        if "p_real" in s:
                            info["p_real"] = round(s["p_real"], 3)
                        if "amt_gap" in s:
                            info["amt_gap"] = s["amt_gap"]
                        if "plan" in s:
                            info["plan"] = s["plan"]
            if src.name in credits:
                info["credit"] = credits[src.name]
            cid = f"{label}-{len([c for c in clips if c['label'] == label]) + 1:02d}"
            mid = out / f"{cid}.mid"
            shutil.copy(src, mid)
            if a.mp3:
                wav = midi_to_wav(mid)
                info["mp3"] = f"audio/{a.batch}/{wav_to_mp3(wav).name}"; wav.unlink()
            clips.append({"id": cid, "label": label, "mid": f"audio/{a.batch}/{mid.name}", "source": str(src), **info})
            print("published", cid, "<-", src)
    data["batches"].append({"id": a.batch, "title": a.title, "notes": a.notes, "date": date.today().isoformat(),
                            "clips": clips})
    BATCHES.write_text(json.dumps(data, indent=2) + "\n")
    print(f"{len(clips)} clips in batch {a.batch}; docs/batches.json updated")


if __name__ == "__main__":
    main()
