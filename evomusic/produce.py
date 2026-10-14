"""Produce: explore the current state of a branch without evolving it.

Takes the pieces a branch currently holds (its latest archive, or the clips of
a batch), generates one large burst of variations (thousands of candidates,
far more and wilder than one evolution step), scores each against the knob
targets and a light quality floor, and keeps a diverse set. No selection loop,
no new generation, nothing is written back to the branch.

    uv run python -m evomusic.produce --site docs --n 8 --strength 3 \
        --knobs '{"complexity": 80, "predictability": 30}' --keep hook --out out/produced

Variation strength 1..5: subtle (one or two small edits per piece) to wild
(many edits, new phrase plans, new motifs, crossing pieces with each other).
Keep: "hook" (the motif of bar 1), "plan" (phrase structure), "harmony" (chords).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from pathlib import Path

import numpy as np

from evomusic.form import FormGenome, crossover_form, from_genome, load_stats, mutate_form, mutation_weights
from evomusic.genome import Genome
from evomusic.knobs import KNOBS, Knobs

ROOT = Path(__file__).resolve().parent.parent
STRENGTH = {1: (1.0, 0.0, 1.0), 2: (2.0, 0.15, 1.0), 3: (4.0, 0.3, 1.5), 4: (8.0, 0.45, 2.5), 5: (16.0, 0.6, 4.0)}
# strength -> (mean edits per piece, chance of crossing with another piece, boost for structural edits)


def _batch_key(name: str) -> tuple:
    m = re.match(r"(\d{8})-run(\d+)", name)
    return (m.group(1), int(m.group(2))) if m else (name, 0)


def load_sources(site: Path, batch: str | None = None) -> tuple[list[FormGenome], str]:
    """Pieces to explore. Prefers the branch's archives (full MAP-Elites state, written by the
    Evolve workflow); otherwise parses the evolved clips of a published batch. Real-pop
    reference clips are never used as sources."""
    from evomusic.midi_io import genome_from_midi
    arch_dir = site / "archive"
    archives = sorted(arch_dir.glob("*.json"), key=lambda p: _batch_key(p.stem)) if arch_dir.exists() else []
    if batch and (arch_dir / f"{batch}.json").exists():
        archives = [arch_dir / f"{batch}.json"]
    if archives and (batch is None or archives[-1].stem == batch):
        out = []
        for e in json.loads(archives[-1].read_text())["saved"]:
            if "form" in e:
                out.append(FormGenome.from_json(e["form"]))
            else:
                out.append(from_genome(Genome([tuple(t) for t in e["melody"]], e["chord_ids"], 8))[0])
        return out, f"archive {archives[-1].stem} ({len(out)} pieces)"
    data = json.loads((site / "batches.json").read_text())
    batches = data["batches"] if batch is None else [b for b in data["batches"] if b["id"] == batch]
    if not batches:
        raise SystemExit(f"batch {batch!r} not found")
    b = batches[-1]
    clips = [c for c in b["clips"] if c["label"] != "reference"]
    out = [from_genome(genome_from_midi(site / c["mid"]))[0] for c in clips]
    return out, f"batch {b['id']} ({len(out)} clips)"


def quality_fitness(mode: str):
    """Light quality floor: rules + corpus bands (+ n-grams, + small critic in critic mode).
    The knob dimensions are left out of the bands so they don't fight the knobs."""
    from evomusic.fitness.factory import build
    models = ROOT / "models"
    excl = ("parallel_rhythm", "parallel_contour", "distinct_chords", "density_ratio", "chord_changes")
    ngram = ROOT / "data/processed/ngram_hooktheory.json"
    if mode == "critic":
        return build(0.5, ngram, 0.5, models / "critic_lm.pt", 1.0,
                     [models / f"critic_disc_r{k}.pt" for k in (7, 8, 9)], 2.0,
                     ROOT / "data/processed/desc_bands.json", 2.5, desc_exclude=excl)
    return build(0.5, ngram, 0.5, desc=ROOT / "data/processed/desc_bands.json", desc_w=2.5, desc_exclude=excl)


def produce(sources: list[FormGenome], n: int, strength: int, targets: dict[str, float], keep: set[str],
            mode: str, seed: int, pool_size: int | None = None, log=print) -> list[dict]:
    rng = random.Random(seed)
    stats = load_stats()
    rate, p_cross, boost = STRENGTH[strength]
    weights = mutation_weights(keep, boost)
    pool_size = pool_size or max(2000, 250 * n)
    knobs, fitness = Knobs(), quality_fitness(mode)
    t0 = time.time()
    cands, parents = [], []
    seen = {(tuple(s.to_genome().melody), tuple(s.to_genome().chords)) for s in sources}   # never return a source unchanged
    for _ in range(pool_size * 3):
        if len(cands) >= pool_size:
            break
        i = rng.randrange(len(sources))
        x = sources[i]
        if len(sources) > 1 and "plan" not in keep and rng.random() < p_cross:
            x = crossover_form(x, rng.choice(sources), rng)
            if "hook" in keep:                        # crossover may replace bar 1's motif: restore it
                x = x.copy(); x.motifs[x.plan[0].motif] = list(sources[i].motifs[sources[i].plan[0].motif]); x.gc()
            if "harmony" in keep:
                x = x.copy(); x.chords = list(sources[i].chords); x.gc()
        x = mutate_form(x, rng, rate, stats, weights=weights, keep_hook="hook" in keep)
        g = x.to_genome()
        key = (tuple(g.melody), tuple(g.chords))
        if key in seen:
            continue
        seen.add(key)
        cands.append(x); parents.append(i)
    log(f"{len(cands)} candidate variations from {len(sources)} pieces in {time.time() - t0:.0f}s (strength {strength}, keep {sorted(keep) or 'nothing'})")

    gs = [c.to_genome() for c in cands]
    quality = fitness.score_all(gs)
    pcts = [knobs.percentiles(g) for g in gs]
    match = [Knobs.match(p, targets) for p in pcts]
    w_knob = 0.6 if targets else 0.0
    score = [(1 - w_knob) * q + w_knob * m for q, m in zip(quality, match)]
    log(f"scored in {time.time() - t0:.0f}s; quality median {np.median(quality):.3f}; knob match median {np.median(match):.3f}")

    # quality floor: only the better half of the candidates may be chosen, however well they match the knobs
    floor = float(np.median(quality))
    eligible = [j for j in range(len(cands)) if quality[j] >= floor]
    log(f"quality floor {floor:.3f}: {len(eligible)} eligible")
    # diverse selection (maximal marginal relevance in knob-percentile space)
    vec = np.array([[p[k] / 100 for k in KNOBS] for p in pcts])
    chosen: list[int] = []
    order = sorted(eligible, key=lambda j: -score[j])[:max(50 * n, 400)]
    while len(chosen) < min(n, len(order)):
        best, best_v = None, -1e9
        for j in order:
            if j in chosen:
                continue
            sim = max((1 - np.abs(vec[j] - vec[c]).mean() for c in chosen), default=0.0)
            v = score[j] - 0.5 * sim
            if v > best_v:
                best, best_v = j, v
        chosen.append(int(best))
    out = []
    for rank, j in enumerate(chosen):
        out.append({"form": cands[j], "genome": gs[j], "parent": parents[j], "quality": quality[j],
                    "knob_match": match[j], "score": score[j], "percentiles": pcts[j], "rank": rank})
    return out


def table(saved: list[dict]) -> str:
    """Markdown table of produced pieces (knob columns are real-pop percentiles)."""
    cols = list(KNOBS)
    amt = any("amt_gap" in e for e in saved)
    head = "| piece | quality | knob match | " + " | ".join(KNOBS[k][0] for k in cols) + (" | AMT gap" if amt else "") + " | phrase plan |"
    rows = [head, "|---|---|---|" + "---|" * len(cols) + ("---|" if amt else "") + "---|"]
    for e in saved:
        rows.append(f"| {e['file'][:-4]} | {e['quality']:.2f} | {e['knob_match']:.2f} | " + " | ".join(str(e['knobs'][k]) for k in cols)
                    + (f" | {e.get('amt_gap', '')}" if amt else "") + f" | {e['plan']} |")
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", type=Path, default=ROOT / "docs", help="a branch's docs/ folder")
    ap.add_argument("--from", dest="from_summary", type=Path, help="explore a local summary.json instead of a site folder")
    ap.add_argument("--batch", help="batch id to explore (default: the latest archive / batch)")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--strength", type=int, choices=sorted(STRENGTH), default=3)
    ap.add_argument("--knobs", default="{}", help='JSON, e.g. {"complexity": 80, "predictability": 30}; omitted knobs are free')
    ap.add_argument("--keep", default="", help="comma list of hook, plan, harmony")
    ap.add_argument("--mode", choices=["corpus", "critic"], default="corpus")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool", type=int, help="number of candidate variations (default max(2000, 250 n))")
    ap.add_argument("--out", type=Path, default=Path("out/produced"))
    a = ap.parse_args(argv)

    targets = {k: float(v) for k, v in json.loads(a.knobs).items() if k in KNOBS and v is not None}
    keep = {k for k in a.keep.split(",") if k in ("hook", "plan", "harmony")}
    if a.from_summary:
        es = json.loads(a.from_summary.read_text())["saved"]
        sources = [FormGenome.from_json(e["form"]) if "form" in e else
                   from_genome(Genome([tuple(t) for t in e["melody"]], e["chord_ids"], 8))[0] for e in es]
        src = f"{a.from_summary} ({len(sources)} pieces)"
    else:
        sources, src = load_sources(a.site, a.batch)
    print(f"source: {src}; knobs {targets or 'free'}", flush=True)
    res = produce(sources, a.n, a.strength, targets, keep, a.mode, a.seed, a.pool, log=lambda *m: print(*m, flush=True))

    from evomusic.complexity import descriptors
    a.out.mkdir(parents=True, exist_ok=True)
    saved = []
    for r in res:
        g = r["genome"]
        name = f"produced-{r['rank'] + 1:02d}.mid"
        g.to_midi().save(a.out / name)
        saved.append({"file": name, "fitness": r["score"], "quality": round(r["quality"], 3), "knob_match": round(r["knob_match"], 3),
                      "knobs": {k: round(v) for k, v in r["percentiles"].items()}, "parent": r["parent"],
                      "chords": g.chord_names(), "plan": " ".join(from_genome(g)[1]),
                      "descriptors": {k: round(v, 3) for k, v in descriptors(g).items()},
                      "melody": g.melody, "chord_ids": g.chords, "form": r["form"].to_json()})
    req = {"source": src, "n": a.n, "strength": a.strength, "knobs": targets, "keep": sorted(keep), "mode": a.mode, "seed": a.seed}
    (a.out / "summary.json").write_text(json.dumps({"request": req, "star": saved[0]["file"], "saved": saved}, indent=1, default=list))
    (a.out / "chosen.txt").write_text("\n".join(str(a.out / e["file"]) for e in saved))
    t = table(saved)
    (a.out / "table.md").write_text(t + "\n")
    print("\n" + t)


if __name__ == "__main__":
    main()
