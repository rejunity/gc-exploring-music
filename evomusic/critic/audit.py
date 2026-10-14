"""Judge audit: can a corruption of real music score higher than real music?

Takes held-out real phrases, applies controlled corruptions (more repetition,
less repetition, thinning, shuffling, phrase swaps, pitch noise, random
notes), and reports every fitness term's mean score per variant, plus the
combined score. A judge FAILS a variant if the corruption scores at least as
high as the unmodified real phrases: that judge has a shortcut a genetic
algorithm will find. Run it before trusting any critic or fitness change.

    uv run python -m evomusic.critic.audit --lm models/critic_lm.pt --disc models/critic_disc_r9.pt \
        --desc data/processed/desc_bands.json --ngram data/processed/ngram_hooktheory.json [--amt-w 3] [--n 60]
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
from pathlib import Path

from evomusic.form import FormGenome, Slot, from_genome
from evomusic.genome import Genome

ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------- corruptions

def derepeat(g: Genome, pool: list[Genome], rng: random.Random) -> Genome:
    """Fewer repeats: every bar that literally repeats an earlier bar is replaced by the
    same-position bar of another real song."""
    toks = from_genome(g)[1]
    b = g.bar_genes()
    for i, t in enumerate(toks):
        if t.endswith("="):
            b[i] = rng.choice(pool).bar_genes()[i]
    return Genome.from_bars(b, g.chords).canonical()


def period_copy(g: Genome, pool, rng) -> Genome:
    b = g.bar_genes()
    return Genome.from_bars(b[:4] + b[:4], g.chords[:8] + g.chords[:8]).canonical()


def loop2(g: Genome, pool, rng) -> Genome:
    b = g.bar_genes()
    return Genome.from_bars(b[:2] * 4, g.chords[:4] * 4).canonical()


def loop1(g: Genome, pool, rng) -> Genome:
    b = g.bar_genes()
    return Genome.from_bars([b[0]] * 8, g.chords[:2] * 8).canonical()


def seq_spam(g: Genome, pool, rng) -> Genome:
    """Bar 1's motif in every bar, a step up or down each time (a a~ a~ ...)."""
    fg = from_genome(g)[0]
    a0 = fg.plan[0].anchor
    plan = [Slot(fg.plan[0].motif, "copy", a0 + k) for k in (0, 1, 2, 1, 0, -1, -2, -1)]
    return FormGenome(fg.motifs, plan, list(g.chords)).gc().to_genome()


def dup_bars(g: Genome, pool, rng) -> Genome:
    b = g.bar_genes()
    for _ in range(3):
        i, j = rng.sample(range(8), 2)
        b[j] = list(b[i])
    return Genome.from_bars(b, g.chords).canonical()


def thin(g: Genome, pool, rng) -> Genome:
    from evomusic.critic.train_disc import _thin
    return _thin(g, rng).canonical()


def bar_shuffle(g: Genome, pool, rng) -> Genome:
    from evomusic.critic.train_disc import _bar_shuffle
    return _bar_shuffle(g, rng).canonical()


def half_swap(g: Genome, pool, rng) -> Genome:
    from evomusic.critic.train_disc import _half_swap
    return _half_swap(g, rng.choice(pool), rng).canonical()


def pitch_noise(g: Genome, pool, rng) -> Genome:
    from evomusic.critic.train_disc import _pitch_noise
    return _pitch_noise(g, rng).canonical()


def random_piece(g: Genome, pool, rng) -> Genome:
    return Genome.random(in_scale=True, rng=rng).canonical()


# name -> (function, kind). kind "more" = more repetition than real, "less" = less, "other" = other corruption
VARIANTS = {
    "real, as is": (None, "real"),
    "fewer repeats": (derepeat, "less"),
    "bars 5-8 copy bars 1-4": (period_copy, "more"),
    "2-bar loop x4": (loop2, "more"),
    "1-bar loop x8": (loop1, "more"),
    "motif a step up/down x8": (seq_spam, "more"),
    "3 bars duplicated": (dup_bars, "more"),
    "thinned (sparse)": (thin, "other"),
    "bars shuffled": (bar_shuffle, "other"),
    "2nd phrase from another song": (half_swap, "other"),
    "pitch noise": (pitch_noise, "other"),
    "random notes": (random_piece, "other"),
}


def exact_share(g: Genome) -> float:
    toks = from_genome(g)[1]
    return sum(t.endswith("=") for t in toks) / len(toks)


# ---------------------------------------------------------------------------- audit

def audit(fitness, real: list[Genome], pool: list[Genome], seed: int = 0, variants: dict | None = None) -> dict:
    rng = random.Random(seed)
    variants = variants or VARIANTS
    names = list(fitness.terms)
    rows, fails, warns = [], [], []
    base = None
    for label, (fn, kind) in variants.items():
        gs = [g.canonical() for g in real] if fn is None else [fn(g, pool, rng) for g in real]
        vals = fitness.term_values(gs)
        comb = fitness.combine(vals, len(gs))
        row = {"variant": label, "kind": kind, "exact_share": st.mean(map(exact_share, gs)),
               "terms": {n: st.mean(vals[n]) for n in names}, "combined": st.mean(comb)}
        if kind == "real":
            base = row
        elif kind in ("more", "other") and base is not None:
            for n in names:
                if row["terms"][n] > base["terms"][n] + 0.01:        # a judge that prefers the corruption
                    (fails if kind == "more" else warns).append((n, label))
            if row["combined"] >= base["combined"] - 0.01:          # the combination must always catch it
                fails.append(("COMBINED", label))
        rows.append(row)
    # natural repetition must not be penalised: real pieces binned by exact-repeat share
    bins = {"0 repeats": [], "1-2 repeats": [], "3+ repeats": []}
    for g in real:
        k = round(exact_share(g) * 8)
        bins["0 repeats" if k == 0 else "1-2 repeats" if k <= 2 else "3+ repeats"].append(g.canonical())
    natural = {}
    for k, gs in bins.items():
        if len(gs) >= 5:
            vals = fitness.term_values(gs)
            natural[k] = {"n": len(gs), "terms": {n: st.mean(vals[n]) for n in names}, "combined": st.mean(fitness.combine(vals, len(gs)))}
    return {"terms": names, "weights": {n: w for n, (w, _) in fitness.terms.items()}, "agg": fitness.agg,
            "rows": rows, "fails": fails, "warns": warns, "natural_repetition": natural}


def render(rep: dict) -> str:
    names = rep["terms"]
    short = lambda n: n.replace("critic_", "").replace("ngram_", "ng_").replace("descriptors", "bands")[:9]
    out = [f"{'variant (real phrases, modified)':32s} {'exact':>5s} " + " ".join(f"{short(n):>9s}" for n in names) + f"  {'combined':>8s}"]
    for r in rep["rows"]:
        flag = " " if r["kind"] in ("real", "less") else ("!" if any(l == r["variant"] for _, l in rep["fails"]) else "ok")
        out.append(f"{r['variant']:32s} {r['exact_share']:5.2f} " + " ".join(f"{r['terms'][n]:9.3f}" for n in names) + f"  {r['combined']:8.3f} {flag}")
    out.append(f"\nnatural repetition in real music (must stay flat):")
    for k, v in rep["natural_repetition"].items():
        out.append(f"  {k:12s} n={v['n']:3d}  " + " ".join(f"{short(n)} {v['terms'][n]:.3f}" for n in names) + f"  combined {v['combined']:.3f}")
    def group(pairs):
        by = {}
        for n, l in pairs:
            by.setdefault(n, []).append(l)
        return by
    if rep["fails"]:
        out.append("\nFAIL (a judge rewards extra repetition, or the combination does not catch a corruption):")
        for n, ls in group(rep["fails"]).items():
            out.append(f"  {n}: " + "; ".join(ls))
    else:
        out.append("\nPASS: no judge rewards extra repetition; the combination scores every corruption below real music")
    if rep["warns"]:
        out.append("note, single judges preferring a non-repetition corruption (acceptable if the combination catches it):")
        for n, ls in group(rep["warns"]).items():
            out.append(f"  {n}: " + "; ".join(ls))
    return "\n".join(out)


def main() -> None:
    from evomusic.critic.train_disc import real_windows
    from evomusic.fitness.factory import build
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rules", type=float, default=0.5)
    ap.add_argument("--ngram", type=Path)
    ap.add_argument("--lm", type=Path)
    ap.add_argument("--lm-w", type=float, default=1.0)
    ap.add_argument("--disc", type=Path, nargs="*")
    ap.add_argument("--disc-w", type=float, default=2.0)
    ap.add_argument("--desc", type=Path)
    ap.add_argument("--desc-w", type=float, default=2.5)
    ap.add_argument("--amt-w", type=float, default=0.0)
    ap.add_argument("--novel-only", action="store_true", help="likelihood judges score only non-repeated bars")
    ap.add_argument("--agg", choices=["geo", "sum"], default="geo")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    valid = [g for g in real_windows("VALID", 8)]
    rng = random.Random(a.seed)
    rng.shuffle(valid)
    real, pool = valid[:a.n], real_windows("TRAIN", 8)[:3000]
    excl = ("notes_per_bar", "symbol_entropy", "period_echo", "climax_pos", "distinct_motifs", "exact_share", "rhythm_share",
            "four_chord_share", "minor_share", "chord_changes", "parallel_rhythm", "parallel_contour", "distinct_chords")
    fitness = build(a.rules, a.ngram, 0.5, a.lm, a.lm_w, a.disc or None, a.disc_w, a.desc, a.desc_w, a.amt_w, desc_exclude=excl, agg=a.agg, novel_only=a.novel_only)
    rep = audit(fitness, real, pool, a.seed)
    print(f"judge audit: {a.n} held-out real phrases, aggregation = {a.agg}, weights {rep['weights']}\n")
    print(render(rep))
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
