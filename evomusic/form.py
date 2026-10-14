"""Hook-and-form genome: a few short motifs + a phrase plan + transformations.

A piece is no longer a flat list of notes. It is built from:

  motifs : 1-bar melodic cells (rhythm + scale-degree contour), e.g. a hook,
           an answer, a contrast. Genes are (rel_degree | None for rest,
           chroma 0/1, duration in 16ths), relative to an anchor.
  plan   : 8 slots, one per bar: which motif, at which anchor (scale degree,
           0 = C4), and how it is transformed:
             copy    exact repeat (same anchor) or sequence (other anchor)
             vary    same beginning, new ending (param: split step, tail genes)
             rhythm  same rhythm, new contour (param: degrees per note)
             frag    first half of the motif twice, second time shifted
                     (param: shift) - the "fragmentation" of a sentence
  chords : 16 half-bar chord genes, as in Genome.

`from_genome` parses any melody (real or evolved) into this form losslessly,
which gives (1) a way to mine how real pop builds its phrases, (2) a way to
seed evolution from flat genomes, and (3) structure descriptors such as the
number of distinct motifs. `to_genome` realises the form as a Genome, so every
fitness term, critic and renderer works unchanged.

    uv run python -m evomusic.form mine      # phrase-plan statistics of real pop -> data/form_stats.json
    uv run python -m evomusic.form demo      # a few random forms, parsed real examples
    uv run python -m evomusic.form plan-model   # P(transformation | bar position) of real pop -> data/plan_model.json
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from evomusic.genome import CHORDS, DUR_PRIOR, MAJOR, STEPS_PER_BAR, Genome

ROOT = Path(__file__).resolve().parent.parent
STATS_PATH = ROOT / "data" / "form_stats.json"
P_LO, P_HI = 36, 96              # realised pitch range (critic token range)
A_LO, A_HI = -4, 14              # anchor range in scale degrees: F3 .. C6
OPS = ("copy", "vary", "rhythm", "frag")
LETTERS = "abcdefgh"


# ---------------------------------------------------------------------------- pitch helpers

def deg_to_pitch(d: int, chroma: int = 0) -> int:
    o, k = divmod(d, 7)
    return 60 + 12 * o + MAJOR[k] + chroma


def pitch_to_deg(p: int) -> tuple[int, int]:
    """Nearest diatonic degree at or below p (C major), plus chromatic raise 0/1."""
    o, pc = divmod(p - 60, 12)
    k = max(i for i in range(7) if MAJOR[i] <= pc)
    return 7 * o + k, pc - MAJOR[k]


def _steps(genes) -> int:
    return sum(g[2] for g in genes)


def truncate(genes, s: int) -> list:
    out, t = [], 0
    for r, c, d in genes:
        if t >= s:
            break
        out.append((r, c, min(d, s - t)))
        t += d
    return out


def genes_from(genes, s: int) -> list | None:
    """Genes starting at step s, or None if a gene crosses s."""
    t = 0
    for i, (_, _, d) in enumerate(genes):
        if t == s:
            return list(genes[i:])
        if t > s:
            return None
        t += d
    return [] if t == s else None


def shift(genes, k: int) -> list:
    return [(None if r is None else r + k, c, d) for r, c, d in genes]


def fit16(genes) -> list:
    """Force a gene list to exactly one bar: drop zero-length genes, truncate or pad."""
    genes = [g for g in genes if g[2] > 0]
    genes = truncate(genes, STEPS_PER_BAR)
    t = _steps(genes)
    if t < STEPS_PER_BAR:
        if genes:
            r, c, d = genes[-1]
            genes[-1] = (r, c, d + STEPS_PER_BAR - t)
        else:
            genes = [(None, 0, STEPS_PER_BAR)]
    return genes


# ---------------------------------------------------------------------------- genotype

@dataclass
class Slot:
    motif: int
    op: str = "copy"
    anchor: int = 4
    param: object = None

    def copy(self) -> "Slot":
        p = self.param
        if self.op == "vary":
            p = (p[0], list(p[1]))
        elif self.op == "rhythm":
            p = list(p)
        return Slot(self.motif, self.op, self.anchor, p)

    def to_json(self) -> list:
        return [self.motif, self.op, self.anchor, self.param]

    @classmethod
    def from_json(cls, x) -> "Slot":
        m, op, a, p = x
        if op == "vary":
            p = (int(p[0]), [tuple(g) for g in p[1]])
        elif op == "rhythm":
            p = [tuple(t) for t in p]
        return cls(int(m), op, int(a), p)


@dataclass
class FormGenome:
    motifs: list[list[tuple]]
    plan: list[Slot]
    chords: list[int]
    _pheno: Genome | None = field(default=None, repr=False, compare=False)

    # ---- phenotype ---------------------------------------------------------
    def bar(self, slot: Slot) -> list[tuple]:
        m = self.motifs[slot.motif]
        if slot.op == "vary":
            split, tail = slot.param
            genes = truncate(m, split) + list(tail)
        elif slot.op == "rhythm":
            degs = list(slot.param)
            notes = [g for g in m if g[0] is not None]
            if not degs:
                degs = [(g[0], g[1]) for g in notes]
            degs = (degs + [degs[-1]] * len(notes))[:len(notes)]
            it = iter(degs)
            genes = [g if g[0] is None else (*next(it), g[2]) for g in m]
        elif slot.op == "frag":
            half = truncate(m, STEPS_PER_BAR // 2)
            genes = half + shift(half, int(slot.param or 0))
        else:
            genes = m
        genes = fit16(genes)
        return [(None if r is None else max(P_LO, min(P_HI, deg_to_pitch(slot.anchor + r, c))), d) for r, c, d in genes]

    def to_genome(self) -> Genome:
        if self._pheno is None:
            g = Genome.from_bars([self.bar(s) for s in self.plan], list(self.chords))
            self._pheno = g.canonical()
        return self._pheno

    # ---- bookkeeping -------------------------------------------------------
    def copy(self) -> "FormGenome":
        return FormGenome([list(m) for m in self.motifs], [s.copy() for s in self.plan], list(self.chords))

    def gc(self) -> "FormGenome":
        """Drop unreferenced motifs, renumber in order of first use, repair invalid genes."""
        order = []
        for s in self.plan:
            if s.motif not in order:
                order.append(s.motif)
        remap = {old: new for new, old in enumerate(order)}
        self.motifs = [fit16(self.motifs[i]) for i in order]
        for s in self.plan:
            s.motif = remap[s.motif]
            s.anchor = max(A_LO, min(A_HI, s.anchor))
            if s.op == "vary":
                split = max(1, min(STEPS_PER_BAR - 1, int(s.param[0])))
                tail = fit16([(None, 0, split)] + list(s.param[1]))[1:] if s.param[1] else []
                s.param = (split, [g for g in tail if g[2] > 0] or [(0, 0, STEPS_PER_BAR - split)])
        self._pheno = None
        return self

    def tokens(self) -> list[str]:
        """Plan tokens as used for corpus statistics (from the realised melody)."""
        return from_genome(self.to_genome())[1]

    def to_json(self) -> dict:
        return {"motifs": self.motifs, "plan": [s.to_json() for s in self.plan], "chords": self.chords}

    @classmethod
    def from_json(cls, d: dict) -> "FormGenome":
        return cls([[tuple(g) for g in m] for m in d["motifs"]], [Slot.from_json(s) for s in d["plan"]], list(d["chords"]))


# ---------------------------------------------------------------------------- parse (Genome -> form)

def _rel_bar(bar, prev_anchor: int) -> tuple[int, list]:
    notes = [p for p, _ in bar if p is not None]
    anchor = pitch_to_deg(notes[0])[0] if notes else prev_anchor
    rel = []
    for p, d in bar:
        if p is None:
            rel.append((None, 0, d))
        else:
            dg, ch = pitch_to_deg(p)
            rel.append((dg - anchor, ch, d))
    return anchor, rel


def _relate(rel, anchor: int, m, mi: int, a0: int):
    """Best transformation of motif m that reproduces bar `rel` exactly: (priority, Slot, token) or None."""
    L = LETTERS[mi] if mi < len(LETTERS) else "z"
    if rel == m:
        return 5, Slot(mi, "copy", anchor), L + ("=" if anchor == a0 else "~")
    half_m = truncate(m, 8)
    if truncate(rel, 8) == half_m:
        second = genes_from(rel, 8)
        if second is not None and len(second) == len(half_m) and second:
            ks = {r2 - r1 for (r1, c1, d1), (r2, c2, d2) in zip(half_m, second) if r1 is not None and r2 is not None}
            same_shape = all((r1 is None) == (r2 is None) and c1 == c2 and d1 == d2
                             for (r1, c1, d1), (r2, c2, d2) in zip(half_m, second))
            if same_shape and len(ks) <= 1:
                k = ks.pop() if ks else 0
                return 4, Slot(mi, "frag", anchor, k), L + "f"
    t, bounds = 0, []
    for _, _, d in rel:
        t += d
        bounds.append(t)
    for s in sorted((b for b in bounds if 8 <= b < STEPS_PER_BAR), reverse=True):
        if truncate(rel, s) == truncate(m, s):
            return 3, Slot(mi, "vary", anchor, (s, genes_from(rel, s))), L + "'"
    if [(g[0] is None, g[2]) for g in rel] == [(g[0] is None, g[2]) for g in m] and any(g[0] is not None for g in m):
        return 2, Slot(mi, "rhythm", anchor, [(g[0], g[1]) for g in rel if g[0] is not None]), L + "*"
    return None


def from_genome(g: Genome) -> tuple[FormGenome, list[str]]:
    """Parse a melody into motifs + plan (lossless: from_genome(g)[0].to_genome() == g.canonical())."""
    g = g.canonical()
    motifs, first_anchor, plan, tokens = [], [], [], []
    prev = 4
    for bar in g.bar_genes():
        anchor, rel = _rel_bar(bar, prev)
        best = None
        for mi, m in enumerate(motifs):
            cand = _relate(rel, anchor, m, mi, first_anchor[mi])
            if cand and (best is None or cand[0] > best[0]):
                best = cand
        if best:
            plan.append(best[1]); tokens.append(best[2])
        else:
            motifs.append(rel); first_anchor.append(anchor)
            mi = len(motifs) - 1
            plan.append(Slot(mi, "copy", anchor)); tokens.append(LETTERS[mi] if mi < len(LETTERS) else "z")
        prev = anchor
    return FormGenome(motifs, plan, list(g.chords)), tokens


# ---------------------------------------------------------------------------- corpus statistics

def _sample(rng: random.Random, pairs, default=0):
    if not pairs:
        return default
    vals, ws = zip(*pairs)
    return rng.choices(vals, weights=ws)[0]


def mine(out: Path = STATS_PATH) -> dict:
    from evomusic.corpus.stats import load
    from evomusic.critic.tokens import windows
    gs = [g for g in load(ROOT / "data/processed/hooktheory.jsonl") if g.meta["split"] == "TRAIN"]
    real = [w for g in gs for w in windows(g, 8, 8) if w.bars == 8]          # phrase-aligned 8-bar windows
    plans, ops, seq, frag, split, nmot, first, new_step, dsteps = (Counter() for _ in range(9))
    lead_rest = bars_n = rests = genes_n = chroma = notes_n = halves_eq = exact = 0
    for g in real:
        fg, toks = from_genome(g)
        exact += fg.to_genome().melody == g.canonical().melody
        plans[" ".join(toks)] += 1
        nmot[len(fg.motifs)] += 1
        first[fg.plan[0].anchor] += 1
        halves_eq += g.chords[:8] == g.chords[8:]
        prev_anchor = None
        for s, t in zip(fg.plan, toks):
            op = t[1:] or "new"
            ops[op] += 1
            if op == "~":
                seq[s.anchor - fg.plan[[x.motif for x in fg.plan].index(s.motif)].anchor] += 1
            elif op == "f":
                frag[s.param] += 1
            elif op == "'":
                split[s.param[0]] += 1
            elif op == "new" and prev_anchor is not None:
                new_step[s.anchor - prev_anchor] += 1
            prev_anchor = s.anchor
        for m in fg.motifs:
            bars_n += 1
            lead_rest += m[0][0] is None
            ns = [x for x in m if x[0] is not None]
            genes_n += len(m); rests += len(m) - len(ns)
            notes_n += len(ns); chroma += sum(x[1] for x in ns)
            dsteps.update(b[0] - a[0] for a, b in zip(ns, ns[1:]))
    clip = lambda c, lo, hi: [[k, v] for k, v in sorted(c.items()) if lo <= k <= hi]
    stats = {
        "n_windows": len(real), "lossless_parse": exact / len(real),
        "plans": [[p, c] for p, c in plans.most_common(500)],
        "plan_mass_top500": sum(c for _, c in plans.most_common(500)) / len(real),
        "ops": dict(ops), "n_motifs": dict(sorted(nmot.items())),
        "seq_shift": clip(seq, -7, 7), "frag_shift": clip(frag, -7, 7), "vary_split": clip(split, 8, 15),
        "new_anchor_step": clip(new_step, -7, 7), "first_anchor": clip(first, A_LO, A_HI), "deg_steps": clip(dsteps, -7, 7),
        "lead_rest_p": lead_rest / bars_n, "rest_p": rests / genes_n, "chroma_p": chroma / max(1, notes_n),
        "chord_halves_equal_p": halves_eq / len(real),
    }
    out.write_text(json.dumps(stats, indent=1))
    return stats


_STATS: dict | None = None


def load_stats() -> dict:
    global _STATS
    if _STATS is None:
        _STATS = json.loads(STATS_PATH.read_text())
    return _STATS


# ---------------------------------------------------------------------------- form likelihood

PLAN_OPS_ORDER = ("new", "=", "~", "'", "*", "f")
PLAN_MODEL_PATH = ROOT / "data" / "plan_model.json"


def _op(token: str) -> str:
    return token[1:] or "new"


def fit_plan_model(out: Path = PLAN_MODEL_PATH) -> dict:
    """P(op | bar position) from real pop (TRAIN), plus the band of real pieces' plan log-likelihood (VALID)."""
    import math
    import statistics as st
    from evomusic.corpus.stats import load
    from evomusic.critic.tokens import windows
    gs = load(ROOT / "data/processed/hooktheory.jsonl")
    win = lambda split: [w for g in gs if g.meta["split"] == split for w in windows(g, 8, 8) if w.bars == 8]
    counts = [[1.0] * len(PLAN_OPS_ORDER) for _ in range(8)]          # add-one smoothing
    for g in win("TRAIN"):
        for i, t in enumerate(from_genome(g)[1]):
            counts[i][PLAN_OPS_ORDER.index(_op(t))] += 1
    logp = [[math.log(c / sum(row)) for c in row] for row in counts]
    model = {"ops": PLAN_OPS_ORDER, "logp": logp}
    ll = [plan_loglik(from_genome(g)[1], model) for g in win("VALID")]
    q = st.quantiles(ll, n=20)
    model["quantiles"] = (q[0], q[4], q[14], q[18])                    # p5, p25, p75, p95
    out.write_text(json.dumps(model, indent=1))
    return model


def plan_loglik(tokens: list[str], model: dict) -> float:
    return sum(model["logp"][i][PLAN_OPS_ORDER.index(_op(t))] for i, t in enumerate(tokens)) / len(tokens)


def plan_term(path: Path = PLAN_MODEL_PATH):
    """Fitness term: 1 when the piece's bar-by-bar form is as likely as typical real pop
    (real p25..p75), decaying to 0 at p5 / p95. Penalises one-trick forms (everything a
    repeat or a sequence) as well as forms that never repeat anything."""
    model = json.loads(Path(path).read_text())
    q5, q25, q75, q95 = model["quantiles"]

    def f(g: Genome) -> float:
        x = plan_loglik(from_genome(g)[1], model)
        if q25 <= x <= q75:
            return 1.0
        if x < q25:
            return max(0.0, (x - q5) / (q25 - q5))
        return max(0.0, (q95 - x) / (q95 - q75))
    f.__name__ = "form_plan"
    return f


# ---------------------------------------------------------------------------- random forms

def random_genes(rng: random.Random, steps: int, stats: dict, start: int = 0, lead_rest: bool = True) -> list:
    """Weakly constrained random notes: durations, rests, chromatic notes and
    melodic steps drawn from corpus frequencies (no musical rules)."""
    genes, t, r = [], 0, start
    while t < steps:
        d = min(rng.choices(list(DUR_PRIOR), weights=list(DUR_PRIOR.values()))[0], steps - t)
        first = not genes
        if (first and lead_rest and rng.random() < stats["lead_rest_p"]) or \
           (not first and genes[-1][0] is not None and rng.random() < stats["rest_p"]):
            genes.append((None, 0, d))
        else:
            if any(x[0] is not None for x in genes):
                r += _sample(rng, stats["deg_steps"])
            r = max(-9, min(9, r))
            genes.append((r, 1 if rng.random() < stats["chroma_p"] else 0, d))
        t += d
    if all(x[0] is None for x in genes):
        genes[-1] = (start, 0, genes[-1][2])
    return genes


def random_plan(rng: random.Random, stats: dict) -> list[str]:
    """A plan token list: usually a real-pop plan sampled by frequency, sometimes a random one."""
    if rng.random() < 0.85:
        return _sample(rng, stats["plans"]).split()
    ops = [(k, v) for k, v in stats["ops"].items() if k != "new"]
    p_new = stats["ops"]["new"] / sum(stats["ops"].values())
    toks, n = ["a"], 1
    for _ in range(7):
        if rng.random() < p_new and n < len(LETTERS):
            toks.append(LETTERS[n]); n += 1
        else:
            toks.append(rng.choice(LETTERS[:n]) + _sample(rng, ops))
    return toks


def instantiate(tokens: list[str], rng: random.Random, stats: dict, motifs: list | None = None,
                chords: list[int] | None = None) -> FormGenome:
    """Turn plan tokens into a FormGenome. Letters reuse `motifs` in order when given, else new random motifs."""
    motifs = [list(m) for m in (motifs or [])]
    letter_to, base, plan = {}, {}, []
    prev = _sample(rng, stats["first_anchor"])
    for i, tok in enumerate(tokens):
        L, op = tok[0], tok[1:]
        if L not in letter_to:
            k = len(letter_to)
            if k >= len(motifs):
                motifs.append(random_genes(rng, STEPS_PER_BAR, stats))
            letter_to[L] = k
            anchor = prev if i == 0 else max(A_LO, min(A_HI, prev + _sample(rng, stats["new_anchor_step"])))
            base[L] = anchor
            plan.append(Slot(k, "copy", anchor))
        else:
            mi, a0 = letter_to[L], base[L]
            if op == "~":
                plan.append(Slot(mi, "copy", max(A_LO, min(A_HI, a0 + (_sample(rng, stats["seq_shift"]) or 1)))))
            elif op == "'":
                s = _sample(rng, stats["vary_split"])
                plan.append(Slot(mi, "vary", a0, (s, random_genes(rng, STEPS_PER_BAR - s, stats, start=0, lead_rest=False))))
            elif op == "*":
                n = sum(g[0] is not None for g in motifs[mi]) or 1
                walk = random_genes(rng, 64, stats, lead_rest=False)
                degs = [(g[0], g[1]) for g in walk if g[0] is not None][:n]
                plan.append(Slot(mi, "rhythm", a0, degs))
            elif op == "f":
                plan.append(Slot(mi, "frag", a0, _sample(rng, stats["frag_shift"])))
            else:
                plan.append(Slot(mi, "copy", a0))
        prev = plan[-1].anchor
    if chords is None:
        pool = 7 if rng.random() < 0.5 else len(CHORDS)
        chords = [rng.randrange(pool) for _ in range(16)]
        if rng.random() < stats["chord_halves_equal_p"]:
            chords[8:] = chords[:8]
    return FormGenome(motifs, plan, list(chords)).gc()


def random_form(rng: random.Random, stats: dict | None = None) -> FormGenome:
    stats = stats or load_stats()
    return instantiate(random_plan(rng, stats), rng, stats)


# ---------------------------------------------------------------------------- variation operators

def _motif_op(x: FormGenome, rng: random.Random, op: str, stats: dict, frozen: frozenset = frozenset()) -> None:
    free = [i for i in range(len(x.motifs)) if i not in frozen]
    if not free:
        return
    mi = rng.choice(free)
    m = list(x.motifs[mi])
    notes = [i for i, g in enumerate(m) if g[0] is not None]
    if op == "m_pitch" and notes:
        i = rng.choice(notes); r, c, d = m[i]
        m[i] = (r + rng.choice([-2, -1, 1, 2]), c if rng.random() > 0.05 else 1 - c, d)
    elif op == "m_split":
        cand = [i for i, g in enumerate(m) if g[2] >= 2]
        if cand:
            i = rng.choice(cand); r, c, d = m[i]; a = rng.randint(1, d - 1)
            r2 = r if r is None or rng.random() < 0.3 else r + rng.choice([-1, 1, -2, 2])
            m[i:i + 1] = [(r, c, a), (r2, c, d - a)]
    elif op == "m_merge" and len(m) > 1:
        i = rng.randrange(len(m) - 1); (r1, c1, d1), (r2, c2, d2) = m[i], m[i + 1]
        m[i:i + 2] = [((r1, c1) if r1 is not None else (r2, c2)) + (d1 + d2,)]
    elif op == "m_nudge" and len(m) > 1:
        i = rng.randrange(len(m) - 1); k = rng.choice([-1, 1])
        (r1, c1, d1), (r2, c2, d2) = m[i], m[i + 1]
        if d1 + k >= 1 and d2 - k >= 1:
            m[i], m[i + 1] = (r1, c1, d1 + k), (r2, c2, d2 - k)
    elif op == "m_rest":
        i = rng.randrange(len(m)); r, c, d = m[i]
        if r is None:
            near = [g[0] for g in m if g[0] is not None]
            m[i] = ((rng.choice(near) if near else 0) + rng.choice([-1, 0, 1]), 0, d)
        elif len(notes) > 1:
            m[i] = (None, 0, d)
    elif op == "m_invert":
        m = [(None if r is None else -r, c, d) for r, c, d in m]
    elif op == "m_retro":
        m = list(reversed(m))
    elif op == "m_new":
        m = random_genes(rng, STEPS_PER_BAR, stats)
    x.motifs[mi] = fit16(m)


def _slot_op(x: FormGenome, rng: random.Random, op: str, stats: dict) -> None:
    i = rng.randrange(len(x.plan))
    s = x.plan[i]
    first_use = [t.motif for t in x.plan].index(s.motif) == i
    if op == "s_anchor":
        k = rng.choice([-2, -1, 1, 2])
        if s.op == "copy" and rng.random() < 0.85:
            # move the whole group of exact repeats together: exact repeats stay exact
            # (moving a single copy turns it into a sequence, which real pop uses for only 3 % of bars)
            for t in x.plan:
                if t.motif == s.motif and t.op == "copy" and t.anchor == s.anchor and t is not s:
                    t.anchor += k
        s.anchor += k
    elif op == "s_transpose_motif":                       # move every occurrence: keeps the relations
        k = rng.choice([-2, -1, 1, 2])
        for t in x.plan:
            if t.motif == s.motif:
                t.anchor += k
    elif op == "s_op" and not first_use:
        o = stats["ops"]
        freq = {"copy": o.get("=", 1) + o.get("~", 1), "vary": o.get("'", 1), "rhythm": o.get("*", 1), "frag": o.get("f", 1)}
        choices = [k for k in OPS if k != s.op]
        new = rng.choices(choices, weights=[freq[k] for k in choices])[0]
        if new == "vary":
            sp = _sample(rng, stats["vary_split"])
            s.op, s.param = "vary", (sp, random_genes(rng, STEPS_PER_BAR - sp, stats, lead_rest=False))
        elif new == "rhythm":
            n = sum(g[0] is not None for g in x.motifs[s.motif]) or 1
            walk = random_genes(rng, 64, stats, lead_rest=False)
            s.op, s.param = "rhythm", [(g[0], g[1]) for g in walk if g[0] is not None][:n]
        elif new == "frag":
            s.op, s.param = "frag", _sample(rng, stats["frag_shift"])
        else:
            s.op, s.param = "copy", None
    elif op == "s_ref" and i > 0:
        if rng.random() < 0.3:
            x.motifs.append(random_genes(rng, STEPS_PER_BAR, stats))
            s.motif = len(x.motifs) - 1
        else:
            s.motif = rng.randrange(len(x.motifs))
        s.op, s.param = "copy", None
    elif op == "s_tail" and s.op == "vary":
        sp, tail = s.param
        if rng.random() < 0.5 and tail:
            j = rng.randrange(len(tail)); r, c, d = tail[j]
            tail = list(tail); tail[j] = ((r if r is not None else 0) + rng.choice([-2, -1, 1, 2]), c, d)
        else:
            tail = random_genes(rng, STEPS_PER_BAR - sp, stats, lead_rest=False)
        s.param = (sp, tail)
    elif op == "s_degrees" and s.op == "rhythm" and s.param:
        degs = list(s.param); j = rng.randrange(len(degs)); r, c = degs[j]
        degs[j] = (r + rng.choice([-2, -1, 1, 2]), c); s.param = degs
    elif op == "s_frag" and s.op == "frag":
        s.param = int(s.param or 0) + rng.choice([-1, 1])


def _chord_op(x: FormGenome, rng: random.Random, op: str) -> None:
    ch = x.chords
    if op == "c_change":
        ch[rng.randrange(len(ch))] = rng.randrange(len(CHORDS))
    elif op == "c_copy":
        ch[rng.randrange(len(ch))] = ch[rng.randrange(len(ch))]
    elif op == "c_phrase":
        h = len(ch) // 2
        ch[h:] = ch[:h]
    elif op == "c_harmonize":                              # pick a chord that contains the melody's strong-beat note
        g = x.to_genome()
        k = rng.randrange(len(ch))
        notes = [p for s, p, _ in g.onsets() if p is not None and k * 8 <= s < k * 8 + 8 and s % 4 == 0]
        if notes:
            pc = notes[0] % 12
            fits = [i for i, c in enumerate(CHORDS) if pc in c.pitch_classes()]
            if fits:
                ch[k] = rng.choice(fits)


MUTATIONS = {
    "m_pitch": 3, "m_split": 1.5, "m_merge": 1.5, "m_nudge": 1, "m_rest": 0.7, "m_invert": 0.2, "m_retro": 0.2, "m_new": 0.4,
    "s_anchor": 1.2, "s_transpose_motif": 0.8, "s_op": 1.2, "s_ref": 0.8, "s_tail": 1, "s_degrees": 0.6, "s_frag": 0.3,
    "p_resample": 0.3, "p_swap": 0.3, "p_register": 0.2,
    "c_change": 2, "c_copy": 0.8, "c_phrase": 0.4, "c_harmonize": 1.2,
}


PLAN_OPS = {"s_op", "s_ref", "p_resample", "p_swap"}
CHORD_OPS = {"c_change", "c_copy", "c_phrase", "c_harmonize"}


def mutation_weights(keep: set[str] = frozenset(), boost_structure: float = 1.0) -> dict[str, float]:
    """Operator weights honouring keep = {"hook", "plan", "harmony"}; boost_structure > 1 favours
    big moves (new plans, new motifs, re-pointed slots) for wide exploration."""
    w = dict(MUTATIONS)
    for k in ("p_resample", "m_new", "s_ref", "s_op"):
        w[k] *= boost_structure
    if "plan" in keep:
        for k in PLAN_OPS:
            w[k] = 0.0
    if "harmony" in keep:
        for k in CHORD_OPS:
            w[k] = 0.0
    return w


def mutate_form(fg: FormGenome, rng: random.Random, rate: float = 1.0, stats: dict | None = None,
                weights: dict | None = None, keep_hook: bool = False) -> FormGenome:
    """`weights` overrides operator weights (see mutation_weights); keep_hook freezes the motif of bar 1."""
    stats = stats or load_stats()
    x = fg.copy()
    names, weights = zip(*(weights or MUTATIONS).items())
    for _ in range(max(1, round(rng.expovariate(1 / rate)))):
        op = rng.choices(names, weights=weights)[0]
        if op.startswith("m_"):
            _motif_op(x, rng, op, stats, frozenset({x.plan[0].motif}) if keep_hook else frozenset())
        elif op.startswith("s_"):
            _slot_op(x, rng, op, stats)
        elif op.startswith("c_"):
            _chord_op(x, rng, op)
        elif op == "p_resample":
            hook = x.plan[0].motif
            order = [hook] + [i for i in range(len(x.motifs)) if i != hook]     # letter "a" keeps the hook
            x = instantiate(random_plan(rng, stats), rng, stats, motifs=[x.motifs[i] for i in order], chords=x.chords)
        elif op == "p_swap":
            i, j = rng.sample(range(1, len(x.plan)), 2)
            x.plan[i], x.plan[j] = x.plan[j], x.plan[i]
        elif op == "p_register":
            k = rng.choice([-7, 7])
            if all(A_LO <= s.anchor + k <= A_HI for s in x.plan):
                for s in x.plan:
                    s.anchor += k
        x._pheno = None
    return x.gc()


def crossover_form(a: FormGenome, b: FormGenome, rng: random.Random) -> FormGenome:
    r = rng.random()
    if r < 0.4:          # structure from one parent, material (motifs) from the other
        motifs = [list(m) for m in b.motifs] + [list(m) for m in a.motifs[len(b.motifs):]]
        child = FormGenome(motifs, [s.copy() for s in a.plan], list(a.chords if rng.random() < 0.5 else b.chords))
    elif r < 0.8:        # question phrase from one parent, answer phrase from the other
        off = len(a.motifs)
        plan = [s.copy() for s in a.plan[:4]]
        for s in b.plan[4:]:
            t = s.copy(); t.motif += off; plan.append(t)
        child = FormGenome([list(m) for m in a.motifs + b.motifs], plan, list(a.chords[:8] + b.chords[8:]))
    else:                # borrow one motif
        motifs = [list(m) for m in a.motifs]
        motifs[rng.randrange(len(motifs))] = list(rng.choice(b.motifs))
        child = FormGenome(motifs, [s.copy() for s in a.plan], list(a.chords))
    return child.gc()


# ---------------------------------------------------------------------------- CLI

def main(argv: list[str]) -> None:
    cmd = argv[0] if argv else ""
    if cmd == "plan-model":
        m = fit_plan_model()
        print("P(op | bar) from real pop:")
        import math
        for i, row in enumerate(m["logp"]):
            print(f"  bar {i + 1}: " + "  ".join(f"{o} {math.exp(v):5.1%}" for o, v in zip(PLAN_OPS_ORDER, row)))
        print("real plan log-likelihood p5/p25/p75/p95:", [round(x, 3) for x in m["quantiles"]])
        return
    if cmd == "mine":
        s = mine()
        tot = s["n_windows"]
        print(f"{tot} phrase-aligned 8-bar windows; lossless parse {s['lossless_parse']:.1%}; top-500 plans cover {s['plan_mass_top500']:.1%}")
        print("op mix:", {k: f"{v / sum(s['ops'].values()):.1%}" for k, v in s["ops"].items()})
        print("distinct motifs per 8 bars:", s["n_motifs"])
        print("chords: second phrase repeats the first:", f"{s['chord_halves_equal_p']:.1%}")
        print("most common phrase plans (= exact repeat, ~ sequence, ' new ending, * same rhythm, f fragment):")
        for p, c in s["plans"][:15]:
            print(f"  {c / tot:6.2%}  {p}")
    elif cmd == "demo":
        rng = random.Random(int(argv[1]) if len(argv) > 1 else 0)
        for _ in range(3):
            fg = random_form(rng)
            print(" ".join(fg.tokens()), "|", fg.to_genome().chord_names())
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
