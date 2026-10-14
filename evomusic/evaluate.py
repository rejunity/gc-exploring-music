""""On par with professional music?" report: compare evolved pieces with held-out
real Hooktheory windows on the critic and on complexity descriptors.

    uv run python -m evomusic.evaluate --disc models/critic_disc_r4.pt --eval-disc models/critic_disc_eval.pt \
        out/me_final/summary.json [more summary.json ...]

For each metric: real p10 / p50 / p90, evolved median, and the fraction of
evolved pieces inside the real p10..p90 band. "On par" = evolved medians inside
the real band on every metric and the critic cannot separate them (AUC ~0.5).
Use an *independent* evaluation discriminator (--eval-disc, trained with a
different seed / on different GA rounds) so we are not grading with the same
critic we optimised against.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path

from evomusic.complexity import descriptors
from evomusic.critic.model import Critic
from evomusic.critic.train_disc import auc, real_windows
from evomusic.genome import Genome

ROOT = Path(__file__).resolve().parent.parent


def load_evolved(paths: list[Path]) -> list[Genome]:
    out = []
    for p in paths:
        s = json.loads(p.read_text())
        for c in s["saved"]:
            if "melody" in c:
                out.append(Genome([tuple(x) for x in c["melody"]], c["chord_ids"], 8, {"file": c["file"], "src": str(p)}))
    return out


def report(real: list[Genome], evolved: list[Genome], critics: dict[str, Critic]) -> list[dict]:
    rows = []
    for name, c in critics.items():
        pr, pe = c.p_real(real), c.p_real(evolved)
        rows.append(_row(f"p_real[{name}]", pr, pe) | {"auc_real_vs_evolved": auc(pr, pe)})
    lm = next((c for c in critics.values() if c.lm), None)
    if lm:
        rows.append(_row("lm_nll", lm.nll(real), lm.nll(evolved)))
    from evomusic.fitness.descriptors import form, phrase_parallel
    ext = lambda g: descriptors(g) | form(g) | {"parallel_rhythm": phrase_parallel(g)[0]}
    dr, de = [ext(g) for g in real], [ext(g) for g in evolved]
    for k in dr[0]:
        rows.append(_row(k, [d[k] for d in dr], [d[k] for d in de]))
    return rows


def _row(name, real, ev):
    real = [x for x in real if not (isinstance(x, float) and math.isnan(x))]
    ev = [x for x in ev if not (isinstance(x, float) and math.isnan(x))]
    q = st.quantiles(real, n=10)
    lo, hi = q[0], q[8]
    return {"metric": name, "real_p10": lo, "real_p50": st.median(real), "real_p90": hi,
            "evolved_p50": st.median(ev) if ev else float("nan"),
            "in_band": sum(lo <= x <= hi for x in ev) / max(1, len(ev))}


DENSITY_BUCKETS = [(0, 2.75), (2.75, 3.5), (3.5, 5), (5, 7), (7, 99)]


def amt_rows(real: list[Genome], evolved: list[Genome], amt) -> list[dict]:
    """AMT NLL overall, plus the density-matched gap: AMT finds sparse music less likely
    even when it is real, so each evolved piece is compared with real pieces of similar density."""
    rn, en = amt.nll(real), amt.nll(evolved)
    dens = lambda g: len(g.pitches()) / g.bars
    gaps = []
    for g, x in zip(evolved, en):
        lo, hi = next(b for b in DENSITY_BUCKETS if b[0] <= dens(g) < b[1])
        ref = [y for h, y in zip(real, rn) if lo <= dens(h) < hi]
        if ref:
            gaps.append(x - st.median(ref))
    row = _row("amt_nll (independent, Lakh)", rn, en) | {"auc_real_vs_evolved": auc([-x for x in rn], [-x for x in en])}
    gap = {"metric": "amt_nll gap, density-matched", "real_p10": 0.0, "real_p50": 0.0, "real_p90": 0.0,
           "evolved_p50": st.median(gaps) if gaps else float("nan"), "in_band": float("nan")}
    return [row, gap]


def human_agreement(scorers: dict, ratings_dir: Path = ROOT / "data/ratings") -> dict[str, tuple[float, int]]:
    """Fraction of human A/B choices (ties skipped) where each scorer prefers the same clip.
    Scorers map list[Genome] -> list[float], higher = better."""
    from evomusic.midi_io import genome_from_midi
    from evomusic.ratings import load as load_ratings
    batches = {b["id"]: {c["id"]: c for c in b["clips"]} for b in json.loads((ROOT / "docs/batches.json").read_text())["batches"]}
    rs = [r for r in load_ratings(sorted(ratings_dir.glob("*.json"))) if r["choice"] in ("A", "B") and r.get("branch", "main") == "main"]
    if not rs:
        return {}
    clips = sorted({(r["batch"], r[k]) for r in rs for k in ("a", "b")})
    gs = [genome_from_midi(ROOT / "docs" / batches[b][c]["mid"]) for b, c in clips]
    out = {}
    for name, fn in scorers.items():
        sc = dict(zip(clips, fn(gs)))
        agree = sum((sc[(r["batch"], r["a"])] > sc[(r["batch"], r["b"])]) == (r["choice"] == "A") for r in rs)
        out[name] = (agree / len(rs), len(rs))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("summaries", type=Path, nargs="+")
    ap.add_argument("--lm", type=Path, default=ROOT / "models/critic_lm.pt")
    ap.add_argument("--disc", type=Path, nargs="*", help="the critic (ensemble) optimised against")
    ap.add_argument("--eval-disc", type=Path, help="independent critic for grading")
    ap.add_argument("--n-real", type=int, default=800)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--amt", action="store_true", help="add the Anticipatory Music Transformer as an independent grader")
    ap.add_argument("--human", action="store_true", help="report agreement of each critic with human A/B ratings")
    a = ap.parse_args()
    real = real_windows("TEST", 8)[:a.n_real]
    ev = load_evolved(a.summaries)
    critics = {}
    if a.disc:
        critics["train"] = Critic(a.lm, a.disc)
    if a.eval_disc:
        critics["eval"] = Critic(a.lm, a.eval_disc)
    rows = report(real, ev, critics)
    amt = None
    if a.amt:
        from evomusic.critic.amt import AMTGrader
        amt = AMTGrader()
        rows = rows[:len(critics)] + amt_rows(real, ev, amt) + rows[len(critics):]
    print(f"real test windows: {len(real)}   evolved pieces: {len(ev)}")
    print(f"{'metric':32s} {'real p10':>9s} {'p50':>8s} {'p90':>8s} | {'evolved p50':>11s} {'in band':>8s}  extra")
    for r in rows:
        extra = f"AUC {r['auc_real_vs_evolved']:.2f}" if "auc_real_vs_evolved" in r else ""
        if r["metric"].startswith("amt_nll gap"):
            print(f"{r['metric']:32s} evolved minus real median in same density bucket: {r['evolved_p50']:+.3f}"); continue
        print(f"{r['metric']:32s} {r['real_p10']:9.3f} {r['real_p50']:8.3f} {r['real_p90']:8.3f} | {r['evolved_p50']:11.3f} {r['in_band']:8.0%}  {extra}")
    if a.human:
        scorers = {f"p_real[{n}]": c.p_real for n, c in critics.items()}
        lmc = next((c for c in critics.values() if c.lm), None)
        if lmc:
            scorers["lm (neg nll)"] = lambda gs: [-x for x in lmc.nll(gs)]
        if amt:
            scorers["amt (neg nll)"] = lambda gs: [-x for x in amt.nll(gs)]
        ha = human_agreement(scorers)
        print("\nagreement with human A/B choices (0.5 = chance):")
        for k, (v, n) in ha.items():
            print(f"  {k:24s} {v:.2f}  (n={n})")
    if a.json:
        a.json.write_text(json.dumps(rows, indent=1))
    # write the independent grader's verdict back into each summary, for the site
    if a.eval_disc:
        pr = critics["eval"].p_real(ev)
        by_src: dict[str, dict] = {}
        for g, p in zip(ev, pr):
            by_src.setdefault(g.meta["src"], {})[g.meta["file"]] = p
        for src, m in by_src.items():
            s = json.loads(Path(src).read_text())
            for c in s["saved"]:
                if c["file"] in m:
                    c["p_real"] = m[c["file"]]
            Path(src).write_text(json.dumps(s, indent=1, default=list))


if __name__ == "__main__":
    main()
