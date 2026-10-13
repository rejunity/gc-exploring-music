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
    dr, de = [descriptors(g) for g in real], [descriptors(g) for g in evolved]
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("summaries", type=Path, nargs="+")
    ap.add_argument("--lm", type=Path, default=ROOT / "models/critic_lm.pt")
    ap.add_argument("--disc", type=Path, help="the critic optimised against")
    ap.add_argument("--eval-disc", type=Path, help="independent critic for grading")
    ap.add_argument("--n-real", type=int, default=800)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    real = real_windows("TEST", 8)[:a.n_real]
    ev = load_evolved(a.summaries)
    critics = {}
    if a.disc:
        critics["train"] = Critic(a.lm, a.disc)
    if a.eval_disc:
        critics["eval"] = Critic(a.lm, a.eval_disc)
    rows = report(real, ev, critics)
    print(f"real test windows: {len(real)}   evolved pieces: {len(ev)}")
    print(f"{'metric':24s} {'real p10':>9s} {'p50':>8s} {'p90':>8s} | {'evolved p50':>11s} {'in band':>8s}  extra")
    for r in rows:
        extra = f"AUC {r['auc_real_vs_evolved']:.2f}" if "auc_real_vs_evolved" in r else ""
        print(f"{r['metric']:24s} {r['real_p10']:9.3f} {r['real_p50']:8.3f} {r['real_p90']:8.3f} | {r['evolved_p50']:11.3f} {r['in_band']:8.0%}  {extra}")
    if a.json:
        a.json.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
