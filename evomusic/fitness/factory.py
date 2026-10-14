"""Build the composite fitness from named components.

Default for "sound like professional pop": the learned critic dominates; the
hand-written rules are one aggregated sanity term with low weight.
"""
from __future__ import annotations

from pathlib import Path

from evomusic.fitness import Composite
from evomusic.fitness.rules import default_rules


def build(rules: float = 1.0, ngram: Path | None = None, ngram_w: float = 0.5,
          lm: Path | None = None, lm_w: float = 1.0, disc: Path | list[Path] | None = None, disc_w: float = 4.0,
          desc: Path | None = None, desc_w: float = 2.0, amt_w: float = 0.0, desc_exclude: tuple = (),
          form_w: float = 0.0, agg: str = "geo", novel_only: bool = False, amt_novel_only: bool = False) -> Composite:
    """novel_only: the LM scores only bars that are not exact repeats, so repetition cannot be used
    to look more typical. amt_novel_only does the same for the AMT; off by default because the
    audit showed it penalising real phrases whose repeats come late (only their early,
    context-poor bars get scored)."""
    f = Composite(agg=agg)
    if form_w:
        from evomusic.form import plan_term
        f.add("form_plan", form_w, plan_term())
    if amt_w:
        from evomusic.critic.amt import AMTGrader, AMTTerm
        from evomusic.critic.train_disc import real_windows
        f.add("amt", amt_w, AMTTerm(AMTGrader(), real_windows("VALID", 8)[:600], novel_only=amt_novel_only))
    if desc:
        from evomusic.fitness.descriptors import term
        f.add("descriptors", desc_w, term(desc, desc_exclude))
    if rules:
        r = default_rules()
        f.add("rules", rules, r)
    if ngram:
        from evomusic.fitness.ngram import STREAMS, CorpusModel
        m = CorpusModel.load(ngram)
        for name in STREAMS:
            f.add(f"ngram_{name}", ngram_w, m.term(name))
    if lm or disc:
        from evomusic.critic.model import Critic
        c = Critic(lm, disc)
        if lm:
            f.add("critic_lm", lm_w, c.lm_term(novel_only=novel_only))
        if disc:
            f.add("critic_disc", disc_w, c.disc_term())
    return f
