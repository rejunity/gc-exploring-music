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
          desc: Path | None = None, desc_w: float = 2.0, amt_w: float = 0.0) -> Composite:
    f = Composite()
    if amt_w:
        from evomusic.critic.amt import AMTGrader, AMTTerm
        from evomusic.critic.train_disc import real_windows
        f.add("amt", amt_w, AMTTerm(AMTGrader(), real_windows("VALID", 8)[:600]))
    if desc:
        from evomusic.fitness.descriptors import term
        f.add("descriptors", desc_w, term(desc))
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
            f.add("critic_lm", lm_w, c.lm_term())
        if disc:
            f.add("critic_disc", disc_w, c.disc_term())
    return f
