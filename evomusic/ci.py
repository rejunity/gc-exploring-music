"""Helpers for the GitHub Actions evolution workflow (.github/workflows/evolve.yml).

    python -m evomusic.ci params          # parse + validate Evolve parameters -> $GITHUB_OUTPUT
    python -m evomusic.ci produce-params  # parse + validate Produce parameters -> $GITHUB_OUTPUT
    python -m evomusic.ci final-check RUN_DIR CHOSEN_TXT   # AMT once on the clips to publish -> summary.json
    python -m evomusic.ci set-download DOCS BATCH URL      # link the run's .mid + .mp3 artifact on the batch

Parameters come either from workflow_dispatch inputs (env INPUT_*) or from the
JSON block of an issue body (env ISSUE_BODY). Issue text is untrusted: it is
only ever parsed here, validated against strict patterns, and never pasted into
a shell command.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BRANCH_RE = re.compile(r"^exp/[a-z0-9][a-z0-9._-]{0,39}$")
MODES = ("corpus", "critic")
GENOMES = ("form", "notes")
DEFAULTS = {"branch": "exp/try", "iters": 200, "seed": 0, "mode": "corpus", "genome": "form", "continue": True, "title": ""}


def _from_issue(body: str) -> dict:
    m = re.search(r"```json\s*(\{.*?\})\s*```", body or "", re.S)
    if not m:
        raise ValueError("issue body has no ```json {...}``` block")
    return json.loads(m.group(1))


def validate(raw: dict) -> dict:
    p = dict(DEFAULTS)
    p.update({k: v for k, v in raw.items() if k in DEFAULTS and v not in (None, "")})
    p["branch"] = re.sub(r"\s+", "-", str(p["branch"]).strip().lower())
    if not p["branch"].startswith("exp/"):
        p["branch"] = "exp/" + p["branch"]
    if not BRANCH_RE.match(p["branch"]):
        raise ValueError(f"branch must match {BRANCH_RE.pattern}")
    p["iters"] = int(p["iters"])
    if not 10 <= p["iters"] <= 2000:
        raise ValueError("iters must be 10..2000")
    p["seed"] = int(p["seed"]) % 100000
    if p["mode"] not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if p["genome"] not in GENOMES:
        raise ValueError(f"genome must be one of {GENOMES}")
    if p["mode"] == "critic" and p["iters"] > 800:
        raise ValueError("critic mode on a CPU runner: iters must be <= 800 (~7.5 s per iteration); continue the branch for more")
    p["continue"] = str(p["continue"]).lower() in ("1", "true", "yes", "on")
    p["title"] = re.sub(r"[^\w .,:;!?()'+-]", "", str(p["title"]))[:80]
    return p


KNOB_NAMES = ("complexity", "predictability", "density", "repetition", "darkness", "harmonic_adventure", "energy_build")
SOURCE_RE = re.compile(r"^(main|exp/[a-z0-9][a-z0-9._-]{0,39})$")
BATCH_RE = re.compile(r"^[A-Za-z0-9._-]{0,40}$")


def validate_produce(raw: dict) -> dict:
    p = {"branch": "main", "batch": "", "n": 8, "strength": 3, "knobs": {}, "keep": [], "mode": "corpus", "seed": 0}
    p.update({k: v for k, v in raw.items() if k in p and v not in (None, "")})
    p["branch"] = re.sub(r"\s+", "-", str(p["branch"]).strip().lower())
    if p["branch"] != "main" and not p["branch"].startswith("exp/"):
        p["branch"] = "exp/" + p["branch"]
    if not SOURCE_RE.match(p["branch"]):
        raise ValueError("branch must be main or exp/<name>")
    p["batch"] = str(p["batch"]).strip()
    if not BATCH_RE.match(p["batch"]):
        raise ValueError("batch id may only contain letters, digits, . _ -")
    p["n"] = int(p["n"])
    if not 1 <= p["n"] <= 32:
        raise ValueError("n must be 1..32")
    p["strength"] = int(p["strength"])
    if not 1 <= p["strength"] <= 5:
        raise ValueError("strength must be 1..5")
    knobs = p["knobs"]
    if isinstance(knobs, str):
        knobs = json.loads(knobs or "{}")
    if not isinstance(knobs, dict):
        raise ValueError("knobs must be a JSON object")
    clean = {}
    for k, v in knobs.items():
        if k not in KNOB_NAMES:
            raise ValueError(f"unknown knob {k!r}; known: {KNOB_NAMES}")
        if v is not None:
            v = float(v)
            if not 0 <= v <= 100:
                raise ValueError(f"knob {k} must be 0..100 (a percentile of real pop)")
            clean[k] = round(v, 1)
    p["knobs"] = clean
    keep = p["keep"]
    if isinstance(keep, str):
        keep = [k.strip() for k in keep.split(",") if k.strip()]
    if any(k not in ("hook", "plan", "harmony") for k in keep):
        raise ValueError("keep may contain hook, plan, harmony")
    p["keep"] = ",".join(sorted(set(keep)))
    if p["mode"] not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    p["seed"] = int(p["seed"]) % 100000
    p["knobs"] = json.dumps(p["knobs"], separators=(",", ":"))
    return p


def produce_params() -> None:
    if os.environ.get("GITHUB_EVENT_NAME") == "issues":
        raw = _from_issue(os.environ.get("ISSUE_BODY", ""))
    else:
        raw = {k: os.environ.get(f"INPUT_{k.upper()}") for k in ("branch", "batch", "n", "strength", "knobs", "keep", "mode", "seed")}
    p = validate_produce(raw)
    out = os.environ.get("GITHUB_OUTPUT")
    lines = [f"{k}={v}" for k, v in p.items()]
    if out:
        with open(out, "a") as f:
            f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def params() -> None:
    if os.environ.get("GITHUB_EVENT_NAME") == "issues":
        raw = _from_issue(os.environ.get("ISSUE_BODY", ""))
    else:
        raw = {k: os.environ.get(f"INPUT_{k.upper()}") for k in DEFAULTS}
    p = validate(raw)
    out = os.environ.get("GITHUB_OUTPUT")
    lines = [f"{k}={json.dumps(v) if isinstance(v, bool) else v}" for k, v in p.items()]
    if out:
        with open(out, "a") as f:
            f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def final_check(run_dir: Path, chosen: Path) -> None:
    """Score only the clips about to be published with the large AMT model (CPU, once).
    gap = NLL minus the median NLL of real pop of similar density (precomputed on GPU)."""
    from evomusic.critic.amt import AMTGrader, DENSITY_BUCKETS
    from evomusic.genome import Genome
    med = json.loads((ROOT / "data/amt_density_medians.json").read_text())
    s = json.loads((run_dir / "summary.json").read_text())
    files = {Path(l).name for l in chosen.read_text().split()}
    entries = [c for c in s["saved"] if c["file"] in files]
    gs = [Genome([tuple(x) for x in c["melody"]], c["chord_ids"], 8) for c in entries]
    nll = AMTGrader().nll(gs, bs=2)
    for c, g, x in zip(entries, gs, nll):
        d = len(g.pitches()) / g.bars
        key = next(f"{lo}-{hi}" for lo, hi in DENSITY_BUCKETS if lo <= d < hi)
        c["amt_gap"] = round(x - med[key], 3)
    (run_dir / "summary.json").write_text(json.dumps(s, indent=1, default=list))
    if (run_dir / "table.md").exists() and all("knobs" in e for e in s["saved"]):     # Produce runs: add the AMT column
        from evomusic.produce import table
        (run_dir / "table.md").write_text(table(s["saved"]) + "\n")
    print("AMT gap per clip (negative = more natural than typical real pop of that density):",
          {c["file"]: c["amt_gap"] for c in entries})


def set_download(docs: Path, batch: str, url: str) -> None:
    """Record the run's artifact link (zip of .mid + .mp3) on the batch, for the site."""
    if not re.match(r"^https://github\.com/[\w.-]+/[\w.-]+/actions/runs/\d+/artifacts/\d+$", url):
        raise ValueError(f"unexpected artifact url: {url}")
    path = docs / "batches.json"
    data = json.loads(path.read_text())
    for b in data["batches"]:
        if b["id"] == batch:
            b["download"] = url
    path.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "params":
        params()
    elif cmd == "produce-params":
        produce_params()
    elif cmd == "set-download":
        set_download(Path(sys.argv[2]), sys.argv[3], sys.argv[4])
    elif cmd == "final-check":
        final_check(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        sys.exit(__doc__)
