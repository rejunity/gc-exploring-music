import json, sys, statistics as st
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
from anticipation.convert import midi_to_events
from anticipation.vocab import AUTOREGRESS
T = Path(sys.argv[1]); name = sys.argv[2] if len(sys.argv) > 2 else "stanford-crfm/music-medium-800k"
rows = json.loads((T / "index.json").read_text())
model = AutoModelForCausalLM.from_pretrained(name).cuda().eval()
@torch.no_grad()
def nll(events):
    x = torch.tensor([[AUTOREGRESS] + events[:1020]]).cuda()
    logits = model(x).logits[0, :-1]
    return torch.nn.functional.cross_entropy(logits, x[0, 1:]).item()
for r in rows:
    ev = midi_to_events(str(T / f"{r['name']}.mid"))
    r["amt_nll"] = round(nll(ev), 3); r["amt_tokens"] = len(ev)
(T / "amt.json").write_text(json.dumps(rows, indent=1))
order = ["random", "b2-rules", "b2-corpus", "b3-evolved", "reference"]
print(f"model {name}: mean NLL per token (lower = more like Lakh MIDI music)")
for l in order:
    rs = [r for r in rows if r["label"] == l]
    print(f"{l:12s} n={len(rs)}  nll {st.mean(r['amt_nll'] for r in rs):.3f}  (min {min(r['amt_nll'] for r in rs):.2f}, max {max(r['amt_nll'] for r in rs):.2f})")
