import json, sys, statistics as st
from pathlib import Path
from audiobox_aesthetics.infer import initialize_predictor
T = Path(sys.argv[1]); rows = json.loads((T / "index.json").read_text())
pred = initialize_predictor()
import soundfile as sf, torch
def ld(p):
    x, sr = sf.read(p, dtype="float32", always_2d=True)
    return {"path": torch.from_numpy(x.T.copy()), "sample_rate": sr}
out = [pred.forward([ld(T / f"{r['name']}.wav")])[0] for r in rows]
for r, o in zip(rows, out): r.update({k: round(v, 2) for k, v in o.items()})
(T / "aes.json").write_text(json.dumps(rows, indent=1))
labels = sorted({r["label"] for r in rows}, key=["random", "b2-rules", "b2-corpus", "b3-evolved", "reference"].index)
keys = ["CE", "CU", "PC", "PQ"]
print(f"{'group':12s} n  " + "  ".join(f"{k:>5s}" for k in keys))
for l in labels:
    rs = [r for r in rows if r["label"] == l]
    print(f"{l:12s} {len(rs)}  " + "  ".join(f"{st.mean(r[k] for r in rs):5.2f}" for k in keys))
