# Task 076 M1: parse m1_logs/*.log + *.vram into a markdown measurement table.
# Uses iterations >= 2 (iter 1 includes JIT compile; reported separately).
import ast
import glob
import os
import re
import sys

rows = []
for log in sorted(glob.glob(os.path.join(os.path.dirname(__file__), "m1_logs", "*.log"))):
    tag = os.path.basename(log)[:-4]
    cfg = {}
    iters = []
    nparams = None
    err = None
    for line in open(log, errors="replace"):
        line = line.strip()
        if line.startswith("env_id="):
            for kv in line.split():
                k, _, v = kv.partition("=")
                cfg[k] = v.strip("'")
        elif line.startswith("m1: num_params:"):
            nparams = int(line.split(":")[-1])
        elif line.startswith("{'iteration'"):
            d = ast.literal_eval(line)
            if "m1/selfplay_s" in d:
                iters.append(d)
        elif "RESOURCE_EXHAUSTED" in line or "Traceback" in line:
            err = err or line[:120]
    vram_file = log[:-4] + ".vram"
    max_vram = None
    if os.path.exists(vram_file):
        vals = [int(x) for x in open(vram_file).read().split() if x.strip().isdigit()]
        if vals:
            max_vram = max(vals)
    steady = iters[1:] if len(iters) > 1 else iters
    def avg(k):
        return sum(d[k] for d in steady) / len(steady) if steady else float("nan")
    rows.append(
        dict(
            tag=tag,
            cfg=cfg,
            nparams=nparams,
            n_iters=len(iters),
            iter1_selfplay=iters[0]["m1/selfplay_s"] if iters else None,
            selfplay_s=avg("m1/selfplay_s"),
            prep_s=avg("m1/prep_s"),
            train_s=avg("m1/train_s"),
            states_per_s=avg("m1/states_per_s"),
            mask_frac=avg("m1/value_mask_frac"),
            episodes=avg("m1/episodes_finished"),
            max_vram=max_vram,
            err=err,
        )
    )

print("| point | sims | net | batch | steps | params | VRAM(MiB) | selfplay s/iter | train s/iter | states/s | value_mask | episodes/iter | iter1(compile) |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in rows:
    c = r["cfg"]
    net = f"{c.get('num_channels','?')}ch×{c.get('num_layers','?')}L"
    total = (r["selfplay_s"] or 0) + (r["prep_s"] or 0) + (r["train_s"] or 0)
    iter1 = f"{r['iter1_selfplay']:.0f}s" if r["iter1_selfplay"] is not None else "?"
    print(
        f"| {r['tag']} ({c.get('env_id','?')}) | {c.get('num_simulations','?')} | {net} | "
        f"{c.get('selfplay_batch_size','?')} | {c.get('max_num_steps','?')} | "
        f"{r['nparams'] or '?'} | {r['max_vram'] or '?'} | "
        f"{r['selfplay_s']:.1f} | {r['train_s']:.1f} (+prep {r['prep_s']:.1f}) | "
        f"{r['states_per_s']:.0f} | {r['mask_frac']:.3f} | {r['episodes']:.0f} | {iter1} |"
        + (f" ERR: {r['err']}" if r["err"] else "")
    )
    print(f"|   -> {r['tag']} total/iter ~{total:.1f}s |", file=sys.stderr)
