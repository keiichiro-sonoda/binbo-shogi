#!/usr/bin/env python3
"""Task 136b: the Task 136 table on held-out positions that are quiescence-replaced like the train rows.

Task 136: on its own train rows the student matches the teacher whatever the king crowding (no trend),
but on the 50k val rows it grows optimistic with enemy pieces near its king (+33 cp per piece). Val is
not quiescence-replaced (Task 128 replaced only train), so "unseen" and "not quiet" were confounded.
Here the same 50k val positions (teacher/run1m shards 95-99 = caches/shared_64m val) are replaced by
the leaf of their qsearch PV (YaneuraOu `qsearch_psv`, as Task 128) and every leaf is labelled afresh by
the teacher (WS-T + Gumbel MCTS S=8, cp = 1200 atanh(v), Task 104's labeller). Then the student's
residual (student - teacher) is tabulated by att2 like Task 136.

Difference from the train replacement: the qsearch eval is the adopted eval (task129) because Task 119's
`task109_student_64m_e24` was not carried over to this machine. qsearch only resolves captures, so the
leaves are quiet either way; which leaf is reached can differ on some rows.

  t136b_val_qs.py label     (venv_pgx: qsearch + WS-T labels -> work/task136/val_qs.npz)
  t136b_val_qs.py analyze   (venv: student predictions + tables -> work/task136/val_qs.txt)
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import cshogi  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, str(Path(os.path.realpath(__file__)).parents[2]))

from trainer import kp  # noqa: E402
from trainer.prepare_data import shard_files  # noqa: E402

DATA = Path(os.environ.get("DATA_ROOT", Path.home() / "binbo-shogi-data"))
WORK = DATA / "work" / "task136"
VAL_CACHE = DATA / "caches" / "shared_64m"
QS_EVAL = DATA / "eval_dirs" / "task129_student_64m_qsrel_e12"
YANE = DATA / "engines/yaneuraou/source/YaneuraOu-by-gcc"
WST = Path.home() / "task074_frameworks/pgx/examples/alphazero/t087_wst/ws2.ckpt"
SIMS, BATCH = 8, 128


def label():
    import qs_replace as qs
    from m2_usi import load_model
    from t099_search_label import build_batched_search, score_sfens, sign_oracle, value_to_cp

    WORK.mkdir(parents=True, exist_ok=True)
    meta = json.loads((VAL_CACHE / "meta.json").read_text(encoding="utf-8"))
    shards = shard_files(meta["run_dir"])[0][-len(meta["val_shards"]):]
    if [s["index"] for s in shards] != meta["val_shards"]:
        raise AssertionError(f"val shards {[s['index'] for s in shards]} != {meta['val_shards']}")
    ents = np.concatenate([np.fromfile(os.path.join(meta["run_dir"], s["file"]), dtype=cshogi.HuffmanCodedPosAndEval)
                           for s in shards])
    val_idx = np.load(VAL_CACHE / "val_indices.npy")
    board = cshogi.Board()
    bad = 0
    for i, e in enumerate(ents):  # the shards must be exactly the cached val rows
        board.set_hcp(e["hcp"])
        bad += not np.array_equal(np.asarray(kp.canonical_shared_row(board), dtype=np.int32), val_idx[i])
    print(f"T136B val rows {len(ents)} (cache {len(val_idx)}), feature mismatches {bad}", flush=True)
    if bad or len(ents) != len(val_idx):
        raise AssertionError("val shards do not reproduce the cached val rows")
    inp, outp = WORK / "val_in.psv", WORK / "val_out.psv"
    qs.hcpe_to_psv(ents).tofile(inp)
    engine = qs.QSearchEngine(YANE, QS_EVAL, threads=4, hash_mb=16)
    try:
        st = engine.run(inp, outp)
    finally:
        engine.close()
    a, b = np.fromfile(inp, dtype=qs.PSV), np.fromfile(outp, dtype=qs.PSV)
    if st["decode_errors"] or st["illegal_pv"] or st["records"] != len(a):
        raise RuntimeError(f"qsearch_psv: {st}")
    changed, flip = qs.leaf_rows(a, b)
    sfens, feats, in_check = [], np.empty((len(b), 77), np.int32), 0
    for i in range(len(b)):
        board.set_psfen(b[i]["sfen"])
        in_check += board.is_check()
        sfens.append(board.sfen())
        feats[i] = kp.canonical_shared_row(board)
    print(f"T136B qsearch: changed {changed.mean():.3f}, flip {flip.mean():.3f}, leaf in check {in_check}", flush=True)
    forward, model, _ = load_model(str(WST))
    sign_oracle(forward, model, SIMS)
    search = build_batched_search(forward, model, SIMS)
    t0 = time.time()
    values, _q, _raw, _mv = score_sfens(search, sfens, BATCH)
    if not np.isfinite(values).all():
        raise AssertionError("non-finite labels")
    cp = value_to_cp(values).astype(np.int16)
    print(f"T136B labelled {len(sfens)} leaves in {time.time() - t0:.0f}s", flush=True)
    np.savez(WORK / "val_qs.npz", feats=feats, cp=cp, changed=changed, flip=flip)
    inp.unlink()
    outp.unlink()
    return 0


def analyze():
    import torch
    from t134_sharp_relabel_s32 import att2_from_half
    from t136_bridge_residual import STUDENTS, load_model, predict, table

    z = np.load(WORK / "val_qs.npz")
    feats, y, changed = z["feats"], z["cp"].astype(np.float32).clip(-2000, 2000), z["changed"]
    a_stm, a_opp = att2_from_half(feats[:, :39]), att2_from_half(feats[:, 39:])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log = open(WORK / "val_qs.txt", "w", encoding="utf-8")

    def f(s):
        print(s, flush=True)
        log.write(s + "\n")

    f(f"val leaves {len(y):,} (changed by qsearch {changed.mean():.3f}); att2_stm>=3 {np.mean(a_stm >= 3):.3f}, "
      f"att2_opp>=3 {np.mean(a_opp >= 3):.3f}")
    out = {}
    for st in ("task129", "task134"):
        p = predict(load_model(STUDENTS[st][0], device), feats, device)
        f("")
        out[st] = table(f"student {st} vs WS-T S=8 on quiescence-replaced val", p - y, a_stm, a_opp, f)
        f("")
        table(f"student {st}, unchanged rows only (P already quiet)", (p - y)[~changed], a_stm[~changed],
              a_opp[~changed], f)
    json.dump(out, open(WORK / "val_qs.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit({"label": label, "analyze": analyze}[sys.argv[1]]())
