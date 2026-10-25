#!/usr/bin/env python3
"""Task 139: on QUIET positions, is the teacher itself more optimistic than the yardsticks about crowded kings?

Task 136: the student reproduces the teacher on quiet positions (residual +2.7 cp per enemy piece near the king),
and Task 133's "teacher -56 vs student +20" compared the teacher on raw positions with the student under search.
So if binbo is optimistic in real games (Task 133: +20 / +27 cp per piece vs tanuki- and AobaNNUE), the cause is
either the teacher's labels on quiet positions or the search. Here Task 133's 2,985 game positions (floodgate
>= 3800 and binbo self-play) are replaced by their qsearch leaf (YaneuraOu qsearch_psv, adopted eval, as Task 136b)
and then, on the SAME leaves: tanuki- / AobaNNUE / binbo at 200k nodes (t133 evaluate) and the teacher WS-T S=8
(t133 label). The Task 133 regression is rerun with target = wst_s8 and target = binbo.

  t139_teacher_vs_rulers_qs.py replace   (venv_pgx or venv: positions_qs.tsv)
  then t133_king_safety_probe.py evaluate / label / analyze on it (see the driver at the bottom)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import cshogi
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, str(Path(os.path.realpath(__file__)).parents[2]))

import qs_replace as qs  # noqa: E402
from t133_king_safety_probe import FEATURES, king_features  # noqa: E402

DATA = Path(os.environ.get("DATA_ROOT", Path.home() / "binbo-shogi-data"))
SRC = DATA / "work/task133/positions.tsv"
WORK = DATA / "work/task139"
QS_EVAL = DATA / "eval_dirs/task129_student_64m_qsrel_e12"
YANE = DATA / "engines/yaneuraou/source/YaneuraOu-by-gcc"


def replace():
    WORK.mkdir(parents=True, exist_ok=True)
    lines = SRC.read_text(encoding="utf-8").splitlines()
    head, rows = lines[0].split("\t"), [l.split("\t") for l in lines[1:]]
    b = cshogi.Board()
    hcpe = np.zeros(len(rows), dtype=cshogi.HuffmanCodedPosAndEval)
    for i, r in enumerate(rows):
        b.set_sfen(r[head.index("sfen")])
        b.to_hcp(hcpe[i]["hcp"])
    inp, outp = WORK / "in.psv", WORK / "out.psv"
    qs.hcpe_to_psv(hcpe).tofile(inp)
    engine = qs.QSearchEngine(YANE, QS_EVAL, threads=4, hash_mb=16)
    try:
        st = engine.run(inp, outp)
    finally:
        engine.close()
    a, o = np.fromfile(inp, dtype=qs.PSV), np.fromfile(outp, dtype=qs.PSV)
    if st["decode_errors"] or st["illegal_pv"] or st["records"] != len(a):
        raise RuntimeError(f"qsearch_psv: {st}")
    changed, _flip = qs.leaf_rows(a, o)
    out, checks = [], 0
    for i, r in enumerate(rows):
        b.set_psfen(o[i]["sfen"])
        checks += b.is_check()
        fs, fo = king_features(b, b.turn), king_features(b, 1 - b.turn)
        feats = [fs["def1"], fs["def2"], fs["att2"], fs["opp_hand"], fs["kfile"], fs["krank"],
                 fo["def1"], fo["def2"], fo["att2"], fo["opp_hand"], fo["kfile"], fo["krank"]]
        out.append([r[0], r[1], r[2], b.sfen()] + [str(x) for x in feats] + [str(int(changed[i]))])
    with open(WORK / "positions_qs.tsv", "w", encoding="utf-8") as f:
        f.write("\t".join(["set", "game", "ply", "sfen"] + FEATURES + ["qs_changed"]) + "\n")
        for r in out:
            f.write("\t".join(r) + "\n")
    print(f"T139 replaced {len(rows)} positions: changed {changed.mean():.3f}, leaf in check {checks}", flush=True)
    inp.unlink()
    outp.unlink()


if __name__ == "__main__":
    replace()
