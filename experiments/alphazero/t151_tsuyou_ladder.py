#!/usr/bin/env python3
"""Task 151: can つようら王11micro (MIT, 2026-10-14) be a third yardstick? -- node ladder against the adopted eval.

Why: Task 142's gain over task129 is skewed to the tanuki- anchor (+48 vs AobaNNUE +12 on the old sets), and the
fine-tune data (Suisho5 depth 9) come from an NNUE lineage close to tanuki-. A strong, free yardstick from another
lineage tells whether the gain is real. Source: https://yaneurao.fanbox.cc/posts/12761272 (public post): nn.bin and
progress.bin under MIT, arch SFNN_ka2_2048_8_64_progress8 (micro), FV_SCALE 24 (eval_options.txt in the archive).
Engine: YaneuraOu GitHub master 32165a8 (9.80git, supports progress buckets; the local 9.41 does not), built with
g++ / AVX2 as `make normal PYTHON=python3 TARGET_CPU=AVX2 COMPILER=g++
YANEURAOU_EDITION=YANEURAOU_ENGINE_SFNN_ka2_2048_8_64_progress8` -> engines/yaneuraou980/source/YaneuraOu-SFNN_ka2_2048_8_64_progress8
(sha256 680dc8d4...). Archive sha256: micro d1326e9a..., mini bd5cd719... (public/tsuyouraou11/sha256.txt).

Ladder (Task 127's procedure, with spaced seeds so the openings do not overlap): the adopted eval task142_finetune at
100,000 nodes vs micro at 1k / 2k / 4k / 8k / 16k / 32k nodes, 4 chunks x 26 games per rung (seeds 20286000 + 100 i,
104 games, 52 distinct openings). Pick the rung whose score (adopted's) is closest to 0.65 within 0.55-0.75 and with
0 errors; none in band -> OUT_OF_BAND. Descriptive only; adding the anchor to the standard frame is a later decision.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import std_frame  # noqa: E402
import t106_filter as t106  # noqa: E402

TAG = "task151"
CAND = "task142_finetune"
RUNGS = (1_000, 2_000, 4_000, 8_000, 16_000, 32_000,
         11_000, 12_000, 13_000)  # added after the factor-2 ladder fell between 8k (0.827) and 16k (0.510)
SEEDS, GAMES = tuple(20286000 + 100 * i for i in range(4)), 26
TARGET, BAND = 0.65, (0.55, 0.75)


def main():
    repo, data, logs, binbo, yane, _anchors = std_frame.paths()
    env = dict(os.environ, PYTHONPATH=str(repo))
    micro_bin = data / "engines/yaneuraou980/source/YaneuraOu-SFNN_ka2_2048_8_64_progress8"
    micro_eval = next((data / "public/tsuyouraou11").glob("TuyouraOu11micro-*/"))
    commands, tsvs, index = [], [], {}
    for nodes in RUNGS:
        names = [logs / f"{TAG}_ladder_micro{nodes}_w{i}_seed{s}_n{GAMES}.tsv" for i, s in enumerate(SEEDS)]
        index[nodes] = names
        for s, t in zip(SEEDS, names):
            commands.append([binbo, "yane-vs-yane-ex", "--a-bin", yane, "--a-eval", data / "eval_dirs" / CAND,
                             "--a-nodes", std_frame.NODES, "--b-bin", micro_bin, "--b-eval", micro_eval,
                             "--b-nodes", nodes, "--seed", s, "--games", GAMES, "--opening-plies", "6",
                             "--max-plies", "320"])
            tsvs.append(t)
    print(f"=== T151 ladder {CAND}@{std_frame.NODES} vs micro {RUNGS} parallel={t106.PARALLEL} ===", flush=True)
    t106.run_chunks(commands, tsvs, env, repo)
    ladder = {}
    for nodes, names in index.items():
        rows, bad = t106.combine(names, GAMES)
        score = sum(r["score"] for r in rows) / len(rows)
        ladder[nodes] = {"score": score, "games": len(rows), "errors": len(bad)}
        print(f"T151_LADDER micro@{nodes} score={score:.3f} games={len(rows)} errors={len(bad)}", flush=True)
    ok = [(n, v["score"]) for n, v in ladder.items() if not v["errors"] and BAND[0] <= v["score"] <= BAND[1]]
    pick = min(ok, key=lambda x: abs(x[1] - TARGET))[0] if ok else None
    out = logs / f"{TAG}_results.json"
    out.write_text(json.dumps({"task": 151, "cand": CAND, "ladder": ladder, "pick": pick,
                               "verdict": "PICKED" if pick else "OUT_OF_BAND"}, indent=2), encoding="utf-8")
    print(f"T151_PICK micro@{pick} results={out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
