#!/usr/bin/env python3
"""Task 179: re-apply the light Suisho5 finish after the self-play wring.

Task 150 (Codex) saw the wring dilute what the Suisho5 finish had added; Task 143 saw more passes over the
same public data trending better. Here the same light finish (Suisho5 validation 7.24M, lr 1e-5, 2 epochs, the
recipe of Task 142) is applied once more on top of the wrung evals:
  F154 = task154_R_result30 (adopted) + finish
  F159 = task159_R30_e2 (adopted + round-2 wring, +12.0 n.s.) + finish
Primary: pooled 3 sets (s280-282), candidate - adopted. CI low > 0 -> confirm the best on the fresh set conf179
(spaced_set(20296000)) with both routine anchors and micro. Gates: 0cp GREEN, frame failures 0.
Judging is split: F154 plays on the Xeon (xeon_frame), F159 on the 14700KF, at the same time.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))

import std_frame  # noqa: E402
import t106_filter as t106  # noqa: E402
import t129_qs_relabel_64m as t129  # noqa: E402
import xeon_frame  # noqa: E402
from t147_self_relabel_train import export  # noqa: E402

TAG = "task179"
ADOPTED = "task154_R_result30"
ARMS = {"F154": ADOPTED, "F159": "task159_R30_e2"}
REMOTE_ARM = "F154"
N_PUB = 7_240_673
CONF = {"conf179": std_frame.spaced_set(20296000)}
CODEX_NPS_FLAG = "work/task178_fast_build/summary.json"   # Task 178 measures NPS on this machine; wait for it


def parallel(jobs):
    """Run {key: thunk} in threads; re-raise the first failure."""
    out, errors = {}, []

    def run(key, thunk):
        try:
            out[key] = thunk()
        except Exception as exc:  # noqa: BLE001
            errors.append((key, exc))
    threads = [threading.Thread(target=run, args=(k, f)) for k, f in jobs.items()]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise RuntimeError(f"{errors[0][0]}: {errors[0][1]}") from errors[0][1]
    return out


def wait_for_quiet(data, limit_s=3600):
    flag, start = data / CODEX_NPS_FLAG, time.time()
    while not flag.exists() and time.time() - start < limit_s:
        time.sleep(60)
    print(f"T179_LOCAL_START codex_flag={flag.exists()} waited={time.time() - start:.0f}s", flush=True)


def main():
    repo, data, logs, *_ = std_frame.paths()
    py = Path(os.environ.get("VENV_PY", data / "venv/bin/python"))
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "experiments/alphazero", "trainer", "src"],
                           cwd=repo, capture_output=True, text=True, check=True).stdout.strip()
    if dirty:
        raise RuntimeError(f"code is dirty; commit first:\n{dirty}")
    env = dict(os.environ, PYTHONPATH=str(repo))
    pub = data / "caches/public_suisho5_val7m"
    failures, names = [], {}
    for key, start in ARMS.items():
        name, out = f"{TAG}_{key}", data / "models" / f"{TAG}_{key}"
        if not (out / "history.json").exists():
            out.mkdir(parents=True, exist_ok=True)
            print(f"=== T179 STAGE train {key} (from {start}) ===", flush=True)
            t106.run([py, "-m", "trainer.train_nnue", pub, "--out", out, "--qat", "--qat-tie", "--feature-dim", "256",
                      "--hidden-dim", "32", "--batch", "1024", "--scale-cp", "600", "--factorize", "--augment-mirror",
                      "--mirror-cache", data / t129.MIRROR_CACHE, "--fast", "--checkpoint-minutes", "10", "--resume",
                      "--lr", "1e-5", "--epochs", "2", "--init-weights", data / "models" / start / "nnue_final.pt",
                      "--eval-npy", data / "work/task142/labels_public", "--limit-train", N_PUB,
                      "--compact-cache", Path(str(pub) + "_compact")], logs / f"{name}.log", cwd=repo, env=env)
        failures += export(name, out, "nnue_final.pt", data, logs, py, repo, env)
        names[key] = name
    if failures:
        raise RuntimeError(f"export gate failed: {failures}")
    base, bad = std_frame.measure_sets(ADOPTED, env)   # all reused (played in Task 154)
    failures += bad

    def local(name):
        wait_for_quiet(data)
        return std_frame.measure_sets(name, env)
    frames = parallel({key: (lambda n=name: xeon_frame.remote_measure_sets(n)) if key == REMOTE_ARM
                       else (lambda n=name: local(n)) for key, name in names.items()})
    results = {}
    for key, name in names.items():
        cand, bad = frames[key]
        total, per, per_set, bad2 = std_frame.delta_sets(cand, base, f"({name} - {ADOPTED})")
        failures += bad + bad2
        results[name] = {"pooled": total, "per_anchor": per, "per_set": per_set}
        print(f"T179_DELTA ({name} - {ADOPTED}) pooled={total['delta_elo']:+.2f} "
              f"ci={[round(v, 2) for v in total['ci95']]} "
              + " ".join(f"{k}={d['delta_elo']:+.1f}" for k, d in per.items()), flush=True)
    ok = [n for n in names.values() if results[n]["pooled"]["ci95"][0] > 0]
    verdict, best = ("INVALID" if failures else "NO_BETTER"), None
    if ok and not failures:
        best = max(ok, key=lambda n: results[n]["pooled"]["delta_elo"])
        conf = {}
        for which, label in ((None, "routine"), (std_frame.CONFIRM_ANCHORS, "micro")):
            got = parallel({"base": lambda w=which: xeon_frame.remote_measure_sets(ADOPTED, sets=CONF, which=w),
                            "cand": lambda w=which: std_frame.measure_sets(best, env, sets=CONF, which=w)})
            (cb, b1), (cc, b2) = got["base"], got["cand"]
            ct, cper, _ps, b3 = std_frame.delta_sets(cc, cb, f"confirm {label} ({best} - {ADOPTED})")
            failures += b1 + b2 + b3
            conf[label] = {"pooled": ct, "per_anchor": cper}
            print(f"T179_CONFIRM {label} {best} pooled={ct['delta_elo']:+.2f} "
                  f"ci={[round(v, 2) for v in ct['ci95']]}", flush=True)
        results[best]["confirm"] = conf
        confirmed = all(conf[k]["pooled"]["delta_elo"] > 0 for k in conf)
        verdict = "INVALID" if failures else ("ADOPT_CANDIDATE" if confirmed else "NOT_CONFIRMED")
    (logs / f"{TAG}_results.json").write_text(json.dumps({"verdict": verdict, "best": best, "results": results,
                                                          "failures": failures}, indent=1, default=str),
                                             encoding="utf-8")
    print(f"T179_VERDICT {verdict} best={best} failures={failures}", flush=True)


if __name__ == "__main__":
    main()
