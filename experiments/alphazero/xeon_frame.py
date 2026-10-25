#!/usr/bin/env python3
"""Play standard-frame chunks on the idle Xeon (docs/remote_worker.md) and bring the TSVs home.

Node-fixed games reproduce across machines (checked 2026-10-24: chunks of the adopted eval replayed on the Xeon
match the 14700KF records game for game, depth and nodes included), so a chunk played there is the same chunk.
`remote_measure_sets` mirrors `std_frame.measure_sets`: same commands, same TSV names; only chunks that are not
complete locally are sent. Each TSV is written under a temporary name on the Xeon and renamed when its process
exits 0, then copied back into LOGS_DIR, where `std_frame.measure_sets` reuses it. The TSV header keeps the Xeon
paths (provenance); every imported file is also listed in logs/xeon_frames.tsv.

    python xeon_frame.py EVAL [EVAL ...] [--sets s280,s281,s282 | --confirm BASE] [--micro] [--parallel 20]
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import std_frame  # noqa: E402

HOST = os.environ.get("BINBO_WORKER_HOST", "xeon-worker")   # ssh の接続先の別名 (~/.ssh/config で定義する)
REMOTE = os.environ.get("BINBO_WORKER_DATA", "")   # mirrors DATA_ROOT: the absolute data directory on the worker
PARALLEL = 20   # 2 engines x ~1.23 GB resident each (default 1 GiB hash) = ~2.45 GB per game; the Xeon has 62 GB
GB_PER_GAME = 2.6


def ssh(cmd, check=True, capture=False):
    if not REMOTE:
        sys.exit("xeon_frame: set BINBO_WORKER_DATA to the absolute data directory on the worker")
    return subprocess.run(["ssh", HOST, cmd], check=check, text=True, capture_output=capture)


def remote_path(path, data):
    return f"{REMOTE}/{Path(path).relative_to(data)}"


def chunk_list(eval_name, sets, which, fv):
    """(command, local tsv) pairs exactly as std_frame.measure_sets builds them."""
    repo, data, logs, binbo, yane, anchors = std_frame.paths()
    eval_dir = data / "eval_dirs" / eval_name
    fv_opt = [] if fv == std_frame.FV_DEFAULT else ["--a-opt", f"FV_SCALE={fv}"]
    out = []
    for set_name, seeds in sets.items():
        for anchor, nodes in which.items():
            bin_b, eval_b = anchors[anchor]
            for s, tsv in zip(seeds, std_frame.tsv_names(logs, eval_name, anchor, fv, seeds, set_name, nodes)):
                cmd = ["yane-vs-yane-ex", "--a-bin", yane, "--a-eval", eval_dir, *fv_opt, "--a-nodes", std_frame.NODES,
                       "--b-bin", bin_b, "--b-eval", eval_b, "--b-nodes", nodes, "--seed", s, "--games",
                       std_frame.CHUNK, "--opening-plies", "6", "--max-plies", "320"]
                out.append((cmd, tsv))
    return out


def free_kb():
    try:
        return int(ssh("awk '/MemAvailable/ {print $2}' /proc/meminfo", capture=True).stdout)
    except (subprocess.CalledProcessError, ValueError):   # a network blip is not a full Xeon
        return 0


def play(job, script, parallel, tag, head, n):
    need_kb = int((parallel * GB_PER_GAME + 4) * 1024 ** 2)
    while free_kb() < need_kb:
        print(f"XEON_FRAME {tag}: waiting for {need_kb >> 20} GB free on the Xeon", flush=True)
        time.sleep(120)
    ssh(f"mkdir -p {job} {REMOTE}/logs && rm -f {job}/DONE")
    subprocess.run(["ssh", HOST, f"cat > {job}/chunks.sh"], input=script, text=True, check=True)
    # setsid -f: a plain `nohup ... &` keeps the ssh session open until the job ends (checked on the Xeon)
    driver = (f"cd {job} && setsid -f bash -c 'xargs -P {parallel} -d \"\\n\" -I CMD bash -c CMD < chunks.sh "
              f"> xargs.log 2>&1; echo $? > DONE' > /dev/null 2>&1 < /dev/null")
    ssh(driver)
    print(f"XEON_FRAME {tag}: {n} chunks sent (HEAD {head}, parallel {parallel})", flush=True)
    while ssh(f"test -f {job}/DONE", check=False).returncode != 0:
        time.sleep(60)


def run_remote(eval_names, chunks, parallel, tag):
    """Send the missing chunks to the Xeon, wait, and import the finished TSVs."""
    repo, data, logs, *_ = std_frame.paths()
    if all(std_frame.t106.complete(t) for _, t in chunks):
        print(f"XEON_FRAME {tag}: nothing to play", flush=True)
        return
    # One frame job at a time (each fills the Xeon's memory): take the lock, then also wait for free memory.
    # The missing chunks are counted and imported while holding it, so a job never replays what the job
    # before it is still playing or importing (Task 187).
    while ssh(f"mkdir -p {REMOTE}/jobs && mkdir {REMOTE}/jobs/.busy", check=False, capture=True).returncode != 0:
        print(f"XEON_FRAME {tag}: waiting for the Xeon lock", flush=True)
        time.sleep(120)
    missing = []
    try:
        todo = [(c, t) for c, t in chunks if not std_frame.t106.complete(t)]
        if not todo:
            print(f"XEON_FRAME {tag}: nothing to play", flush=True)
            return
        for name in eval_names:
            subprocess.run(["rsync", "-a", "--relative", f"{data}/./eval_dirs/{name}", f"{HOST}:{REMOTE}/"], check=True)
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=repo, text=True,
                              capture_output=True).stdout.strip()
        job = f"{REMOTE}/jobs/{tag}"
        lines = []
        for cmd, tsv in todo:
            args = [f"{REMOTE}/bin/binbo-shogi"] + [remote_path(a, data) if str(a).startswith(str(data)) else str(a)
                                                    for a in cmd]
            out = f"{REMOTE}/logs/{tsv.name}"
            lines.append(f"cd {REMOTE} && {shlex.join(args)} > {out}.part 2>&1 && mv {out}.part {out}")
        play(job, "\n".join(lines) + "\n", parallel, tag, head, len(todo))
        stage = data / "work/xeon/import" / tag
        stage.mkdir(parents=True, exist_ok=True)
        names = [t.name for _, t in todo]
        subprocess.run(["rsync", "-a", "--files-from=-", f"{HOST}:{REMOTE}/logs/", f"{stage}/"],
                       input="\n".join(names) + "\n", text=True, check=False)
        rc = ssh(f"cat {job}/DONE", check=False, capture=True).stdout.strip()
        imported = 0
        with open(logs / "xeon_frames.tsv", "a", encoding="utf-8") as manifest:
            for _, tsv in todo:
                got = stage / tsv.name
                if got.exists() and "status=ok" in got.read_text(encoding="utf-8") and std_frame.t106.complete(got):
                    if not std_frame.t106.complete(tsv):
                        tsv.write_bytes(got.read_bytes())
                        manifest.write(f"{tsv.name}\t{HOST}\tHEAD={head}\t{time.strftime('%Y-%m-%dT%H:%M:%S')}\n")
                        imported += 1
                else:
                    missing.append(tsv.name)
        print(f"XEON_FRAME {tag}: xargs rc={rc} imported={imported} missing={len(missing)}", flush=True)
    finally:
        ssh(f"rmdir {REMOTE}/jobs/.busy", check=False)
    if missing:
        raise RuntimeError(f"Xeon chunks did not finish: {missing[:5]} ...")


def remote_measure_sets(eval_name, sets=None, which=None, fv=std_frame.FV_DEFAULT, parallel=PARALLEL, tag=None):
    """Play `eval_name`'s missing chunks on the Xeon, then return std_frame.measure_sets (all reused locally)."""
    sets = sets or std_frame.SEED_SETS
    which = which or std_frame.ANCHORS
    tag = tag or f"{eval_name}_{'_'.join(sets)}_{'_'.join(which)}"
    run_remote([eval_name], chunk_list(eval_name, sets, which, fv), parallel, tag)
    return std_frame.measure_sets(eval_name, fv=fv, sets=sets, which=which)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("evals", nargs="+")
    ap.add_argument("--confirm", type=int, help="fresh confirmation set from this base seed (std_frame.spaced_set)")
    ap.add_argument("--micro", action="store_true", help="the micro confirmation anchor instead of the routine two")
    ap.add_argument("--prefix", default=None, help="set name for --confirm (default conf<base>)")
    ap.add_argument("--parallel", type=int, default=PARALLEL)
    args = ap.parse_args()
    sets = ({args.prefix or f"conf{args.confirm}": std_frame.spaced_set(args.confirm)} if args.confirm
            else std_frame.SEED_SETS)
    which = std_frame.CONFIRM_ANCHORS if args.micro else std_frame.ANCHORS
    chunks = [c for name in args.evals for c in chunk_list(name, sets, which, std_frame.FV_DEFAULT)]
    run_remote(args.evals, chunks, args.parallel, f"{'+'.join(args.evals)}_{'_'.join(sets)}_{'_'.join(which)}")


if __name__ == "__main__":
    main()
