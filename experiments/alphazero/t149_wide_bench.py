"""Task 149: indicative ABBA NPS comparison on the same 20 positions, Threads=1."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def sha(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--current", required=True)
    ap.add_argument("--wide", required=True)
    ap.add_argument("--current-eval", required=True)
    ap.add_argument("--wide-eval", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--positions", default="tools/parity_positions.sfen")
    ap.add_argument("--nodes", type=int, default=200000)
    args = ap.parse_args()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    positions = [s.strip() for s in Path(args.positions).read_text().splitlines() if s.strip()][:20]
    if len(positions) != 20:
        raise ValueError("20 positions required")
    sfens = out / "positions20.sfen"
    sfens.write_text("\n".join(positions) + "\n")
    records = []
    for i, name in enumerate(("current", "wide", "wide", "current")):
        engine = Path(getattr(args, name)).resolve()
        evaluation = Path(getattr(args, name + "_eval")).resolve()
        commands = ["usi", "setoption name Threads value 1", "setoption name USI_Hash value 64",
                    "setoption name BookFile value no_book", "setoption name USI_OwnBook value false",
                    f"setoption name EvalDir value {evaluation}", "setoption name FV_SCALE value 16",
                    "isready", f"bench 64 1 {args.nodes} {sfens} nodes", "quit"]
        result = subprocess.run([str(engine)], input="\n".join(commands) + "\n", text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
        (out / f"{i}_{name}.log").write_text(result.stdout)
        result.check_returncode()
        if re.search(r"error|mismatch|failed|No such option", result.stdout, re.I):
            raise RuntimeError(f"{name}: engine error")
        counts = re.findall(r"Nodes searched\s*:\s*(\d+)", result.stdout)
        times = re.findall(r"Total time \(ms\)\s*:\s*(\d+)", result.stdout)
        if len(counts) != 1 or len(times) != 1:
            raise RuntimeError(f"{name}: benchmark summary missing")
        if len(re.findall(r"Position:\s*\d+/20", result.stdout)) != 20:
            raise RuntimeError(f"{name}: benchmark position count mismatch")
        nodes, ms = int(counts[0]), int(times[0])
        record = dict(name=name, nodes=nodes, time_ms=ms, nps=nodes * 1000 / ms)
        records.append(record)
        print(record, flush=True)
    totals = {name: dict(nodes=sum(r["nodes"] for r in records if r["name"] == name),
                         time_ms=sum(r["time_ms"] for r in records if r["name"] == name))
              for name in ("current", "wide")}
    for v in totals.values():
        v["nps"] = v["nodes"] * 1000 / v["time_ms"]
    ratio = totals["wide"]["nps"] / totals["current"]["nps"]
    metadata = {name: dict(binary_sha256=sha(getattr(args, name)),
                           eval_sha256=sha(Path(getattr(args, name + "_eval")) / "nn.bin"))
                for name in ("current", "wide")}
    report = dict(args=vars(args), records=records, totals=totals, wide_current_ratio=ratio,
                  metadata=metadata, caveat="loaded machine; different weights/search trees; indicative only")
    (out / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"wide/current NPS={ratio:.4f} (indicative, loaded machine)")


if __name__ == "__main__":
    main()
