#!/usr/bin/env python3
"""Thin USI proxy: stdout is exclusively the child's USI output; stderr inherited.
Usage: python3 -B clock_proxy.py --rule cap --cap-ms 30000 --k 15 -- ENGINE [args]
Bare ponderhit is forwarded; the rewrite already applied to the matching go ponder.
"""
import argparse
import subprocess
import sys
import threading
from clock_rule import rewrite

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--rule', choices=['none', 'cap', 'byoyomi'], default='cap')
    ap.add_argument('--k', type=int, default=15)
    ap.add_argument('--cap-ms', type=int, default=30000)
    ap.add_argument('--byoyomi-ms', type=int, default=3000)
    ap.add_argument('--cwd')
    ap.add_argument('engine', nargs=argparse.REMAINDER)
    a = ap.parse_args()
    argv = a.engine[1:] if a.engine[:1] == ['--'] else a.engine
    if not argv or min(a.k, a.cap_ms, a.byoyomi_ms) < 0:
        ap.error('engine command and nonnegative parameters required')
    p = subprocess.Popen(argv, cwd=a.cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         text=True, bufsize=1)
    def output():
        for line in p.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
    reader = threading.Thread(target=output, daemon=True)
    reader.start()
    position = None
    try:
        for raw in sys.stdin:
            line = raw.strip()
            if line.startswith('position '):
                position = line
            elif line == 'usinewgame':
                position = None
            line = rewrite(line, position, a.rule, a.k, a.cap_ms, a.byoyomi_ms)
            p.stdin.write(line + '\n')
            p.stdin.flush()
            if line == 'quit':
                break
    except (BrokenPipeError, KeyboardInterrupt):
        pass
    finally:
        try:
            p.stdin.close()
            p.wait(timeout=3)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
        reader.join(timeout=1)
    return p.returncode
if __name__ == '__main__':
    sys.exit(main())
