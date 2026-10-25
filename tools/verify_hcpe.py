#!/usr/bin/env python3
import argparse
import csv
import sys

import cshogi
import numpy as np


def sfen_core(sfen: str) -> str:
    return " ".join(sfen.split()[:3])


def hcp_hex(hcp) -> str:
    return bytes(hcp.tolist()).hex()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("hcpe")
    parser.add_argument("expected_tsv")
    args = parser.parse_args()

    dtype = cshogi.HuffmanCodedPosAndEval
    if dtype.itemsize != 38:
        raise AssertionError(f"unexpected cshogi HCPE itemsize: {dtype.itemsize}")

    entries = np.fromfile(args.hcpe, dtype=dtype)
    with open(args.expected_tsv, newline="", encoding="utf-8") as file:
        expected = list(csv.DictReader(file, delimiter="\t"))

    if len(entries) != len(expected):
        raise AssertionError(f"entry count mismatch: got {len(entries)}, expected {len(expected)}")

    for index, (entry, row) in enumerate(zip(entries, expected)):
        board = cshogi.Board()
        board.set_hcp(entry["hcp"])

        got_sfen = sfen_core(board.sfen())
        got_eval = int(entry["eval"])
        got_best_move16 = int(entry["bestMove16"])
        got_bestmove = cshogi.move_to_usi(got_best_move16)
        got_game_result = int(entry["gameResult"])
        got_dummy = int(entry["dummy"])
        got_hcp_hex = hcp_hex(entry["hcp"])

        checks = {
            "sfen_core": (got_sfen, row["sfen_core"]),
            "eval": (str(got_eval), row["eval"]),
            "bestmove_usi": (got_bestmove, row["bestmove_usi"]),
            "bestMove16": (str(got_best_move16), row["bestMove16"]),
            "gameResult": (str(got_game_result), row["gameResult"]),
            "dummy": (str(got_dummy), row["dummy"]),
            "hcp_hex": (got_hcp_hex, row["hcp_hex"]),
        }
        mismatches = {
            key: (got, want)
            for key, (got, want) in checks.items()
            if got != want
        }
        if mismatches:
            for key, (got, want) in mismatches.items():
                print(
                    f"mismatch index={index} name={row['name']} field={key} got={got} expected={want}",
                    file=sys.stderr,
                )
            return 1

        print(
            "ok "
            f"index={index} name={row['name']} sfen='{got_sfen}' "
            f"eval={got_eval} bestmove={got_bestmove} bestMove16={got_best_move16} "
            f"gameResult={got_game_result}"
        )

    print(f"cshogi_dtype_itemsize={dtype.itemsize}")
    print(f"entries={len(entries)}")
    print("cshogi_readback=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
