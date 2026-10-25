#!/usr/bin/env python3
"""Rescore an NNUE cache with frozen value and legal raw-policy argmax labels."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

import cshogi
import jax
import jax.numpy as jnp
import numpy as np
from pgx._src.games.shogi import _observe

import m2_moves
from m2_usi import load_model
from m4_distill_data import batch_state, parse_sfen_np
from t091_model import KIND, tree_digest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, REPO_ROOT)
from trainer.prepare_data import shard_files  # noqa: E402


def value_to_cp(value):
    value = np.clip(np.asarray(value, dtype=np.float64), -1.0 + 1e-7, 1.0 - 1e-7)
    return np.clip(1200.0 * np.arctanh(value), -5000.0, 5000.0).astype(np.int16)


def aligned_groups(cache_dir):
    with open(os.path.join(cache_dir, "meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    primary, _ = shard_files(meta["run_dir"])
    val_count = len(meta["val_shards"])
    return meta, {
        "train": [(meta["run_dir"], primary[:-val_count])] + [
            (run, shard_files(run)[0]) for run in meta["extra_runs"]
        ],
        "val": [(meta["run_dir"], primary[-val_count:])],
    }


def checkpoint_value_digest(path):
    with open(path, "rb") as f:
        checkpoint = pickle.load(f)
    if checkpoint.get("kind") == KIND:
        model = checkpoint["model"]
        digest = tree_digest((model[0]["value"], model[1]["value"]))
        if digest != checkpoint["frozen_value_digest"]:
            raise AssertionError("checkpoint frozen value digest mismatch")
        return digest
    return tree_digest(checkpoint["model"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--reference-value-dir", default=None)
    parser.add_argument("--reference-policy-dir", default=None)
    parser.add_argument("--random-policy-seed", type=int, default=None,
                        help="also emit fixed uniformly random legal policy labels")
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    meta, groups = aligned_groups(args.cache_dir)
    forward, model, info = load_model(args.ckpt)
    value_model_digest = checkpoint_value_digest(args.ckpt)
    reference_value_digest = None
    if args.reference_value_dir:
        with open(os.path.join(args.reference_value_dir, "rescore_metrics.json"),
                  encoding="utf-8") as f:
            reference_meta = json.load(f)
        reference_value_digest = reference_meta.get("value_model_digest")
        if reference_value_digest is None:
            reference_value_digest = checkpoint_value_digest(reference_meta["ckpt"])
        if value_model_digest != reference_value_digest:
            raise AssertionError("frozen value parameter digest differs from reference")
    print(f"ckpt={args.ckpt} info={info}", flush=True)
    obs_fn = jax.jit(jax.vmap(lambda x: _observe(x, flip=jnp.bool_(True))))

    @jax.jit
    def predict(obs):
        (logits, value), _ = forward.apply(model[0], model[1], obs, is_eval=True)
        return logits, value

    all_metrics = {}
    global_started = time.time()
    for split, count_key in (("train", "n_train"), ("val", "n_val")):
        n = int(meta[count_key])
        values = np.empty(n, dtype=np.int16)
        actions = np.empty(n, dtype=np.int16)
        move16 = np.empty(n, dtype=np.uint16)
        random_actions = (
            np.empty(n, dtype=np.int16) if args.random_policy_seed is not None else None
        )
        random_rng = (
            np.random.default_rng(args.random_policy_seed + (0 if split == "train" else 1))
            if args.random_policy_seed is not None else None
        )
        gikou_actions = np.empty(n, dtype=np.int16)
        original_evals = np.empty(n, dtype=np.int16)
        cursor = 0
        top1_hits = 0
        entropy_sum = 0.0
        illegal_raw_argmax = 0
        board = cshogi.Board()
        audit_board = cshogi.Board()

        colors = []
        boards = []
        hands = []
        sfens = []
        legal_lists = []
        gikou_batch = []
        eval_batch = []
        reference_values = None
        if args.reference_value_dir:
            reference_values = np.load(os.path.join(
                args.reference_value_dir, f"{split}_dlshogi_evals.npy"
            ))
            if len(reference_values) != n:
                raise AssertionError(f"{split}: reference value count mismatch")

        def flush():
            nonlocal cursor, top1_hits, entropy_sum, illegal_raw_argmax
            if not colors:
                return
            obs = obs_fn(batch_state(colors, boards, hands))
            logits, value = predict(obs)
            logits = np.asarray(logits)
            value = np.asarray(value)
            legal_mask = np.zeros(logits.shape, dtype=bool)
            for row, legal in enumerate(legal_lists):
                legal_mask[row, legal] = True
            raw_argmax = np.argmax(logits, axis=1)
            illegal_raw_argmax += int((~legal_mask[np.arange(len(colors)), raw_argmax]).sum())
            masked = np.where(legal_mask, logits, -np.inf)
            chosen = np.argmax(masked, axis=1).astype(np.int16)
            shifted = masked - np.max(masked, axis=1, keepdims=True)
            exp = np.where(legal_mask, np.exp(shifted), 0.0)
            probs = exp / exp.sum(axis=1, keepdims=True)
            entropy_sum += float((-np.sum(
                np.where(probs > 0.0, probs * np.log(np.maximum(probs, 1e-30)), 0.0), axis=1
            )).sum())
            top1_hits += int((chosen == np.asarray(gikou_batch, dtype=np.int16)).sum())
            end = cursor + len(colors)
            values[cursor:end] = value_to_cp(value)
            actions[cursor:end] = chosen
            gikou_actions[cursor:end] = np.asarray(gikou_batch, dtype=np.int16)
            original_evals[cursor:end] = np.asarray(eval_batch, dtype=np.int16)
            for row, action in enumerate(chosen):
                if not legal_mask[row, int(action)]:
                    raise AssertionError("masked argmax selected an illegal action")
                usi = m2_moves.action_to_usi(int(action), boards[row], colors[row])
                audit_board.set_sfen(sfens[row])
                move = audit_board.move_from_usi(usi)
                if move not in set(audit_board.legal_moves):
                    raise AssertionError(f"action {action} -> illegal {usi} at {sfens[row]}")
                move16[cursor + row] = cshogi.move16(move)
                if random_actions is not None:
                    legal = legal_lists[row]
                    random_actions[cursor + row] = legal[int(random_rng.integers(len(legal)))]
            cursor = end
            colors.clear(); boards.clear(); hands.clear(); sfens.clear()
            legal_lists.clear(); gikou_batch.clear(); eval_batch.clear()

        for run_dir, shards in groups[split]:
            for shard in shards:
                entries = np.fromfile(
                    os.path.join(run_dir, shard["file"]),
                    dtype=cshogi.HuffmanCodedPosAndEval,
                )
                for entry in entries:
                    board.set_hcp(np.asarray(entry["hcp"]))
                    sfen = board.sfen()
                    color, pgx_board, hand = parse_sfen_np(sfen)
                    legal = []
                    for move in board.legal_moves:
                        legal.append(m2_moves.usi_to_action(cshogi.move_to_usi(move), color))
                    if not legal:
                        raise AssertionError(f"teacher cache contains terminal position: {sfen}")
                    gikou_move = board.move_from_move16(int(entry["bestMove16"]))
                    gikou_action = m2_moves.usi_to_action(cshogi.move_to_usi(gikou_move), color)
                    if gikou_action not in legal:
                        raise AssertionError("Gikou bestMove16 is not legal")
                    colors.append(color)
                    boards.append(pgx_board)
                    hands.append(hand)
                    sfens.append(sfen)
                    legal_lists.append(legal)
                    gikou_batch.append(gikou_action)
                    eval_batch.append(int(entry["eval"]))
                    if len(colors) >= args.batch:
                        flush()
                if cursor % 200_000 < len(entries):
                    rate = cursor / max(time.time() - global_started, 1e-9)
                    print(f"  {split}: {cursor}/{n} ({rate:.0f}/s)", flush=True)
        flush()
        if cursor != n:
            raise AssertionError(f"{split}: {cursor} != cache count {n}")
        cache_eval = np.load(os.path.join(args.cache_dir, f"{split}_evals.npy"))
        if not np.array_equal(cache_eval, original_evals):
            raise AssertionError(f"{split}: source HCPE order differs from cache")
        metrics = {
            "count": n,
            "policy_top1_gikou": top1_hits / n,
            "policy_entropy": entropy_sum / n,
            "raw_argmax_illegal_rate": illegal_raw_argmax / n,
        }
        if reference_values is not None:
            value_delta = values.astype(np.int32) - reference_values.astype(np.int32)
            max_abs_cp = int(np.max(np.abs(value_delta)))
            metrics["reference_value_max_abs_cp"] = max_abs_cp
            metrics["reference_value_mismatch_count"] = int(np.sum(value_delta != 0))
            metrics["reference_value_param_digest_exact"] = True
            values[:] = reference_values
            metrics["reference_value_exact"] = True
        metrics["value_cp_std"] = float(values.astype(np.float64).std())
        if args.reference_policy_dir:
            reference = np.load(os.path.join(
                args.reference_policy_dir, f"{split}_policy_actions.npy"
            ))
            metrics["policy_action_change_rate"] = float(np.mean(reference != actions))
        np.save(os.path.join(args.out_dir, f"{split}_dlshogi_evals.npy"), values)
        np.save(os.path.join(args.out_dir, f"{split}_policy_actions.npy"), actions)
        np.save(os.path.join(args.out_dir, f"{split}_policy_move16.npy"), move16)
        if random_actions is not None:
            np.save(os.path.join(args.out_dir, f"{split}_policy_random_actions.npy"), random_actions)
            metrics["random_policy_seed"] = args.random_policy_seed + (
                0 if split == "train" else 1
            )
            metrics["random_policy_argmax_match_rate"] = float(np.mean(random_actions == actions))
            metrics["random_policy_gikou_match_rate"] = float(
                np.mean(random_actions == gikou_actions)
            )
        all_metrics[split] = metrics
        print(f"  {split}: {metrics}", flush=True)

    with open(os.path.join(args.out_dir, "rescore_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"ckpt": os.path.abspath(args.ckpt), "info": info,
                   "value_model_digest": value_model_digest,
                   "reference_value_digest": reference_value_digest,
                   "metrics": all_metrics}, f, indent=2)
    print(f"T091_RESCORE_DONE elapsed={time.time() - global_started:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
