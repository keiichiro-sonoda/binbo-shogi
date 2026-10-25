#!/usr/bin/env python3
"""QAT NNUE teacher test with an optional training-only policy target.

The deployable evaluator is unchanged. During training only, an optional
2187-class head branches from either QATSharedNNUE's final hidden activation or
its concatenated feature-transform accumulators.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
from torch import nn

try:
    from . import kp
    from .binbo_to_yane import build_collapse_map
    from .nnue import QATSharedNNUE
    from .train_nnue import pearson
except ImportError:
    import kp
    from binbo_to_yane import build_collapse_map
    from nnue import QATSharedNNUE
    from train_nnue import pearson


NUM_ACTIONS = 2187


class PolicyAuxStudent(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        scale_cp: float,
        policy_head: str = "linear",
        policy_hidden: int = 512,
    ):
        super().__init__()
        self.policy_head_kind = policy_head
        self.evaluator = QATSharedNNUE(
            total_features=kp.CANONICAL_SHARED_TOTAL,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            fake_quant=True,
            scale_cp=scale_cp,
            tie_collapse_map=build_collapse_map(),
        )
        if policy_head == "none":
            self.policy_head = None
        elif policy_head == "linear":
            self.policy_head = nn.Linear(hidden_dim, NUM_ACTIONS)
        elif policy_head == "mlp":
            self.policy_head = nn.Sequential(
                nn.Linear(hidden_dim, policy_hidden),
                nn.ReLU(),
                nn.Linear(policy_hidden, NUM_ACTIONS),
            )
        elif policy_head == "ft-mlp":
            self.policy_head = nn.Sequential(
                nn.Linear(2 * feature_dim, policy_hidden),
                nn.ReLU(),
                nn.Linear(policy_hidden, NUM_ACTIONS),
            )
        else:
            raise ValueError(f"unsupported policy head: {policy_head}")
        if isinstance(self.policy_head, nn.Linear):
            nn.init.kaiming_uniform_(self.policy_head.weight, nonlinearity="linear")
            nn.init.zeros_(self.policy_head.bias)
        elif isinstance(self.policy_head, nn.Sequential):
            linear_layers = [m for m in self.policy_head if isinstance(m, nn.Linear)]
            nn.init.kaiming_uniform_(linear_layers[0].weight, nonlinearity="relu")
            nn.init.zeros_(linear_layers[0].bias)
            nn.init.kaiming_uniform_(linear_layers[1].weight, nonlinearity="linear")
            nn.init.zeros_(linear_layers[1].bias)

    def forward(self, own_flat, own_off, opp_flat, opp_off):
        features = self.evaluator.forward_features(own_flat, own_off, opp_flat, opp_off)
        hidden = self.evaluator.forward_hidden_from_features(features)
        policy_input = features if self.policy_head_kind == "ft-mlp" else hidden
        policy_logits = None if self.policy_head is None else self.policy_head(policy_input)
        return self.evaluator.forward_output(hidden), policy_logits


def load_targets(directory: str, split: str, stem: str, dtype=None):
    array = np.load(os.path.join(directory, f"{split}_{stem}.npy"))
    return torch.from_numpy(array.astype(dtype) if dtype is not None else array)


def model_args(indices: torch.Tensor, rows: torch.Tensor, device: str):
    batch = indices[rows].long()
    count = batch.shape[0]
    own = batch[:, :39].reshape(-1).to(device)
    opp = batch[:, 39:77].reshape(-1).to(device)
    own_off = torch.arange(0, count * 39, 39, dtype=torch.long, device=device)
    opp_off = torch.arange(0, count * 38, 38, dtype=torch.long, device=device)
    return own, own_off, opp, opp_off


def run_epoch(model, indices, cp_targets, policy_targets, batch_size, device,
              policy_weight, scale_cp, optimizer, order):
    training = optimizer is not None
    model.train(training)
    n = indices.shape[0]
    total_value = total_value_abs = total_policy = total_policy_entropy = 0.0
    total_policy_weighted = total_combined = 0.0
    cp_predictions = []
    hits = 0
    for start in range(0, n, batch_size):
        rows = order[start:start + batch_size] if order is not None else torch.arange(
            start, min(start + batch_size, n)
        )
        args = model_args(indices, rows, device)
        cp_target = cp_targets[rows].to(device)
        action_target = (
            policy_targets[rows].long().to(device)
            if policy_targets is not None else None
        )
        if training:
            optimizer.zero_grad(set_to_none=True)
            logit, policy_logits = model(*args)
        else:
            with torch.no_grad():
                logit, policy_logits = model(*args)
        cp_prediction = logit * scale_cp
        value_loss = nn.functional.mse_loss(cp_prediction, cp_target)
        value_abs = nn.functional.l1_loss(cp_prediction, cp_target)
        if policy_logits is None:
            policy_loss = value_loss.new_zeros(())
            policy_entropy = value_loss.new_zeros(())
            policy_weighted = value_loss.new_zeros(())
        else:
            policy_loss = nn.functional.cross_entropy(policy_logits, action_target)
            policy_entropy = value_loss.new_zeros(())
            policy_weighted = policy_weight * scale_cp * scale_cp * policy_loss
        combined = value_loss + policy_weighted
        if training:
            combined.backward()
            optimizer.step()
        count = rows.shape[0]
        total_value += float(value_loss.detach()) * count
        total_value_abs += float(value_abs.detach()) * count
        total_policy += float(policy_loss.detach()) * count
        total_policy_entropy += float(policy_entropy.detach()) * count
        total_policy_weighted += float(policy_weighted.detach()) * count
        total_combined += float(combined.detach()) * count
        if not training:
            cp_predictions.append(cp_prediction.detach().cpu())
            if policy_logits is not None:
                hits += int((policy_logits.argmax(dim=1) == action_target).sum())
    predictions = torch.cat(cp_predictions) if cp_predictions else torch.empty(0)
    target_std = float(cp_targets.float().std(unbiased=False)) if not training else None
    prediction_std = float(predictions.float().std(unbiased=False)) if not training else None
    has_policy = policy_targets is not None
    return {
        "value_mse": total_value / n,
        "value_mae": total_value_abs / n,
        "value_target_std": target_std,
        "value_prediction_std": prediction_std,
        "value_std_ratio": (
            prediction_std / target_std if target_std and prediction_std is not None else None
        ),
        "policy_ce": total_policy / n if has_policy else None,
        "policy_target_entropy": total_policy_entropy / n if has_policy else None,
        "policy_kl": (
            (total_policy - total_policy_entropy) / n if has_policy else None
        ),
        "policy_weighted": total_policy_weighted / n,
        "combined": total_combined / n,
        "value_corr": pearson(predictions, cp_targets) if not training else None,
        "policy_top1": hits / n if not training and has_policy else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache_dir")
    parser.add_argument("--value-dir", required=True,
                        help="aligned {train,val}_dlshogi_evals.npy from frozen WS-T")
    parser.add_argument("--policy-dir", default=None,
                        help="aligned {train,val}_policy_actions.npy from the teacher policy")
    parser.add_argument("--policy-stem", default="policy_actions",
                        help="hard-label filename stem under --policy-dir")
    parser.add_argument("--out", required=True)
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--scale-cp", type=float, default=600.0)
    parser.add_argument("--clamp-cp", type=float, default=5000.0)
    parser.add_argument("--policy-weight", type=float, default=0.1)
    parser.add_argument("--policy-head", choices=["none", "linear", "mlp", "ft-mlp"],
                        default="linear")
    parser.add_argument("--policy-hidden", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--limit-train", type=int, default=0)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    with open(os.path.join(args.cache_dir, "meta.json"), encoding="utf-8") as f:
        cache_meta = json.load(f)
    if not cache_meta.get("canonical_shared") or cache_meta["features_per_position"] != 77:
        raise ValueError("Task 091 policy auxiliary requires canonical-shared HalfKP cache")

    train_indices = torch.from_numpy(np.load(os.path.join(args.cache_dir, "train_indices.npy")))
    val_indices = torch.from_numpy(np.load(os.path.join(args.cache_dir, "val_indices.npy")))
    train_cp = load_targets(args.value_dir, "train", "dlshogi_evals", np.float32)
    val_cp = load_targets(args.value_dir, "val", "dlshogi_evals", np.float32)
    if args.policy_head == "none":
        train_actions = val_actions = None
    else:
        if not args.policy_dir:
            raise ValueError("--policy-dir is required unless --policy-head=none")
        train_actions = load_targets(args.policy_dir, "train", args.policy_stem)
        val_actions = load_targets(args.policy_dir, "val", args.policy_stem)
    if args.limit_train > 0:
        train_indices = train_indices[:args.limit_train]
        train_cp = train_cp[:args.limit_train]
        if train_actions is not None:
            train_actions = train_actions[:args.limit_train]
    for name, indices, cp, actions in (
        ("train", train_indices, train_cp, train_actions),
        ("val", val_indices, val_cp, val_actions),
    ):
        if len(indices) != len(cp) or (actions is not None and len(indices) != len(actions)):
            raise AssertionError(f"{name} alignment mismatch")
        if actions is not None and (int(actions.min()) < 0 or int(actions.max()) >= NUM_ACTIONS):
            raise AssertionError(f"{name} policy target out of range")
    train_cp = train_cp.clamp(-args.clamp_cp, args.clamp_cp)
    val_cp = val_cp.clamp(-args.clamp_cp, args.clamp_cp)

    device = "cuda" if (
        args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    ) else "cpu"
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model = PolicyAuxStudent(
        args.feature_dim, args.hidden_dim, args.scale_cp,
        policy_head=args.policy_head, policy_hidden=args.policy_hidden,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    generator = torch.Generator().manual_seed(args.seed)
    print(
        f"device={device} train={len(train_indices)} val={len(val_indices)} "
        f"policy_head={args.policy_head} policy_hidden={args.policy_hidden} "
        f"policy_weight={args.policy_weight} loss=cp_mse+weight*scale_cp^2*policy_ce",
        flush=True,
    )

    initial = run_epoch(
        model, val_indices, val_cp, val_actions, args.batch_size, device,
        args.policy_weight, args.scale_cp, None, None,
    )
    print(f"epoch=0 val={initial}", flush=True)
    history = []
    best = (float("inf"), 0)
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        order = torch.randperm(len(train_indices), generator=generator)
        train = run_epoch(
            model, train_indices, train_cp, train_actions, args.batch_size, device,
            args.policy_weight, args.scale_cp, optimizer, order,
        )
        val = run_epoch(
            model, val_indices, val_cp, val_actions, args.batch_size, device,
            args.policy_weight, args.scale_cp, None, None,
        )
        seconds = time.time() - started
        record = {"epoch": epoch, "train": train, "val": val, "seconds": seconds}
        history.append(record)
        print(f"epoch={epoch} train={train} val={val} seconds={seconds:.1f}", flush=True)
        selection_loss = val["value_mse"] if args.policy_head == "none" else val["combined"]
        if selection_loss < best[0]:
            best = (selection_loss, epoch)
            torch.save(model.evaluator.state_dict(), os.path.join(args.out, "nnue_best.pt"))
            if model.policy_head is not None:
                torch.save(model.policy_head.state_dict(), os.path.join(args.out, "policy_aux_best.pt"))

    torch.save(model.evaluator.state_dict(), os.path.join(args.out, "nnue_final.pt"))
    if model.policy_head is not None:
        torch.save(model.policy_head.state_dict(), os.path.join(args.out, "policy_aux_final.pt"))
    model_config = {
        "total_features": kp.CANONICAL_SHARED_TOTAL,
        "feature_dim": args.feature_dim,
        "hidden_dim": args.hidden_dim,
        "scale_cp": args.scale_cp,
        "objective": "cp" if args.policy_head == "none" else "cp+policy_aux",
        "clamp_cp": args.clamp_cp,
        "loss": "mse",
        "huber_delta": 0.0,
        "augment_mirror": False,
        "dropout": 0.0,
        "canonical": False,
        "canonical_shared": True,
        "canonical_shared_ka": False,
        "canonical_shared_kav2": False,
        "kpp_zone": False,
        "kpp_zone_total": 0,
        "qat": True,
        "qat_fake_quant": True,
        "qat_tie": True,
        "wdl_weight": 0.0,
        "wdl_lambda": 1.0,
        "policy_aux": args.policy_head != "none",
        "policy_weight": args.policy_weight,
        "policy_head": args.policy_head,
        "policy_hidden": args.policy_hidden,
        "policy_stem": args.policy_stem if args.policy_head != "none" else None,
        "num_policy_actions": NUM_ACTIONS,
        "arch": (
            "QATSharedNNUE(BNNQ)"
            if args.policy_head == "none"
            else f"QATSharedNNUE(BNNQ)+training-only {args.policy_head} policy auxiliary"
        ),
    }
    with open(os.path.join(args.out, "model_config.json"), "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2)
    with open(os.path.join(args.out, "history.json"), "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "initial": initial,
                   "best_epoch": best[1], "history": history}, f, indent=2)
    print(f"POLICY_AUX_DONE best_epoch={best[1]} out={args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
