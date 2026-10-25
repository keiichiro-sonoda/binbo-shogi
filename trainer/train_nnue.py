#!/usr/bin/env python3
"""Train the minimal NNUE on a cached teacher-data split.

Loads the .npy cache produced by `prepare_data.py`, trains `MinimalNNUE`
(unchanged architecture from Task 014), and records per-epoch train/val loss plus
interpretable validation metrics.

Three training objectives (Task 033 added `cp`; Task 051 added `wdl`):

- `cp` (default): regress Gikou's evaluation in centipawns directly. The target is
  the raw cp clamped to +/-`--clamp-cp` (extreme mate-ish values are the search's
  job, not the eval's). The loss is Huber (`--huber-delta`) or MSE. The model
  output (a small "logit") is mapped to cp by *`--scale-cp`* (default 600), the
  SAME factor the Rust search uses (`EVAL_SCALE`), so no inference change is
  needed: `forward()*600 ~= cp`. This is the fix for the "magnitude compression"
  diagnosed in Task 031/032: the winrate target below saturates on decisive
  positions and gives the model no pressure to output large evaluations.

- `winrate` (Task 017 original): target `sigmoid(eval_cp/scale_cp)`, loss
  BCEWithLogitsLoss. Kept for comparison/back-compat.

- `wdl` (Task 051): the standard Stockfish/yaneuraou-style winrate-space target.
  The target is `lambda*sigmoid(eval_cp/scale_cp) + (1-lambda)*game_result`
  (`--wdl-lambda`, `--scale-cp` is the steepness; `lambda<1` mixes the actual
  outcome from `*_wdl.npy`), and the loss is MSE on `sigmoid(forward)`. Unlike the
  Task 050 additive `--wdl-weight` term, the ENTIRE loss lives in winrate space, so
  large evals that saturate in winrate contribute little gradient -- capacity is
  spent on the near-0 region that decides games. No inference change: `forward()`
  is still used as the ordinal eval (`forward*600`), order-preserving for search.

The verdict is NOT on val loss alone (Task 017 showed val loss can look good while
the magnitude is compressed). Use `nnue-diag`'s decisive/marginal resolution as
the primary metric after training; this script additionally reports a cp MAE and
the sign-agreement rate on decisive positions.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
import warnings

import numpy as np
import torch
from torch import nn

try:
    from . import kp
    from .nnue import (DEFAULT_EVAL_SCALE_CP, CanonicalNNUE, KppNNUE, MinimalNNUE,
                       QATSharedNNUE, eval_to_winrate)
except ImportError:
    import kp
    from nnue import (DEFAULT_EVAL_SCALE_CP, CanonicalNNUE, KppNNUE, MinimalNNUE,
                      QATSharedNNUE, eval_to_winrate)


def load_split(cache_dir: str, name: str, mmap: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    # mmap (Task 106): the indices stay on disk and are paged in per batch, so a 64M
    # cache (~20 GB) trains inside a 22 GB WSL. Values are identical to the in-RAM load.
    indices = np.load(os.path.join(cache_dir, f"{name}_indices.npy"), mmap_mode="r" if mmap else None)
    evals = np.load(os.path.join(cache_dir, f"{name}_evals.npy"))
    if mmap:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # read-only memmap; never written to
            idx_t = torch.from_numpy(indices)
    else:
        idx_t = torch.from_numpy(indices)
    return (
        idx_t,  # int32 (n, 77)
        torch.from_numpy(evals.astype(np.float32)),  # float32 (n,) raw cp
    )


class PowerLoss(nn.Module):
    """mean |pred - target|^power (Task 131: win-rate-space power loss, as nnue-pytorch)."""

    def __init__(self, power: float, reduction: str = "mean") -> None:
        super().__init__()
        self.power = power
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = (pred - target).abs().pow(self.power)
        return loss if self.reduction == "none" else loss.mean()


def pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = a.norm() * b.norm()
    if denom.item() == 0.0:
        return 0.0
    return (a * b).sum().item() / denom.item()


def run_batches(
    model: MinimalNNUE,
    indices: torch.Tensor,
    targets: torch.Tensor,
    batch_size: int,
    features_per_pos: int,
    order: torch.Tensor | None,
    *,
    train: bool,
    objective: str,
    criterion: nn.Module,
    scale_cp: float,
    optimizer: torch.optim.Optimizer | None,
    mirror_perm: torch.Tensor | None = None,
    gen: torch.Generator | None = None,
    device: str = "cpu",
    canonical: bool = False,
    shared: bool = False,
    wdl_target: torch.Tensor | None = None,
    wdl_weight: float = 0.0,
    start_at: int = 0,
    loss_so_far: float = 0.0,
    on_batch=None,
    row_weight: torch.Tensor | None = None,
    async_loss: bool = False,
    end_at: int | None = None,
) -> tuple[float, torch.Tensor]:
    """Run one pass. For `cp`, the model output is scaled to cp (`logit*scale_cp`)
    and compared to the clamped-cp target; `preds` (val only) are predicted cp.
    For `winrate`, the loss is on raw logits (BCEWithLogits) and `preds` are the
    sigmoid winrate. For `wdl`, the loss is MSE on `sigmoid(logits)` against the
    winrate target and `preds` are that winrate. Returns (mean loss, preds).

    Resume (Task 106): `order` may be shorter than `indices` (a row mask); the mean
    is over the rows visited. `start_at`/`loss_so_far` continue an interrupted pass
    and `on_batch(next_start, total_loss)` is called after every train batch (it
    may checkpoint, or raise to pause).

    Row weights (Task 117): `row_weight` (train only, cp objective) scales each
    row's loss; the batch loss is sum(w * loss_row) / b, so mean-1 weights keep the
    overall step size."""
    n = indices.shape[0]
    if row_weight is not None:
        unreduced = copy.copy(criterion)
        unreduced.reduction = "none"
    m = order.shape[0] if order is not None else n
    if end_at is not None:
        m = min(m, end_at)
    total_loss = loss_so_far
    # Task 135: Python accumulated float32 loss.item() values in float64. Keep
    # that precision/order on device, and materialize only at a checkpoint/end.
    if async_loss:
        total_loss = torch.tensor(loss_so_far, dtype=torch.float64, device=device)
    resident = getattr(indices, "resident", False)
    preds = torch.empty(n, dtype=torch.float32) if not train else torch.empty(0)
    pred_cursor = 0
    for start in range(start_at, m, batch_size):
        rows = order[start : start + batch_size] if order is not None else torch.arange(
            start, min(start + batch_size, n)
        )
        if resident:
            rows = rows.to(device, non_blocking=True)
        batch_idx = indices[rows].long()
        b = batch_idx.shape[0]
        # Mirror augmentation (train only): flip each position L-R with prob 0.5 by
        # mapping its KP indices through `mirror_perm`. Same target cp (the flip is
        # a value-preserving equivalence). Val is never flipped (measured on the
        # original orientation), so there is no train/val pair leakage.
        # Mirror augmentation done on CPU (cheap gather) before moving to device.
        if train and mirror_perm is not None:
            flip = torch.rand(b, generator=gen) < 0.5
            if resident:
                batch_idx = torch.where(flip.to(device, non_blocking=True).unsqueeze(1),
                                        mirror_perm[batch_idx], batch_idx)
            elif bool(flip.any()):
                batch_idx = torch.where(flip.unsqueeze(1), mirror_perm[batch_idx], batch_idx)
        target = targets[rows].to(device)

        if shared:
            # Canonical-shared / HalfKA cache layout: cols [0:own_n] = own (body +
            # side), [own_n:fpp] = opp (body). own_n=(fpp-1)//2+1, opp_n=(fpp-1)//2.
            # Both halves index the single shared band. shared HalfKP fpp=77 -> 39/38;
            # HalfKA fpp=81 -> 41/40 (Task 052).
            own_n = (features_per_pos - 1) // 2 + 1
            opp_n = features_per_pos - own_n
            own_cols = batch_idx[:, :own_n]
            opp_cols = batch_idx[:, own_n:features_per_pos]
            own_flat = own_cols.reshape(-1).to(device)
            opp_flat = opp_cols.reshape(-1).to(device)
            own_off = torch.arange(0, b * own_n, own_n, dtype=torch.long, device=device)
            opp_off = torch.arange(0, b * opp_n, opp_n, dtype=torch.long, device=device)
            model_args = (own_flat, own_off, opp_flat, opp_off)
        elif canonical:
            # Task 046 two-band: sorted 77 features have fixed blocks: cols [0:38]
            # own-half body (< VIEW_DIM), [38:76] opp-half body, [76] side feature.
            # own = own body + side (39); opp = opp body (38). Mirror preserves view
            # membership and EmbeddingBag-sum is order-free, so the split stays valid.
            own_cols = torch.cat([batch_idx[:, :38], batch_idx[:, 76:77]], dim=1)
            opp_cols = batch_idx[:, 38:76]
            own_flat = own_cols.reshape(-1).to(device)
            opp_flat = opp_cols.reshape(-1).to(device)
            own_off = torch.arange(0, b * 39, 39, dtype=torch.long, device=device)
            opp_off = torch.arange(0, b * 38, 38, dtype=torch.long, device=device)
            model_args = (own_flat, own_off, opp_flat, opp_off)
        else:
            flat = batch_idx.reshape(-1).to(device)
            offsets = torch.arange(0, b * features_per_pos, features_per_pos, dtype=torch.long, device=device)
            model_args = (flat, offsets)

        if train:
            optimizer.zero_grad(set_to_none=True)
            logits = model(*model_args)
            pred = (logits * scale_cp if objective == "cp"
                    else torch.sigmoid(logits) if objective == "wdl" else logits)
            if row_weight is not None:
                loss = (unreduced(pred, target) * row_weight[rows].to(device)).sum() / b
            else:
                loss = criterion(pred, target)
            if wdl_target is not None and wdl_weight > 0.0:
                # Mix in the actual game outcome (win/draw/loss as a winrate target)
                # via BCE on sigmoid(logits). Un-compressible ground truth that the
                # cp regression alone lacks (Task 050).
                wdl_t = wdl_target[rows].to(device)
                wdl_loss = nn.functional.binary_cross_entropy_with_logits(logits, wdl_t)
                loss = (1.0 - wdl_weight) * loss + wdl_weight * wdl_loss
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                logits = model(*model_args)
                pred = (logits * scale_cp if objective == "cp"
                    else torch.sigmoid(logits) if objective == "wdl" else logits)
                loss = criterion(pred, target)
                preds[pred_cursor : pred_cursor + b] = (
                    pred if objective != "winrate" else torch.sigmoid(logits)
                ).detach().cpu()
                pred_cursor += b
        if async_loss:
            total_loss += loss.detach().to(torch.float64) * b
        else:
            total_loss += loss.item() * b
        if train and on_batch is not None:
            on_batch(start + batch_size, total_loss)
    return (total_loss.item() if async_loss else total_loss) / m, preds


def run_batches_kpp(
    model, kp_idx, kpp, targets, batch_size, order, *,
    train, objective, criterion, scale_cp, optimizer,
    kp_mirror=None, kpp_mirror=None, gen=None, device="cpu",
):
    """KPP variant (Task 053): HalfKP shared KP row (fixed 77) + king-zone KPP
    ragged CSR (own/opp). Builds the four accumulators (KP own/opp, KPP own/opp) and
    runs `KppNNUE`. Mirror augmentation (train) flips KP rows via `kp_mirror` and KPP
    pair indices via `kpp_mirror` under the SAME per-position flip mask."""
    own_flat, own_off, opp_flat, opp_off = kpp
    n = kp_idx.shape[0]
    total_loss = 0.0
    preds = torch.empty(n, dtype=torch.float32) if not train else torch.empty(0)
    pcur = 0
    for start in range(0, n, batch_size):
        rows = order[start:start + batch_size] if order is not None else torch.arange(
            start, min(start + batch_size, n))
        rows_np = rows.numpy()
        b = rows_np.shape[0]
        kp_b = kp_idx[rows].long()
        target = targets[rows].to(device)
        if train and kp_mirror is not None:
            flip = torch.rand(b, generator=gen) < 0.5
            kp_b = torch.where(flip.unsqueeze(1), kp_mirror[kp_b], kp_b)
            flip_np = flip.numpy()
        else:
            flip_np = np.zeros(b, dtype=bool)
        kp_own = kp_b[:, :39].reshape(-1).to(device)
        kp_own_off = torch.arange(0, b * 39, 39, dtype=torch.long, device=device)
        kp_opp = kp_b[:, 39:77].reshape(-1).to(device)
        kp_opp_off = torch.arange(0, b * 38, 38, dtype=torch.long, device=device)

        def build_kpp(flat, off):
            parts, offs, c = [], np.empty(b, dtype=np.int64), 0
            for i in range(b):
                r = rows_np[i]
                seg = flat[off[r]:off[r + 1]]
                if flip_np[i]:
                    seg = kpp_mirror[seg]
                offs[i] = c
                c += seg.shape[0]
                parts.append(seg)
            # CSR concat: exactly c elements (empty bags contribute nothing, so the
            # per-bag offsets stay correct and empty bags sum to 0 in EmbeddingBag).
            allf = np.concatenate(parts).astype(np.int64)
            return (torch.from_numpy(allf).to(device), torch.from_numpy(offs).to(device))

        pp_own_f, pp_own_o = build_kpp(own_flat, own_off)
        pp_opp_f, pp_opp_o = build_kpp(opp_flat, opp_off)
        args = (kp_own, kp_own_off, kp_opp, kp_opp_off,
                pp_own_f, pp_own_o, pp_opp_f, pp_opp_o)

        def pred_from(logits):
            return (logits * scale_cp if objective == "cp"
                    else torch.sigmoid(logits) if objective == "wdl" else logits)

        if train:
            optimizer.zero_grad(set_to_none=True)
            logits = model(*args)
            loss = criterion(pred_from(logits), target)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                logits = model(*args)
                pred = pred_from(logits)
                loss = criterion(pred, target)
                preds[pcur:pcur + b] = (
                    pred if objective != "winrate" else torch.sigmoid(logits)
                ).detach().cpu()
                pcur += b
        total_loss += loss.item() * b
    return total_loss / n, preds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache_dir", help="dir produced by prepare_data.py")
    parser.add_argument("--manifest", help="verified shared HalfKP multi-cache manifest (one val)")
    parser.add_argument("--chunk-manifests", default=argparse.SUPPRESS,
                        help="comma-separated manifests; resident one at a time in CLI order every epoch; "
                             "same val record required; one chunk is an exact --manifest alias")
    parser.add_argument("--max-steps", type=int, default=None, help="stop after N total optimizer updates; save final weights")
    parser.add_argument("--out", default=None, help="dir to save weights/history (default cache_dir/model)")
    parser.add_argument("--init-weights", default=None,
                        help="warm-start (Task 066 self-reinforcement): load this .pt state_dict into the "
                             "model before training (must be the SAME arch). Default None = random init "
                             "(numerically identical to prior behavior).")
    parser.add_argument("--objective", choices=["cp", "winrate", "wdl"], default="cp",
                        help="cp: regress clamped centipawns (Task 033); winrate: sigmoid BCE (Task 017); "
                             "wdl: standard winrate-space MSE on sigmoid(eval/scale_cp), optionally mixed "
                             "with the actual game result via --wdl-lambda (Task 051)")
    parser.add_argument("--clamp-cp", type=float, default=2000.0, help="cp objective: clamp target to +/- this")
    parser.add_argument("--loss", choices=["huber", "mse"], default="huber", help="cp objective loss")
    parser.add_argument("--huber-delta", type=float, default=600.0, help="cp objective: Huber delta (cp units)")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--feature-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--hidden-dim2", type=int, default=None,
                        help="Task 149: QAT second hidden width (default: --hidden-dim)")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Adam L2 weight decay")
    parser.add_argument("--lr-schedule", choices=["none", "cosine", "step"], default="none",
                        help="per-epoch LR decay (Task 061). none=constant lr (default, unchanged). "
                             "cosine=CosineAnnealingLR(T_max=epochs, eta_min=lr*lr-min-frac). "
                             "step=StepLR(step=max(1,epochs//3), gamma=lr-step-gamma). Targets the "
                             "epoch-1-then-overfit pattern: keep the fast early descent, then refine.")
    parser.add_argument("--lr-min-frac", type=float, default=0.01,
                        help="cosine schedule: eta_min = lr * this (default 0.01)")
    parser.add_argument("--lr-step-gamma", type=float, default=0.3,
                        help="step schedule: multiply lr by this every epochs//3 epochs (default 0.3)")
    parser.add_argument("--scale-cp", type=float, default=DEFAULT_EVAL_SCALE_CP,
                        help="output->cp factor; MUST match Rust EVAL_SCALE (600)")
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--limit-train", type=int, default=0, help="cap train positions (timing/sweep)")
    # Task 106: large caches and interruptible runs. All off by default (= prior behaviour).
    parser.add_argument("--cp-result-lambda", type=float, default=1.0,
                        help="cp objective (Task 112): on rows whose game result is a win or a loss, replace the "
                             "target by scale*logit(lambda*sigmoid(eval/scale) + (1-lambda)*result), i.e. mix "
                             "the result in win-rate space but keep the cp loss. Rows stored as 0 (draw or "
                             "unknown) keep the eval target exactly. 1.0 = off (default, prior behaviour)")
    parser.add_argument("--wdl-known-only", action="store_true",
                        help="wdl objective with lambda<1 (Task 111): mix the game result only into rows whose "
                             "result is a win or a loss; rows stored as 0 (draw OR unknown: max-ply / engine "
                             "crash in gen-teacher) keep the pure eval target")
    parser.add_argument("--mmap", action="store_true",
                        help="memory-map the indices cache instead of loading it into RAM (64M cache)")
    parser.add_argument("--compact-cache", default=None,
                        help="Task 135: verified compact HalfKP cache DIR, kept on the training device")
    parser.add_argument("--sparse-ft", action="store_true",
                        help="Task 135: QAT effective FT for touched rows only (dense gradients/Adam)")
    parser.add_argument("--fused-adam", action="store_true",
                        help="Task 135: torch Adam(fused=True), still updates every row each step")
    parser.add_argument("--async-loss", action="store_true",
                        help="Task 135: device float64 loss sum; sync at checkpoints/epoch end")
    parser.add_argument("--fast", action="store_true",
                        help="Task 135: enable sparse FT, fused Adam, async loss; requires --compact-cache")
    parser.add_argument("--row-mask", default=None,
                        help=".npy bool over the (limited) train rows; False rows are never trained on")
    parser.add_argument("--vm", action="store_true",
                        help="Task 132: HalfKP_vm as weight tying -- a king on files 6-9 shares the row of its "
                             "left-right mirror (needs --qat-tie; the export stays plain HalfKP)")
    parser.add_argument("--factorize", action="store_true",
                        help="QAT only (Task 125): add a king-independent P table to the feature "
                             "transformer during training; folded into the real rows at export")
    parser.add_argument("--row-weight", default=None,
                        help=".npy float over the (limited) train rows: per-row loss weight (cp objective; "
                             "normalised to mean 1 over the trained rows)")
    parser.add_argument("--save-epochs", action="store_true",
                        help="Task 138: also save every epoch's weights as nnue_epoch<N>.pt (for weight averaging)")
    parser.add_argument("--checkpoint-minutes", type=float, default=0.0,
                        help="save a resumable train_state.pt every N minutes (and at every epoch end)")
    parser.add_argument("--resume", action="store_true",
                        help="continue from <out>/train_state.pt if present (same args required)")
    parser.add_argument("--stop-file", default=None,
                        help="if this file appears, save train_state.pt and exit with code 75 (paused)")
    parser.add_argument("--decisive-cp", type=float, default=500.0, help="|cp|>= this counts as a decisive position")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                        help="training device; auto=cuda if available (Task 039). Inference is always CPU.")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="dropout on the accumulator (embedding output); train-only (Task 038)")
    parser.add_argument("--augment-mirror", action="store_true",
                        help="left-right mirror data augmentation (train only; Task 036)")
    parser.add_argument("--mirror-cache", default="/tmp/binbo-task033/mirror_perm.npy",
                        help="path to cache the mirror permutation (.npy)")
    parser.add_argument("--canonical", action="store_true",
                        help="canonical HalfKP two-band (Task 046, NON-shared W1). "
                             "Requires a cache built with prepare_data --canonical.")
    parser.add_argument("--canonical-shared", action="store_true",
                        help="TRUE canonical shared-W1 HalfKP (Task 047): single band, "
                             "embedding ~half size. Requires prepare_data --canonical-shared.")
    parser.add_argument("--canonical-shared-ka", action="store_true",
                        help="HalfKA shared band (Task 052): kings included as features. "
                             "Same 2-accumulator path as --canonical-shared but a larger "
                             "table (VIEW_DIM_KA+2). Requires prepare_data --canonical-shared-ka.")
    parser.add_argument("--canonical-shared-kav2", action="store_true",
                        help="HalfKAv2 (Task 053): HalfKA minus each half's redundant anchor "
                             "king. Same KA table. Requires prepare_data --canonical-shared-kav2.")
    parser.add_argument("--kpp-zone", action="store_true",
                        help="HalfKP shared + king-zone KPP (Task 053): 4-accumulator KppNNUE. "
                             "Requires a cache built with prepare_data --kpp-zone.")
    parser.add_argument("--qat", action="store_true",
                        help="quantization-aware shared HalfKP (Task 058): QATSharedNNUE with "
                             "ClippedReLU + explicit FT bias + fake-quant to YaneuraOu's int grids, "
                             "so the trained eval survives quantization (exports as BNNQ). Implies "
                             "--canonical-shared (same own/opp cache). Use fd256/hidden32 to match the "
                             "256x2-32-32 arch; --scale-cp folds into the output int scale.")
    parser.add_argument("--qat-no-fakequant", action="store_true",
                        help="QAT structure (ClippedReLU + FT bias) but NO fake-quant: the continuous "
                             "float model used to observe post-training-quantization residual before QAT.")
    parser.add_argument("--qat-tie", action="store_true",
                        help="tie features that share a YaneuraOu HalfKP slot (gold collapse) + drop the "
                             "side feature (Task 059), so the trained eval is bit-exact writable to nn.bin "
                             "via trainer/binbo_to_yane.py. Requires --qat.")
    parser.add_argument("--eval-npy", default=None,
                        help="teacher-label swap (Task 055): dir with {train,val}_dlshogi_evals.npy "
                             "(cp, aligned to the cache positions) to use as the cp target instead of "
                             "the cache's Gikou evals. Everything else (positions/arch/search/objective) unchanged.")
    parser.add_argument("--wdl-weight", type=float, default=0.0,
                        help="mix in the game outcome (WDL) as a BCE term on the train loss "
                             "(Task 050): loss = (1-w)*cp + w*BCE(sigmoid(forward), outcome). "
                             "Requires a cache with *_wdl.npy (regenerated teacher data).")
    parser.add_argument("--wdl-power", type=float, default=2.0,
                        help="objective=wdl only (Task 131): loss = mean |sigmoid(pred) - target|^power "
                             "(2.0 = the MSE of Task 051; Stockfish's nnue-pytorch uses about 2.5)")
    parser.add_argument("--wdl-lambda", type=float, default=1.0,
                        help="objective=wdl only (Task 051): interpolate the target winrate as "
                             "lambda*sigmoid(eval/scale_cp) + (1-lambda)*game_result. lambda=1 uses the "
                             "eval winrate only; lambda<1 mixes the actual outcome (needs *_wdl.npy).")
    args = parser.parse_args()
    # Remove the new CLI field entirely to retain legacy run identities/history.
    chunk_arg = vars(args).pop('chunk_manifests', None)
    chunk_paths = None
    chunks = None
    if chunk_arg is not None:
        if args.manifest or args.compact_cache:
            parser.error("--chunk-manifests is mutually exclusive with --manifest/--compact-cache")
        chunk_paths = [p.strip() for p in chunk_arg.split(',')]
        if not all(chunk_paths):
            parser.error("--chunk-manifests requires nonempty comma-separated paths")
        if len(chunk_paths) > 1 and (args.row_mask or args.row_weight):
            parser.error("multiple chunks require masks in their manifests; --row-mask/--row-weight unsupported")
        args.manifest = chunk_paths[0]
        try:
            from . import chunked_cache
        except ImportError:
            import chunked_cache
    if args.hidden_dim2 is not None and (not args.qat or args.hidden_dim2 < 1):
        parser.error("--hidden-dim2 needs --qat and a positive width")
    if args.epochs < 0:
        parser.error("--epochs must be nonnegative (0: Task 149 initialization-only check)")
    if args.max_steps is not None and (args.max_steps < 1 or args.kpp_zone or args.epochs < 1):
        parser.error("--max-steps requires N>0, epochs>0 and a KP model")
    if args.manifest and (args.compact_cache or args.eval_npy or args.limit_train):
        parser.error("--manifest replaces --compact-cache/--eval-npy/--limit-train")
    if args.fast:
        if not (args.compact_cache or args.manifest):
            parser.error("--fast requires --compact-cache DIR (convert it first)")
        args.sparse_ft = args.fused_adam = args.async_loss = True
    if args.sparse_ft and not args.qat:
        parser.error("--sparse-ft requires --qat")
    kpp_zone = args.kpp_zone
    qat = args.qat
    if args.vm and not args.qat_tie:
        raise SystemExit("--vm needs --qat --qat-tie (it rides on the tie collapse map)")
    if args.factorize and not qat:
        raise SystemExit("--factorize needs --qat (QATSharedNNUE)")
    kav2 = args.canonical_shared_kav2
    ka = args.canonical_shared_ka
    ka_table = ka or kav2  # both use the larger KA table (VIEW_DIM_KA+2)
    # QAT (Task 058) uses the same shared own/opp cache layout as --canonical-shared.
    shared = args.canonical_shared or ka_table or kpp_zone or qat

    out_dir = args.out or os.path.join(args.cache_dir, "model")
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    with open(os.path.join(args.cache_dir, "meta.json"), encoding="utf-8") as f:
        cache_meta = json.load(f)
    features_per_pos = cache_meta["features_per_position"]
    if (args.compact_cache or args.manifest) and (not shared or ka_table or kpp_zone or features_per_pos != 77
                               or cache_meta["total_features"] != kp.CANONICAL_SHARED_TOTAL):
        parser.error("--compact-cache requires a shared HalfKP 77-column cache/model")

    manifest = None
    if args.manifest:
        try:
            from .multi_cache import Manifest
        except ImportError:
            from multi_cache import Manifest
        manifest = Manifest(args.manifest)
        if chunk_paths and len(chunk_paths) > 1:
            chunks = [manifest, *[Manifest(p) for p in chunk_paths[1:]]]
            if any(m.val != manifest.val for m in chunks[1:]):
                parser.error("all chunk manifests must share the same val record")
            if any(not m.vector('mask').any() for m in chunks):
                parser.error("each chunk must keep at least one training row")
            print(f"chunk_manifests={chunk_paths} order=fixed "
                  f"rows={[m.rows for m in chunks]} total_rows={sum(m.rows for m in chunks)}")
        if os.path.realpath(args.cache_dir) != manifest.val['cache']:
            parser.error("cache_dir must be the manifest shared val cache")
        class RowSpace:
            shape = (manifest.rows, 77)
        train_idx = RowSpace()
        train_eval = torch.from_numpy(manifest.vector('labels'))
        val_idx, val_eval = load_split(manifest.val['cache'], "val")
        val_eval = torch.from_numpy(np.load(manifest.val['labels']).astype(np.float32))
    else:
        train_idx, train_eval = load_split(args.cache_dir, "train", mmap=args.mmap or bool(args.compact_cache))
        val_idx, val_eval = load_split(args.cache_dir, "val")
    # Teacher-label swap (Task 055): replace the cp targets with externally re-scored
    # evals (e.g. dlshogi distillation), aligned to the cache positions. Same KP
    # representation / positions / search / objective -- only the teacher changes.
    if args.eval_npy:
        d = args.eval_npy
        tr = np.load(os.path.join(d, "train_dlshogi_evals.npy"))
        va = np.load(os.path.join(d, "val_dlshogi_evals.npy"))
        if tr.shape[0] != train_idx.shape[0] or va.shape[0] != val_idx.shape[0]:
            raise SystemExit(f"--eval-npy shape mismatch: train {tr.shape[0]}!={train_idx.shape[0]} "
                             f"or val {va.shape[0]}!={val_idx.shape[0]}")
        train_eval = torch.from_numpy(tr.astype(np.float32))
        val_eval = torch.from_numpy(va.astype(np.float32))
        print(f"eval_npy={d} (teacher labels swapped; train {tr.shape[0]} val {va.shape[0]})")
    if args.limit_train and args.limit_train < train_idx.shape[0]:
        train_idx = train_idx[: args.limit_train]
        train_eval = train_eval[: args.limit_train]

    # King-zone KPP (Task 053): load the ragged CSR halves (own/opp) and build
    # per-position offset arrays for batching. KP base is the shared 77-col row.
    train_kpp = val_kpp = None
    if kpp_zone:
        def _load_csr(split, half, limit=0):
            flat = np.load(os.path.join(args.cache_dir, f"{split}_kpp_{half}_flat.npy"))
            length = np.load(os.path.join(args.cache_dir, f"{split}_kpp_{half}_len.npy"))
            if limit and limit < length.shape[0]:
                length = length[:limit]
                flat = flat[: int(length.sum())]
            off = np.zeros(length.shape[0] + 1, dtype=np.int64)
            np.cumsum(length, out=off[1:])
            return flat, off
        lim = args.limit_train
        train_kpp = (*_load_csr("train", "own", lim), *_load_csr("train", "opp", lim))
        val_kpp = (*_load_csr("val", "own"), *_load_csr("val", "opp"))
        print(f"kpp_zone=on KPP_ZONE_TOTAL={kp.KPP_ZONE_TOTAL} "
              f"train_pairs(own/opp)={train_kpp[0].shape[0]}/{train_kpp[2].shape[0]}")

    objective = args.objective
    scale_cp = args.scale_cp

    # WDL (game outcome) in winrate space: win=1, draw=0.5, loss=0. Needed by the
    # Task 050 additive term (--wdl-weight) and the Task 051 `wdl` objective with
    # lambda<1 (mix the eval winrate with the actual result).
    def _load_wdl_winrate(split, n_limit=0):
        p = os.path.join(args.cache_dir, f"{split}_wdl.npy")
        if not os.path.exists(p):
            raise SystemExit(f"WDL target requested but {p} missing (regenerate cache with WDL teacher data)")
        raw = (manifest.vector("wdl") if manifest is not None and split == "train"
               else np.load(p).astype(np.float32))
        if n_limit and n_limit < raw.shape[0]:
            raw = raw[:n_limit]
        return raw, torch.from_numpy((raw + 1.0) / 2.0)  # -1/0/1 -> 0/0.5/1

    need_result = (args.wdl_weight > 0.0 or (objective == "wdl" and args.wdl_lambda < 1.0)
                   or (objective == "cp" and args.cp_result_lambda < 1.0))
    train_wdl = None        # additive-term path (passed to run_batches, Task 050)
    train_result = val_result = None
    if need_result:
        train_raw, train_result = _load_wdl_winrate("train", args.limit_train)
        _, val_result = _load_wdl_winrate("val")
        train_wdl = train_result  # same winrate tensor reused by the additive path
        print(f"wdl win/draw/loss={float((train_raw>0).mean()):.2f}/"
              f"{float((train_raw==0).mean()):.2f}/{float((train_raw<0).mean()):.2f}")

    if objective == "cp":
        train_target = train_eval.clamp(-args.clamp_cp, args.clamp_cp)
        val_target = val_eval.clamp(-args.clamp_cp, args.clamp_cp)
        if args.cp_result_lambda < 1.0:
            lam = args.cp_result_lambda

            def mix_cp(evals, result, target):
                known = result != 0.5                   # 0.5 = draw OR unknown in gen-teacher data
                mixed = lam * eval_to_winrate(evals, scale_cp) + (1.0 - lam) * result
                mixed_cp = scale_cp * torch.logit(mixed.clamp(1e-6, 1 - 1e-6))
                out = target.clone()
                out[known] = mixed_cp[known].clamp(-args.clamp_cp, args.clamp_cp)
                return out, float(known.float().mean())

            train_target, frac = mix_cp(train_eval, train_result, train_target)
            val_target, vfrac = mix_cp(val_eval, val_result, val_target)
            print(f"cp_result_lambda={lam:g}: result mixed (win-rate space, cp loss) into {frac:.3f} of train "
                  f"rows / {vfrac:.3f} of val rows; the rest keep the eval target")
        criterion: nn.Module = (
            nn.HuberLoss(delta=args.huber_delta) if args.loss == "huber" else nn.MSELoss()
        )
    elif objective == "wdl":
        # Standard winrate-space target: lambda*sigmoid(eval/scale) + (1-lambda)*result.
        # Loss is MSE on sigmoid(forward) (computed in run_batches), de-emphasising
        # large evals that saturate in winrate (Task 051).
        lam = args.wdl_lambda
        train_target = lam * eval_to_winrate(train_eval, scale_cp)
        val_target = lam * eval_to_winrate(val_eval, scale_cp)
        if lam < 1.0:
            if args.wdl_known_only:
                # gen-teacher stores draw and unknown (max-ply, crash) alike as 0 -> 0.5
                # winrate; mixing those would pull real positions toward "even".
                tr_known = (train_result != 0.5).float()
                va_known = (val_result != 0.5).float()
                tr_mix = (1.0 - lam) * tr_known
                va_mix = (1.0 - lam) * va_known
                train_target = (1.0 - tr_mix) * eval_to_winrate(train_eval, scale_cp) + tr_mix * train_result
                val_target = (1.0 - va_mix) * eval_to_winrate(val_eval, scale_cp) + va_mix * val_result
                print(f"wdl_known_only: mixing lambda={lam:g} into {float(tr_known.mean()):.3f} of train rows "
                      f"(the rest keep the eval target)")
            else:
                train_target = train_target + (1.0 - lam) * train_result
                val_target = val_target + (1.0 - lam) * val_result
        criterion = nn.MSELoss() if args.wdl_power == 2.0 else PowerLoss(args.wdl_power)
    else:  # winrate (Task 017 BCE)
        train_target = eval_to_winrate(train_eval, scale_cp)
        val_target = eval_to_winrate(val_eval, scale_cp)
        criterion = nn.BCEWithLogitsLoss()

    n_train = train_idx.shape[0]
    n_val = val_idx.shape[0]
    keep_rows = None
    if args.row_mask or manifest is not None:
        mask = manifest.vector("mask") if manifest is not None else np.load(args.row_mask)
        if args.row_mask and manifest is not None:
            external_mask = np.load(args.row_mask)
            if external_mask.shape != mask.shape or external_mask.dtype != np.bool_:
                raise ValueError("external row mask shape/dtype mismatch")
            mask &= external_mask
        if mask.shape != (n_train,) or mask.dtype != np.bool_:
            raise SystemExit(f"--row-mask must be bool ({n_train},), got {mask.dtype} {mask.shape}")
        # An all-true manifest must preserve the legacy shuffle/RNG sequence.
        if args.row_mask or not mask.all():
            keep_rows = torch.from_numpy(np.flatnonzero(mask).astype(np.int64))
        kept = int(mask.sum())
        print(f"row_mask={args.row_mask} keep={kept}/{n_train} ({kept / n_train:.4f})")

    if keep_rows is not None and not len(keep_rows):
        raise ValueError("empty training mask")

    row_weight = None
    if args.row_weight:
        if objective != "cp" or args.wdl_weight > 0.0 or kpp_zone:
            raise SystemExit("--row-weight supports the plain cp objective only")
        w = np.load(args.row_weight).astype(np.float32)
        if w.shape != (n_train,) or not np.isfinite(w).all() or (w < 0).any():
            raise SystemExit(f"--row-weight must be finite, >= 0, ({n_train},), got {w.dtype} {w.shape}")
        trained = w[keep_rows.numpy()] if keep_rows is not None else w
        w = w / float(trained.mean())
        row_weight = torch.from_numpy(w)
        tw = trained / float(trained.mean())
        print(f"row_weight={args.row_weight} (normalised over trained rows) min={tw.min():.3f} "
              f"p50={np.median(tw):.3f} p90={np.percentile(tw, 90):.3f} max={tw.max():.3f}")

    print(f"cache_dir={args.cache_dir}")
    print(f"n_train={n_train} n_val={n_val} features_per_pos={features_per_pos}")
    print(f"val_shards={cache_meta['val_shards']}")
    loss_label = (args.loss if objective == "cp" else "bce" if objective == "winrate"
                  else "mse_winrate" if args.wdl_power == 2.0 else f"power{args.wdl_power:g}_winrate")
    print(f"objective={objective} loss={loss_label} wdl_lambda={args.wdl_lambda:g} "
          f"clamp_cp={args.clamp_cp:g} huber_delta={args.huber_delta:g} scale_cp={scale_cp:g}")
    print(f"batch_size={args.batch_size} epochs={args.epochs} lr={args.lr} weight_decay={args.weight_decay:g}")
    print(f"torch_threads={torch.get_num_threads()}")
    print(
        f"model=EmbeddingBag({kp.TOTAL_FEATURES},{args.feature_dim},sum)"
        f"->Linear({args.feature_dim},{args.hidden_dim})->Linear({args.hidden_dim},{args.hidden_dim})->Linear(...,1)"
    )

    # Baseline: a constant predictor at the val target mean (in the loss's space).
    if objective != "winrate":  # cp (MSE/Huber) or wdl (MSE on the winrate target)
        const = float(val_target.mean().item())
        baseline_val_loss = float(
            criterion(torch.full_like(val_target, const), val_target).item()
        )
        unit = "cp" if objective == "cp" else "wr"
        print(f"val_target_mean_{unit}={const:.4g} baseline_val_loss(const)={baseline_val_loss:.4f}")
    else:
        p0 = float(val_target.mean().item())
        p0c = min(max(p0, 1e-7), 1 - 1e-7)
        baseline_val_loss = float(
            nn.functional.binary_cross_entropy(torch.full_like(val_target, p0c), val_target).item()
        )
        print(f"val_target_mean={p0:.6f} baseline_val_loss(const)={baseline_val_loss:.6f}")

    decisive_mask = val_eval.abs() >= args.decisive_cp
    n_decisive = int(decisive_mask.sum().item())

    mirror_perm = None
    if args.augment_mirror:
        try:
            from . import mirror as mirror_mod
        except ImportError:
            import mirror as mirror_mod
        # The mirror permutation is the SAME for naive and canonical numbering:
        # a horizontal (file) mirror commutes with the 180-degree rotation that
        # defines the canonical opponent half (80 - mirror(80 - q) == mirror(q)),
        # so mirroring the stored-frame squares yields the correct mirrored index
        # in either numbering. Verified empirically in verify_mirror.py --canonical.
        if ka_table:
            # HalfKA and HalfKAv2 share the SAME KA table, so the SAME mirror perm.
            mirror_cache = args.mirror_cache.replace(".npy", "_shared_ka.npy")
            os.makedirs(os.path.dirname(mirror_cache) or ".", exist_ok=True)
            mirror_perm = torch.from_numpy(mirror_mod.build_mirror_perm_shared_ka(mirror_cache)).long()
        elif shared:
            mirror_cache = args.mirror_cache.replace(".npy", "_shared.npy")
            os.makedirs(os.path.dirname(mirror_cache) or ".", exist_ok=True)
            mirror_perm = torch.from_numpy(mirror_mod.build_mirror_perm_shared(mirror_cache)).long()
        else:
            os.makedirs(os.path.dirname(args.mirror_cache) or ".", exist_ok=True)
            mirror_perm = torch.from_numpy(mirror_mod.build_mirror_perm(args.mirror_cache)).long()
        print(f"augment_mirror=on shared={shared} ka={ka} kav2={kav2} kpp_zone={kpp_zone}")

    kpp_mirror = None
    if kpp_zone and args.augment_mirror:
        kpp_cache = args.mirror_cache.replace(".npy", "_kpp_zone.npy")
        os.makedirs(os.path.dirname(kpp_cache) or ".", exist_ok=True)
        kpp_mirror = mirror_mod.build_kpp_zone_mirror_perm(kpp_cache)  # numpy int64

    device = "cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu"
    print(f"device={device}" + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else f" (threads={torch.get_num_threads()})"))
    if args.compact_cache or manifest is not None:
        try:
            from .compact_cache import CompactCache
        except ImportError:  # direct script run: trainer/ itself is on sys.path
            from compact_cache import CompactCache
        if chunks is None:
            train_idx = (manifest.resident(device) if manifest is not None else
                         CompactCache(args.compact_cache, "train", device, limit=n_train, source=args.cache_dir))
        val_idx = CompactCache(manifest.val['compact'] if manifest is not None else args.compact_cache,
                               "val", device, source=args.cache_dir)
        if train_idx.shape[0] != n_train or val_idx.shape[0] != n_val:
            raise SystemExit("compact cache row count mismatch")
        train_target = train_target.to(device) if chunks is None else None
        resident_val_target = val_target.to(device)
        if mirror_perm is not None:
            mirror_perm = mirror_perm.to(device)
        if train_wdl is not None and chunks is None:
            train_wdl = train_wdl.to(device)
        if row_weight is not None:
            row_weight = row_weight.to(device)
        print(f"compact_cache={args.compact_cache} sparse_ft={args.sparse_ft} "
              f"fused_adam={args.fused_adam} async_loss={args.async_loss}")
        if chunks is not None:
            # Only the shared validation set stays resident between chunks.
            train_idx = train_wdl = keep_rows = None

    two_acc = args.canonical or shared
    total_features = (kp.CANONICAL_SHARED_KA_TOTAL if ka_table
                      else kp.CANONICAL_SHARED_TOTAL if shared else kp.TOTAL_FEATURES)
    qat_fake_quant = qat and not args.qat_no_fakequant
    if kpp_zone:
        model = KppNNUE(
            kp_total=kp.CANONICAL_SHARED_TOTAL,
            kpp_total=kp.KPP_ZONE_TOTAL,
            feature_dim=args.feature_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        ).to(device)
        total_features = kp.CANONICAL_SHARED_TOTAL  # KP table size (for meta)
        print(f"model=KppNNUE(BNN6) KP({kp.CANONICAL_SHARED_TOTAL})+KPP({kp.KPP_ZONE_TOTAL}) "
              f"4x EmbeddingBag->concat({4 * args.feature_dim})->Linear->...->Linear(1)")
    elif qat:
        tie_map = None
        if args.qat_tie:
            try:
                from .binbo_to_yane import build_collapse_map
            except ImportError:
                from binbo_to_yane import build_collapse_map
            tie_map = build_collapse_map(vm=args.vm)
            n_tied = int((tie_map[:kp.VIEW_DIM] != np.arange(kp.VIEW_DIM)).sum())
            print(f"qat_tie=on collapse_map: {n_tied} features tied to representatives, side->pad")
        model = QATSharedNNUE(
            total_features=kp.CANONICAL_SHARED_TOTAL,
            feature_dim=args.feature_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            fake_quant=qat_fake_quant,
            scale_cp=scale_cp,
            tie_collapse_map=tie_map,
            factorize=args.factorize,
            sparse_ft=args.sparse_ft,
            hidden_dim2=args.hidden_dim2,
        ).to(device)
        total_features = kp.CANONICAL_SHARED_TOTAL
        if args.factorize:
            print(f"factorize=on ft_factor({kp.FEATURES_PER_KING},{args.feature_dim}) init 0, folded at export")
        print(f"model=QATSharedNNUE(BNNQ) shared EmbeddingBag({total_features},{args.feature_dim})"
              f"+ft_bias->ClippedReLU->concat({2 * args.feature_dim})->Linear->ClippedReLU->Linear->ClippedReLU->Linear(1) "
              f"hidden={model.hidden_dim}/{model.hidden_dim2} "
              f"fake_quant={qat_fake_quant} tie={args.qat_tie} scale_cp={scale_cp:g}")
    else:
        model_cls = CanonicalNNUE if two_acc else MinimalNNUE
        model = model_cls(
            total_features=total_features,
            feature_dim=args.feature_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        ).to(device)
        if two_acc:
            kind = "shared(BNN3, single band)" if shared else "two-band(BNN2)"
            print(f"model=CanonicalNNUE {kind} 2x EmbeddingBag({total_features},{args.feature_dim})"
                  f"->concat({2 * args.feature_dim})->Linear({2 * args.feature_dim},{args.hidden_dim})->...->Linear(...,1)")
    if args.dropout > 0:
        print(f"dropout={args.dropout} (accumulator, train-only)")
    # Warm-start (Task 066): load champion weights before training. Fresh optimizer (standard
    # warm-start). strict=True so an arch mismatch fails loudly instead of silently dropping rows.
    if args.init_weights:
        state = torch.load(args.init_weights, map_location=device)
        model.load_state_dict(state)
        print(f"init_weights={args.init_weights} (warm-start: loaded prior-gen weights before training)")
    optimizer = (torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=True)
                 if args.fused_adam else
                 torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay))
    # Optional per-epoch LR decay (Task 061). None => constant lr (identical to prior behavior).
    if args.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * args.lr_min_frac)
    elif args.lr_schedule == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, args.epochs // 3), gamma=args.lr_step_gamma)
    else:
        scheduler = None
    if scheduler is not None:
        print(f"lr_schedule={args.lr_schedule} lr0={args.lr:g} "
              f"(min_frac={args.lr_min_frac:g} step_gamma={args.lr_step_gamma:g})")
    generator = torch.Generator().manual_seed(args.seed)

    def do_pass(idx, kpp, tgt, order, train_flag, opt, use_mirror):
        """Dispatch one pass to run_batches_kpp (KPP) or run_batches (KP modes)."""
        if kpp_zone:
            return run_batches_kpp(
                model, idx, kpp, tgt, args.batch_size, order,
                train=train_flag, objective=objective, criterion=criterion, scale_cp=scale_cp,
                optimizer=opt, kp_mirror=(mirror_perm if use_mirror else None),
                kpp_mirror=kpp_mirror, gen=generator, device=device)
        return run_batches(
            model, idx, (resident_val_target if (args.compact_cache or manifest is not None) and not train_flag else tgt),
            args.batch_size, features_per_pos, order,
            train=train_flag, objective=objective, criterion=criterion, scale_cp=scale_cp,
            optimizer=opt, mirror_perm=(mirror_perm if use_mirror else None), gen=generator,
            device=device, canonical=args.canonical, shared=shared,
            wdl_target=(train_wdl if train_flag else None), wdl_weight=args.wdl_weight,
            async_loss=args.async_loss)

    def val_metrics(preds: torch.Tensor) -> dict:
        """preds are in target space (cp or winrate). Returns interpretable metrics."""
        mae = float((preds - val_target).abs().mean().item())
        corr = pearson(preds, val_target)
        out = {"val_mae": mae, "val_corr": corr}
        if n_decisive > 0:
            # Sign agreement on decisive positions (direction of advantage). The
            # neutral point is 0 in cp space and 0.5 in winrate space (wdl/winrate).
            center = 0.0 if objective == "cp" else 0.5
            agree = ((preds[decisive_mask] >= center) == (val_eval[decisive_mask] >= 0)).float().mean()
            out["decisive_sign_agree"] = float(agree.item())
            # Dispersion around the neutral point: did magnitude separate decisive
            # from near-equal positions? (cp magnitude, or winrate distance from 0.5)
            near_mask = val_eval.abs() < 100.0
            out["pred_abs_decisive"] = float((preds[decisive_mask] - center).abs().mean().item())
            if int(near_mask.sum().item()) > 0:
                out["pred_abs_near"] = float((preds[near_mask] - center).abs().mean().item())
        return out

    state_path = os.path.join(out_dir, "train_state.pt")

    def atomic_save(obj, path):
        tmp = path + ".tmp"
        torch.save(obj, tmp)
        os.replace(tmp, path)

    run_identity = None
    if args.manifest or args.max_steps is not None:
        run_identity = dict(vars(args))
        for key in ('resume', 'checkpoint_minutes', 'stop_file'):
            run_identity.pop(key)
        run_identity['manifest_sha256'] = manifest.identity if manifest is not None else None
        if chunks is not None:
            try:
                from .multi_cache import file_stamp
            except ImportError:
                from multi_cache import file_stamp
            run_identity['chunks'] = [file_stamp(p) for p in chunk_paths]
        if manifest is None:
            try:
                from .multi_cache import file_stamp
                from .compact_cache import source_stamp
            except ImportError:
                from multi_cache import file_stamp
                from compact_cache import source_stamp
            run_identity['source_stamps'] = [source_stamp(args.cache_dir, s) for s in ('train', 'val')]
            label_paths = ([os.path.join(args.eval_npy, f'{s}_dlshogi_evals.npy') for s in ('train', 'val')]
                           if args.eval_npy else [os.path.join(args.cache_dir, f'{s}_evals.npy') for s in ('train', 'val')])
            run_identity['label_stamps'] = [file_stamp(p) for p in label_paths]
        if args.row_mask or args.row_weight:
            try:
                from .multi_cache import file_stamp
            except ImportError:
                from multi_cache import file_stamp
            run_identity['selection_stamps'] = [file_stamp(p) for p in (args.row_mask, args.row_weight) if p]

    resume = None
    if args.resume and os.path.exists(state_path):
        # CPU: the generator states must be CPU ByteTensors; load_state_dict moves the
        # model and optimizer tensors to their parameters' device.
        resume = torch.load(state_path, map_location="cpu", weights_only=False)
        if args.manifest or args.max_steps is not None:
            if resume.get('run_identity') != run_identity:
                raise ValueError("resume arguments or manifest changed")
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        if args.fused_adam:
            # Task 135: load_state_dict restores saved group flags, including a
            # slow checkpoint's fused=None and CPU step scalars. Honor this run.
            for group in optimizer.param_groups:
                group["fused"], group["foreach"] = True, None
                for parameter in group["params"]:
                    state = optimizer.state[parameter]
                    if "step" in state:
                        state["step"] = state["step"].to(parameter.device)
        if scheduler is not None and resume["scheduler"] is not None:
            scheduler.load_state_dict(resume["scheduler"])
        generator.set_state(resume["generator"])
        if chunk_paths and 'rng' in resume:
            chunked_cache.restore_rng(resume['rng'])
        print(f"resume={state_path} epoch={resume['epoch']} cursor={resume['cursor']} "
              f"saved_at={time.strftime('%F %T', time.localtime(resume['saved_at']))}")

    if resume is None:
        # Initial (untrained) validation loss.
        model.eval()
        init_val_loss, _ = do_pass(val_idx, val_kpp, val_target, None, False, None, False)
        print(f"epoch=0 val_loss={init_val_loss:.4f} (untrained) n_decisive={n_decisive}")
        history, best_val, best_epoch, start_epoch = [], float("inf"), 0, 1
    else:
        init_val_loss = resume["init_val_loss"]
        history, best_val, best_epoch = resume["history"], resume["best_val"], resume["best_epoch"]
        start_epoch = resume["epoch"]

    updates = resume.get("updates", 0) if resume is not None else 0

    class Paused(Exception):
        pass

    def epoch_order():
        order = torch.randperm(n_train, generator=generator)
        if keep_rows is not None:
            order = keep_rows[torch.randperm(keep_rows.shape[0], generator=generator)]
        return order

    def save_state(epoch, cursor, epoch_gen_state, loss_so_far, epoch_secs, chunk_state=None):
        # The epoch's order is not stored (512 MB at 64M): the generator state at the
        # start of the epoch regenerates it, and the current state continues the
        # mirror draws exactly where they stopped.
        if args.async_loss and isinstance(loss_so_far, torch.Tensor):
            loss_so_far = loss_so_far.item()
        extra_state = {"updates": updates, "run_identity": run_identity} if run_identity is not None else {}
        if chunk_paths:
            extra_state.update(rng=chunked_cache.rng_state(device))
        if chunks is not None:
            extra_state['chunk_state'] = chunk_state
        atomic_save({
            **extra_state,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "generator": generator.get_state(), "epoch_generator": epoch_gen_state,
            "epoch": epoch, "cursor": cursor,
            "loss_so_far": loss_so_far, "epoch_secs": epoch_secs, "history": history,
            "best_val": best_val, "best_epoch": best_epoch, "init_val_loss": init_val_loss,
            "saved_at": time.time(),
        }, state_path)

    last_ckpt = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        if (args.max_steps is not None and updates >= args.max_steps and
                not (resume is not None and resume['epoch'] == epoch and
                     (resume['cursor'] > 0 or resume.get('chunk_state') is not None))):
            break
        model.train()
        if chunks is not None:
            def checkpoint_chunk(state, count):
                nonlocal updates
                updates = count
                save_state(epoch, 0, None, 0.0, state['epoch_secs'], chunk_state=state)

            try:
                train_loss, train_secs, chunk_losses, partial_epoch, updates = chunked_cache.train_epoch(
                    chunks, args=args, epoch=epoch, model=model, optimizer=optimizer,
                    generator=generator, device=device, criterion=criterion, mirror_perm=mirror_perm,
                    run_batches=run_batches, checkpoint=checkpoint_chunk, paused=Paused, updates=updates,
                    resume=resume.get('chunk_state') if resume is not None else None)
            except Paused:
                print(f"paused (stop file {args.stop_file}); resume with the same command + --resume", flush=True)
                raise SystemExit(75)
            resume = None
        else:
            if resume is not None and resume["epoch"] == epoch and resume["cursor"] > 0:
                epoch_gen_state = resume["epoch_generator"]
                generator.set_state(epoch_gen_state)
                order = epoch_order()
                generator.set_state(resume["generator"])
                cursor, loss_so_far, prior_secs = resume["cursor"], resume["loss_so_far"], resume["epoch_secs"]
            else:
                epoch_gen_state = generator.get_state()
                order = epoch_order()
                cursor, loss_so_far, prior_secs = 0, 0.0, 0.0
            resume = None
            updates_at_epoch_start = updates
            t0 = time.time()

            def on_batch(next_start, total_loss, epoch=epoch, order=order, epoch_gen_state=epoch_gen_state):
                nonlocal last_ckpt, updates
                if run_identity is not None:
                    updates += 1
                stop = args.stop_file is not None and (next_start // args.batch_size) % 200 == 0 \
                    and os.path.exists(args.stop_file)
                due = args.checkpoint_minutes > 0 and time.time() - last_ckpt >= 60 * args.checkpoint_minutes
                if (stop or due) and next_start < order.shape[0]:
                    save_state(epoch, next_start, epoch_gen_state, total_loss, prior_secs + time.time() - t0)
                    last_ckpt = time.time()
                    print(f"checkpoint epoch={epoch} cursor={next_start}/{order.shape[0]} -> {state_path}",
                          flush=True)
                    if stop:
                        raise Paused()

            try:
                train_loss, _ = run_batches(
                    model, train_idx, train_target, args.batch_size, features_per_pos, order,
                    train=True, objective=objective, criterion=criterion, scale_cp=scale_cp,
                    optimizer=optimizer, mirror_perm=mirror_perm, gen=generator, device=device,
                    canonical=args.canonical, shared=shared, wdl_target=train_wdl,
                    wdl_weight=args.wdl_weight, start_at=cursor, loss_so_far=loss_so_far,
                    on_batch=on_batch, row_weight=row_weight,
                    async_loss=args.async_loss,
                    end_at=(min(len(order), cursor + (args.max_steps - updates) * args.batch_size)
                            if args.max_steps is not None else None),
                ) if not kpp_zone else do_pass(train_idx, train_kpp, train_target, order, True, optimizer, True)
            except Paused:
                print(f"paused (stop file {args.stop_file}); resume with the same command + --resume", flush=True)
                raise SystemExit(75)
            train_secs = prior_secs + time.time() - t0

        model.eval()
        val_loss, preds = do_pass(val_idx, val_kpp, val_target, None, False, None, False)
        m = val_metrics(preds)
        gap = val_loss - train_loss

        extra = ""
        if "decisive_sign_agree" in m:
            extra = (f" decisive_sign={m.get('decisive_sign_agree', float('nan')):.4f}"
                     f" |pred|_dec={m.get('pred_abs_decisive', float('nan')):.3g}"
                     f" |pred|_near={m.get('pred_abs_near', float('nan')):.3g}")
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"gap={gap:+.4f} val_mae={m['val_mae']:.4f} val_corr={m['val_corr']:.4f}"
            f"{extra} train_secs={train_secs:.1f}"
        )
        cur_lr = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "gap": gap, "train_secs": round(train_secs, 2), "lr": cur_lr, **m})

        if chunks is not None:
            history[-1].update(chunks=chunk_losses, updates=updates, partial_epoch=partial_epoch)
        elif args.max_steps is not None:
            history[-1].update(updates=updates, partial_epoch=(updates >= args.max_steps and
                              cursor + (args.max_steps - updates_at_epoch_start) * args.batch_size < len(order)))
        if args.save_epochs:
            atomic_save(model.state_dict(), os.path.join(out_dir, f"nnue_epoch{epoch:02d}.pt"))
        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            atomic_save(model.state_dict(), os.path.join(out_dir, "nnue_best.pt"))

        if scheduler is not None:
            scheduler.step()
        if args.checkpoint_minutes > 0 or args.stop_file or args.resume:
            save_state(epoch + 1, 0, None, 0.0, 0.0)  # next epoch starts from the saved generator
            last_ckpt = time.time()

    torch.save(model.state_dict(), os.path.join(out_dir, "nnue_final.pt"))
    if args.max_steps is not None:
        print(f"optimizer_updates={updates} max_steps={args.max_steps}", flush=True)
    if (args.manifest or args.max_steps is not None) and device == 'cuda' and chunks is None:
        print(f"CUDA_PEAK allocated={torch.cuda.max_memory_allocated()} reserved={torch.cuda.max_memory_reserved()}", flush=True)
    config = {
        "total_features": total_features,
        "feature_dim": args.feature_dim,
        "hidden_dim": args.hidden_dim,
        "scale_cp": scale_cp,
        "objective": objective,
        "clamp_cp": args.clamp_cp,
        "loss": loss_label,
        "huber_delta": args.huber_delta,
        "augment_mirror": bool(args.augment_mirror),
        "dropout": args.dropout,
        "canonical": bool(args.canonical),
        "canonical_shared": bool(args.canonical_shared),
        "canonical_shared_ka": bool(ka),
        "canonical_shared_kav2": bool(kav2),
        "kpp_zone": bool(kpp_zone),
        "kpp_zone_total": kp.KPP_ZONE_TOTAL if kpp_zone else 0,
        "qat": bool(qat),
        "qat_fake_quant": bool(qat_fake_quant),
        "qat_tie": bool(args.qat_tie),
        "factorize": bool(args.factorize),
        "vm": bool(args.vm),
        "wdl_weight": args.wdl_weight,
        "wdl_lambda": args.wdl_lambda,
        "arch": (
            "QATSharedNNUE(BNNQ): quantization-aware shared HalfKP, 2x EmbeddingBag+ft_bias->ClippedReLU->concat(2*fd)->Linear->ClippedReLU->Linear->ClippedReLU->Linear(1)"
            if qat else
            "KppNNUE(BNN6): HalfKP shared + king-zone KPP, 4x EmbeddingBag->concat(4*fd)->Linear->ReLU->Linear->ReLU->Linear(1)"
            if kpp_zone else
            "HalfKAv2-Shared(BNN5): HalfKA minus anchor-king redundancy ->concat(2*fd)->Linear->ReLU->Linear->ReLU->Linear(1)"
            if kav2 else
            "HalfKA-Shared(BNN4): 2x EmbeddingBag(shared KA band, kings incl)->concat(2*fd)->Linear->ReLU->Linear->ReLU->Linear(1)"
            if ka else
            "CanonicalShared(BNN3): 2x EmbeddingBag(shared band)->concat(2*fd)->Linear->ReLU->Linear->ReLU->Linear(1)"
            if shared else
            "CanonicalTwoBand(BNN2): 2x EmbeddingBag(sum)->concat(2*fd)->Linear->ReLU->Linear->ReLU->Linear(1)"
            if args.canonical
            else "EmbeddingBag(sum)->Linear->ReLU->Linear->ReLU->Linear(1)"
        ),
    }
    if args.hidden_dim2 is not None:  # Task 149: old configurations retain their shape.
        config["hidden_dim2"] = args.hidden_dim2
    with open(os.path.join(out_dir, "model_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    if args.epochs == 0:  # Task 149: validate fast/cache/model setup without training.
        print(f"initialization_only=ok weights={os.path.join(out_dir, 'nnue_final.pt')}")
        return 0

    summary = {
        "n_train": n_train, "n_val": n_val, "objective": objective,
        "clamp_cp": args.clamp_cp, "loss": config["loss"], "huber_delta": args.huber_delta,
        "batch_size": args.batch_size, "epochs": args.epochs, "lr": args.lr,
        "weight_decay": args.weight_decay, "scale_cp": scale_cp,
        "baseline_val_loss": baseline_val_loss, "init_val_loss": init_val_loss,
        "best_val_loss": best_val, "best_epoch": best_epoch,
        "n_decisive": n_decisive, "history": history, "val_shards": cache_meta["val_shards"],
    }
    if chunks is not None:
        summary.update(n_train=sum(m.rows for m in chunks), chunks=run_identity['chunks'], chunk_order='fixed')
    if run_identity is not None:
        summary.update(updates=updates, max_steps=args.max_steps, run_identity=run_identity)
    with open(os.path.join(out_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    last = history[-1]
    print(f"baseline_val_loss={baseline_val_loss:.4f}")
    print(f"init_val_loss={init_val_loss:.4f}")
    print(f"best_val_loss={best_val:.4f}@epoch{best_epoch}")
    if "decisive_sign_agree" in last:
        print(f"final_decisive_sign_agree={last.get('decisive_sign_agree', float('nan')):.4f}")
        print(f"final_pred_abs_decisive={last.get('pred_abs_decisive', float('nan')):.3g}")
        print(f"final_pred_abs_near={last.get('pred_abs_near', float('nan')):.3g}")
    print(f"final_val_corr={last['val_corr']:.4f} final_val_mae={last['val_mae']:.4f}")
    print(f"weights={os.path.join(out_dir, 'nnue_best.pt')}")
    print("train_nnue=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
