#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math

import torch
from torch import nn

try:
    from . import kp
    from .nnue import DEFAULT_EVAL_SCALE_CP, MinimalNNUE, collate_sparse_batch, eval_to_winrate
except ImportError:
    import kp
    from nnue import DEFAULT_EVAL_SCALE_CP, MinimalNNUE, collate_sparse_batch, eval_to_winrate


def assert_label_transform(scale_cp: float) -> None:
    values = torch.tensor([-1200.0, -600.0, 0.0, 600.0, 1200.0])
    winrates = eval_to_winrate(values, scale_cp)
    if not torch.isclose(winrates[2], torch.tensor(0.5)):
        raise AssertionError("eval=0 must map to 0.5")
    if not (winrates[3] > 0.5 and winrates[4] > winrates[3]):
        raise AssertionError("positive eval must map above 0.5 monotonically")
    if not (winrates[1] < 0.5 and winrates[0] < winrates[1]):
        raise AssertionError("negative eval must map below 0.5 monotonically")
    pairs = " ".join(
        f"{int(value.item())}:{winrate.item():.6f}"
        for value, winrate in zip(values, winrates)
    )
    print(f"label_check scale_cp={scale_cp:g} {pairs}")


def cyclic_batch(examples: list[kp.KpExample], start: int, batch_size: int) -> list[kp.KpExample]:
    return [examples[(start + offset) % len(examples)] for offset in range(batch_size)]


def parameter_vector(model: torch.nn.Module) -> torch.Tensor:
    return torch.nn.utils.parameters_to_vector(
        [parameter.detach().cpu().clone() for parameter in model.parameters()]
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("hcpe")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--feature-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--scale-cp", type=float, default=DEFAULT_EVAL_SCALE_CP)
    parser.add_argument("--seed", type=int, default=20260621)
    args = parser.parse_args()

    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    torch.manual_seed(args.seed)
    device = torch.device("cpu")
    assert_label_transform(args.scale_cp)

    examples = kp.read_hcpe(args.hcpe)
    if not examples:
        raise ValueError(f"hcpe has no entries: {args.hcpe}")

    evals = torch.tensor([example.eval for example in examples], dtype=torch.float32)
    targets = eval_to_winrate(evals, args.scale_cp)
    print(f"data_entries={len(examples)}")
    print(f"eval_min={int(evals.min().item())}")
    print(f"eval_max={int(evals.max().item())}")
    print(f"target_min={targets.min().item():.6f}")
    print(f"target_max={targets.max().item():.6f}")
    print(f"kp_total_features={kp.TOTAL_FEATURES}")
    print(
        "model="
        f"EmbeddingBag({kp.TOTAL_FEATURES},{args.feature_dim},sum)"
        f" -> Linear({args.feature_dim},{args.hidden_dim})"
        f" -> Linear({args.hidden_dim},{args.hidden_dim})"
        " -> Linear(...,1)"
    )
    print("loss=BCEWithLogitsLoss optimizer=Adam")

    model = MinimalNNUE(
        total_features=kp.TOTAL_FEATURES,
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    before = parameter_vector(model)
    first_predictions_logged = False
    first_loss = None
    last_loss = None

    for step in range(1, args.steps + 1):
        batch_examples = cyclic_batch(examples, (step - 1) * args.batch_size, args.batch_size)
        batch = collate_sparse_batch(batch_examples, device=device)
        target = eval_to_winrate(batch.eval_cp, args.scale_cp)

        optimizer.zero_grad(set_to_none=True)
        logits = model(batch.flat_indices, batch.offsets)
        probabilities = torch.sigmoid(logits)
        loss = criterion(logits, target)

        if not torch.isfinite(loss):
            raise AssertionError(f"non-finite loss at step {step}: {loss.item()}")
        if not torch.isfinite(probabilities).all():
            raise AssertionError(f"non-finite probability at step {step}")

        if not first_predictions_logged:
            print(
                "forward_sample "
                f"logit0={logits[0].item():.6f} "
                f"prob0={probabilities[0].item():.6f} "
                f"target0={target[0].item():.6f}"
            )
            first_predictions_logged = True

        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        if first_loss is None:
            first_loss = loss_value
        last_loss = loss_value
        print(
            f"step={step} loss={loss_value:.8f} "
            f"prob_mean={probabilities.mean().item():.6f} "
            f"target_mean={target.mean().item():.6f}"
        )

    after = parameter_vector(model)
    max_abs_delta = torch.max(torch.abs(after - before)).item()
    l2_delta = torch.linalg.vector_norm(after - before).item()
    if not math.isfinite(max_abs_delta) or not math.isfinite(l2_delta):
        raise AssertionError("non-finite parameter delta")
    if max_abs_delta <= 0.0:
        raise AssertionError("parameters did not change")

    print(f"weight_update_max_abs_delta={max_abs_delta:.10f}")
    print(f"weight_update_l2_delta={l2_delta:.10f}")
    print(f"loss_first={first_loss:.8f}")
    print(f"loss_last={last_loss:.8f}")
    print("nan_check=ok")
    print("train_smoke=ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
