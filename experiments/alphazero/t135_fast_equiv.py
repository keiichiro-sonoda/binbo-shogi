"""Task 135: real-cache equivalence/noise-floor benchmark and independent CPU checks.

  source scripts/env.sh
  PYTHONPATH=. "$VENV_PY" experiments/alphazero/t135_fast_equiv.py --cpu-checks
  PYTHONPATH=. "$VENV_PY" experiments/alphazero/t135_fast_equiv.py --cli-checks \
    --eval-npy "$DATA_ROOT/work/task129/labels"
  PYTHONPATH=. "$VENV_PY" experiments/alphazero/t135_fast_equiv.py \
    --cache "$DATA_ROOT/caches/shared_64m_qs129" \
    --compact-cache "$DATA_ROOT/caches/compact_qs129" \
    --eval-npy <adopted-label-dir> --mirror-cache <mirror-perm.npy> --steps 2000 \
    --output experiments/alphazero/t135_equiv_result.json

The benchmark trains warmup+steps from identical seeds for each variant, through
train_nnue.run_batches. Timings exclude warmup, initial cache upload, randperm,
and final parameter/loss readback. Default is repeated to measure GPU noise.
Relative difference = max|candidate-reference| / max|reference| per parameter.
Decision: variants that change no float summation order (repeat, compact, async_loss) must be
bit-exact; sparse_ft / fused_adam / all are judged against `reorder` (default with each half's
features reversed = the same math, another summation order), whose drift after N steps is the
floor: feature-table RMS drift and last-500-step mean loss within 3x of it. First-step checks
compare sparse vs dense gradients and fused vs foreach Adam directly, before any drift.
JSON contains every step's loss and per-parameter comparisons, not just an average.
The all-fast profiler pass reports scalar readbacks / stream syncs (sparse_ft's torch.unique
costs one small sync per step by design; it is reported, not failed).
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import warnings

import numpy as np
import torch
from torch import nn

from trainer import kp, mirror
from trainer.binbo_to_yane import build_collapse_map
from trainer.compact_cache import CompactCache, convert, decode_chunk, encode_chunk
from trainer.nnue import QATSharedNNUE
from trainer.train_nnue import load_split, run_batches


class CpuChecks(unittest.TestCase):
    """Task 135: small-table gradient oracles, exact data/RNG, and serialized resume."""

    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(135)
        self.layout = contextlib.ExitStack()
        for name, value in dict(SQUARE_NB=3, FEATURES_PER_KING=7, VIEW_DIM=21,
                                CANONICAL_SHARED_TOTAL=23).items():
            self.layout.enter_context(patch.object(kp, name, value))
        self.addCleanup(self.layout.close)

    def model(self, factorize=True, tie=True, quant=True, sparse=False):
        collapse = np.arange(23)
        collapse[[4, 5]] = 3
        collapse[21:] = 23
        return QATSharedNNUE(total_features=23, feature_dim=9, hidden_dim=7,
                             factorize=factorize, tie_collapse_map=collapse if tie else None,
                             fake_quant=quant, sparse_ft=sparse)

    def test_forward_backward(self):
        largest = 0.0
        for factorize in (False, True):
            for tie in (False, True):
                for quant in (False, True):
                    with self.subTest(factorize=factorize, tie=tie, quant=quant):
                        dense = self.model(factorize, tie, quant)
                        with torch.no_grad():
                            dense.feature_transform.weight.uniform_(-0.1, 0.1)
                            dense.ft_bias.uniform_(0.15, 0.9)
                            if factorize:
                                dense.ft_factor.uniform_(-0.02, 0.02)
                            # A nonzero PAD must still be excluded (forward/backward).
                            if tie:
                                dense.feature_transform.weight[23].fill_(0.7)
                        fast = copy.deepcopy(dense)
                        fast.sparse_ft = True
                        own = torch.tensor([0, 3, 4, 21, 8, 8, 22, 20, 21, 22, 21, 22])
                        opp = torch.tensor([4, 0, 3, 22, 20, 8, 8, 21, 22, 21, 22, 21])
                        offsets = torch.tensor([0, 4, 8])
                        upstream = torch.randn(3, 18)
                        args = (own, offsets, opp, offsets)
                        features = []
                        for model in (dense, fast):
                            f = model.forward_features(*args)
                            features.append(f.detach())
                            (f.mul(upstream).sum() + model(*args).square().sum()).backward()
                        torch.testing.assert_close(*features, rtol=0, atol=0)
                        for (name, a), (_, b) in zip(dense.named_parameters(), fast.named_parameters()):
                            torch.testing.assert_close(a.grad, b.grad, rtol=1e-5, atol=2e-6, msg=name)
                            largest = max(largest, (a.grad - b.grad).abs().max().item())
                        if tie:
                            self.assertEqual(fast.feature_transform.weight.grad[23].count_nonzero(), 0)
        print(f"CPU forward/backward: 8 configurations, max_gradient_diff={largest:.9g}")

    def test_compact_escapes(self):
        # Restore the canonical codec constants; the small layout patch is for NNUE only.
        with patch.object(kp, "VIEW_DIM", 81 * 2182):
            rng = np.random.default_rng(135)
            rows = rng.integers(np.iinfo(np.int32).min, np.iinfo(np.int32).max,
                                size=(31, 77), dtype=np.int32)
            rows[0] = 0
            rows[1, :39] = 2182 * 80 + np.arange(39)
            rows[1, 39:] = 2182 * 8 + np.arange(38)
            rows[2, :5] = [176742, 176743, 176744, -1, 0]
            k, c, bases, extra = encode_chunk(rows)
            np.testing.assert_array_equal(decode_chunk(k, c, bases, extra), rows)
            with tempfile.TemporaryDirectory() as tmp:
                source, compact = Path(tmp) / "source", Path(tmp) / "compact"
                source.mkdir()
                for split in ("train", "val"):
                    np.save(source / f"{split}_indices.npy", rows)
                convert(source, compact, chunk_rows=5)
                for split in ("train", "val"):
                    cache = CompactCache(compact, split, "cpu", limit=29)
                    order = torch.tensor([28, 2, 1, 0, 2, 17])
                    torch.testing.assert_close(cache[order], torch.from_numpy(rows[order]).long())

    def fixture(self, tmp):
        source, compact = Path(tmp) / "source", Path(tmp) / "compact"
        source.mkdir()
        # Piece codes stay in the real canonical band; only low rows are needed
        # for this model. Include both side rows and repeated features.
        rows = np.random.default_rng(136).integers(0, 21, (43, 77), dtype=np.int32)
        rows[:, 38] = np.arange(len(rows)) % 2 + 21
        for split in ("train", "val"):
            np.save(source / f"{split}_indices.npy", rows)
        convert(source, compact, chunk_rows=11)
        cache = CompactCache(compact, "train", "cpu")
        target = torch.linspace(-1900, 1900, len(rows))
        perm = torch.arange(24)
        perm[:21] = torch.arange(21).flip(0)
        return torch.from_numpy(rows), cache, target, perm

    def train_pass(self, model, data, target, perm, generator, order, optimizer,
                   async_loss=False, **kwargs):
        return run_batches(model, data, target, 7, 77, order, train=True, objective="cp",
                           criterion=nn.HuberLoss(delta=600), scale_cp=600, optimizer=optimizer,
                           mirror_perm=perm, gen=generator, shared=True,
                           async_loss=async_loss, **kwargs)

    def test_data_rng_and_async_loss(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, cache, target, perm = self.fixture(tmp)
            seed_model = self.model()
            results = []
            for source, async_loss in ((data, False), (data, True), (cache, False), (cache, True)):
                model = copy.deepcopy(seed_model)
                generator = torch.Generator().manual_seed(135)
                order = torch.randperm(len(data), generator=generator)[1:-1:2]  # row-mask-like subset, tail
                optimizer = torch.optim.Adam(model.parameters(), lr=6e-5)
                loss, _ = self.train_pass(model, source, target, perm, generator, order,
                                          optimizer, async_loss=async_loss)
                results.append((model.state_dict(), generator.get_state(), loss))
            for state, rng, loss in results[1:]:
                for name in state:
                    torch.testing.assert_close(state[name], results[0][0][name], rtol=0, atol=0)
                self.assertTrue(torch.equal(rng, results[0][1]))
                self.assertEqual(loss, results[0][2])
            print("CPU data/RNG/loss: compact and async individually/together are bit-exact")

    def test_mid_epoch_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            data, cache, target, perm = self.fixture(tmp)
            initial = self.model()
            snapshots = []
            for fast in (False, True):
                generator = torch.Generator().manual_seed(135)
                order = torch.randperm(len(data), generator=generator)
                model = copy.deepcopy(initial)
                optimizer = torch.optim.Adam(model.parameters(), lr=6e-5)
                def stop(cursor, total):
                    if cursor == 14:
                        stream = io.BytesIO()
                        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                                        generator=generator.get_state(), cursor=cursor,
                                        loss=float(total)), stream)
                        snapshots.append(stream.getvalue())
                        raise InterruptedError()
                with self.assertRaises(InterruptedError):
                    self.train_pass(model, data, target, perm, generator, order, optimizer, on_batch=stop)
                saved = torch.load(io.BytesIO(snapshots[-1]), weights_only=False)
                resumed = copy.deepcopy(initial)
                resumed.sparse_ft = fast
                resumed.load_state_dict(saved["model"])
                opt = torch.optim.Adam(resumed.parameters(), lr=6e-5)
                opt.load_state_dict(saved["optimizer"])
                generator.set_state(saved["generator"])
                loss, _ = self.train_pass(resumed, cache if fast else data, target, perm,
                                          generator, order, opt, async_loss=fast,
                                          start_at=saved["cursor"], loss_so_far=saved["loss"])
                if not fast:
                    reference, ref_rng, ref_loss = resumed, generator.get_state(), loss
                else:
                    for name, parameter in resumed.named_parameters():
                        torch.testing.assert_close(parameter, dict(reference.named_parameters())[name],
                                                   rtol=1e-5, atol=2e-7, msg=name)
                    self.assertTrue(torch.equal(generator.get_state(), ref_rng))
                    self.assertAlmostEqual(loss, ref_loss, delta=0.1)
            # Same slow checkpoint, continued uninterrupted, must be bit-exact.
            uninterrupted = copy.deepcopy(initial)
            gen = torch.Generator().manual_seed(135)
            order = torch.randperm(len(data), generator=gen)
            opt = torch.optim.Adam(uninterrupted.parameters(), lr=6e-5)
            loss, _ = self.train_pass(uninterrupted, data, target, perm, gen, order, opt)
            for a, b in zip(uninterrupted.parameters(), reference.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertEqual(loss, ref_loss)
            print("CPU serialized mid-epoch resume: slow exact; slow -> compact/sparse/async allclose")

    def test_adam_fused(self):
        parameters = [nn.Parameter(torch.randn(23, 9)), nn.Parameter(torch.randn(9))]
        fused = [nn.Parameter(p.detach().clone()) for p in parameters]
        a = torch.optim.Adam(parameters, lr=6e-5, foreach=True, weight_decay=0.01)
        b = torch.optim.Adam(fused, lr=6e-5, fused=True, weight_decay=0.01)
        for step in range(60):
            for p, q in zip(parameters, fused):
                p.grad = torch.randn_like(p) if step < 3 else torch.zeros_like(p)
                q.grad = p.grad.clone()  # zero rows still move through their Adam momentum
            a.step()
            b.step()
        largest = max((p - q).abs().max().item() for p, q in zip(parameters, fused))
        for p, q in zip(parameters, fused):
            torch.testing.assert_close(p, q, rtol=1e-5, atol=1e-7)
        print(f"CPU fused vs foreach Adam (60 steps, dense momentum/weight decay): max_diff={largest:.9g}")


def cpu_checks():
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CpuChecks))
    return 0 if result.wasSuccessful() else 1


def cli_checks(args):
    """Task 135: actual CLI checkpoints/resume on a small real-cache prefix, CPU."""
    from trainer import train_nnue
    import sys

    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source, labels = root / "source", root / "labels"
        source.mkdir()
        labels.mkdir()
        for split, n in (("train", 403), ("val", 17)):
            for suffix in ("indices", "evals"):
                values = np.load(Path(args.cache) / f"{split}_{suffix}.npy", mmap_mode="r")[:n]
                np.save(source / f"{split}_{suffix}.npy", values)
            if args.eval_npy:
                values = np.load(Path(args.eval_npy) / f"{split}_dlshogi_evals.npy", mmap_mode="r")[:n]
                np.save(labels / f"{split}_dlshogi_evals.npy", values)
        meta = json.loads((Path(args.cache) / "meta.json").read_text())
        meta.update(n_train=403, n_val=17)
        (source / "meta.json").write_text(json.dumps(meta))
        convert(source, root / "compact", chunk_rows=101)
        # The trainer appends _shared to its --mirror-cache filename. The
        # benchmark takes the actual permutation filename instead.
        np.save(root / "mirror_shared.npy", mirror.build_mirror_perm_shared(args.mirror_cache))
        base = [str(source), "--qat", "--qat-tie", "--factorize", "--augment-mirror",
                "--mirror-cache", str(root / "mirror.npy"), "--feature-dim", "4", "--hidden-dim", "4",
                "--lr", "6e-5", "--epochs", "2", "--batch-size", "2", "--mmap",
                "--checkpoint-minutes", "10", "--device", "cpu"]
        if args.eval_npy:
            base += ["--eval-npy", str(labels)]
        fast_flags = ["--fast", "--compact-cache", str(root / "compact")]

        def run(name, fast=False, resume=False, stop=False, periodic=False):
            stopfile = root / "stop"
            if stop:
                stopfile.touch()
            else:
                stopfile.unlink(missing_ok=True)
            command = base + ["--out", str(root / name)] + (fast_flags if fast else [])
            if resume:
                command += ["--resume"]
            if stop:
                command += ["--stop-file", str(stopfile)]
            if periodic:
                command += ["--limit-train", "5", "--epochs", "1", "--checkpoint-minutes", "1e-9"]
            captured = io.StringIO()
            with patch.object(sys, "argv", ["train_nnue"] + command), contextlib.redirect_stdout(captured):
                try:
                    code = train_nnue.main()
                except SystemExit as exc:
                    code = exc.code
            if code != (75 if stop else 0):
                raise AssertionError(captured.getvalue())
            if periodic and "checkpoint epoch=1 cursor=" not in captured.getvalue():
                raise AssertionError("periodic mid-epoch checkpoint was not saved")
            if stop:
                saved = torch.load(root / name / "train_state.pt", weights_only=False)
                assert saved["cursor"] == 400 and isinstance(saved["loss_so_far"], float)
            print(f"CLI {name}: resume={resume} exit={code}", flush=True)

        run("slow")
        run("fast", fast=True)
        run("slow_to_fast", stop=True)
        run("slow_to_fast", fast=True, resume=True)
        run("fast_resume", fast=True, stop=True)
        run("fast_resume", fast=True, resume=True)
        run("periodic", fast=True, periodic=True)
        states = {name: torch.load(root / name / "train_state.pt", weights_only=False)
                  for name in ("slow", "fast", "slow_to_fast", "fast_resume")}
        for name in ("fast", "slow_to_fast", "fast_resume"):
            reference = states["fast" if name == "fast_resume" else "slow"]
            candidate = states[name]
            delta = difference(reference["model"], candidate["model"])
            largest = max(d["max_abs"] for d in delta.values())
            for key, value in reference["model"].items():
                torch.testing.assert_close(value, candidate["model"][key], rtol=1e-5, atol=2e-7)
                if name == "fast_resume":
                    assert torch.equal(value, candidate["model"][key]), key
            assert torch.equal(reference["generator"], candidate["generator"])
            assert candidate["epoch"] == 3 and candidate["cursor"] == 0
            assert all(group["fused"] for group in candidate["optimizer"]["param_groups"])
            print(f"CLI {name}: max_parameter_diff={largest:.9g}, generator exact", flush=True)
    print("CLI checkpoint/resume=PASS (slow -> fast, fast -> fast, periodic save)")
    return 0


def difference(reference, candidate):
    output = {}
    for name, a in reference.items():
        b = candidate[name]
        if not a.is_floating_point():
            if not torch.equal(a, b):
                raise AssertionError(f"nonfloating state changed: {name}")
            continue
        absolute = (a - b).abs().max().item()
        norm = a.abs().max().item()
        output[name] = dict(max_abs=absolute, relative=absolute / norm if norm else absolute)
    return output


class Reordered:
    """Task 135 floor: the same rows with each half's features in reverse order. The sum inside
    an embedding bag is order-free, so this changes only the floating-point summation order --
    the same kind of difference the fast path introduces. Its drift after N steps is the yardstick."""

    def __init__(self, base):
        self.base, self.shape = base, base.shape
        self.resident = getattr(base, "resident", False)
        self.cols = torch.tensor(list(range(38, -1, -1)) + list(range(76, 38, -1)))

    def __getitem__(self, rows):
        out = self.base[rows]
        return out[:, self.cols.to(out.device)]


def first_step_checks(args, indices, target, perm, device):
    """Task 135: one batch from one init. sparse_ft vs dense gradients; fused vs foreach Adam on
    identical gradients. These are the direct numerical comparisons (before any drift)."""
    rows = torch.arange(args.batch)
    batch = indices[rows].long()
    own = batch[:, :39].reshape(-1).to(device)
    opp = batch[:, 39:].reshape(-1).to(device)
    oo = torch.arange(0, args.batch * 39, 39, device=device)
    po = torch.arange(0, args.batch * 38, 38, device=device)
    y = target[rows].to(device)
    grads = []
    for sparse in (False, True):
        torch.manual_seed(args.seed)
        m = QATSharedNNUE(feature_dim=args.feature_dim, hidden_dim=32, fake_quant=True, factorize=True,
                          tie_collapse_map=build_collapse_map(vm=False), sparse_ft=sparse).to(device)
        nn.HuberLoss(delta=600)(m(own, oo, opp, po) * 600, y).backward()
        grads.append({n: p.grad.detach().clone() for n, p in m.named_parameters()})
    out = {"grad_sparse_vs_dense": {n: dict(max_abs=(grads[0][n] - grads[1][n]).abs().max().item(),
                                            ref_max=grads[0][n].abs().max().item()) for n in grads[0]}}
    params = []
    for fused in (False, True):
        torch.manual_seed(args.seed)
        m = QATSharedNNUE(feature_dim=args.feature_dim, hidden_dim=32, fake_quant=True, factorize=True,
                          tie_collapse_map=build_collapse_map(vm=False)).to(device)
        opt = torch.optim.Adam(m.parameters(), lr=6e-5, **({"fused": True} if fused else {}))
        for _ in range(3):  # three steps on the same dense gradients
            for n, p in m.named_parameters():
                p.grad = grads[0][n].clone()
            opt.step()
        params.append({n: p.detach().clone() for n, p in m.named_parameters()})
    out["adam_fused_vs_foreach_3steps"] = {n: dict(max_abs=(params[0][n] - params[1][n]).abs().max().item())
                                           for n in params[0]}
    for key, d in out.items():
        worst = max(d.items(), key=lambda kv: kv[1]["max_abs"])
        print(f"{key}: worst {worst[0]} max_abs={worst[1]['max_abs']:.3g}"
              + (f" (|ref| max {worst[1]['ref_max']:.3g})" if "ref_max" in worst[1] else ""), flush=True)
    return out


def drift(reference, state):
    """RMS and max |diff| of the feature table after N steps (the bulk of the parameters)."""
    a, b = reference["feature_transform.weight"], state["feature_transform.weight"]
    d = (a - b).float()
    return dict(ft_rms=d.square().mean().sqrt().item(), ft_max=d.abs().max().item())


def benchmark(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable: run --cpu-checks here; run the real benchmark on a CUDA host")
    device = args.device
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        indices, evals = load_split(args.cache, "train", mmap=True)
    if args.eval_npy:
        raw = np.load(Path(args.eval_npy) / "train_dlshogi_evals.npy")
        if len(raw) != len(indices):
            raise ValueError("--eval-npy length mismatch")
        evals = torch.from_numpy(raw.astype(np.float32))
    if args.limit_train:
        indices, evals = indices[:args.limit_train], evals[:args.limit_train]
    target = evals.clamp(-2000, 2000)
    del evals
    perm = torch.from_numpy(mirror.build_mirror_perm_shared(args.mirror_cache)).long()
    needed = (args.warmup + args.steps) * args.batch
    if needed > len(indices):
        raise ValueError("warmup+steps exceed one epoch; reduce steps or increase --limit-train")
    # Keep a single encoded dataset resident across sequential runs; loading and
    # decoded-data verification are outside the steady-state measurements.
    compact = CompactCache(args.compact_cache, "train", device, limit=len(indices), source=args.cache)
    gpu_target, gpu_perm = target.to(device), perm.to(device)
    sync = torch.cuda.synchronize if device == "cuda" else lambda: None
    checks = first_step_checks(args, indices, target, perm, device)
    variants = ["default", "repeat", "reorder", "compact", "sparse_ft", "fused_adam", "async_loss", "all"]
    results, reference, noise = {}, None, None
    for variant in variants:
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        torch.manual_seed(args.seed)
        resident = variant in ("compact", "all")
        sparse = variant in ("sparse_ft", "all")
        fused = variant in ("fused_adam", "all")
        asynchronous = variant in ("async_loss", "all")
        model = QATSharedNNUE(feature_dim=args.feature_dim, hidden_dim=32, fake_quant=True,
                              factorize=True, tie_collapse_map=build_collapse_map(vm=False),
                              sparse_ft=sparse).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=6e-5, **({"fused": True} if fused else {}))
        model.train()
        generator = torch.Generator().manual_seed(args.seed)
        order = torch.randperm(len(indices), generator=generator)  # exact trainer stream, CPU
        curve = torch.zeros(args.steps + args.warmup, dtype=torch.float64,
                            device=device if asynchronous else "cpu")
        def run(selected, offset):
            def record(cursor, total):
                curve[offset + cursor // args.batch - 1] = total
            data = compact if resident else indices
            if variant == "reorder":
                data = Reordered(data)
            return run_batches(model, data, gpu_target if resident else target,
                               args.batch, 77, selected, train=True, objective="cp",
                               criterion=nn.HuberLoss(delta=600), scale_cp=600, optimizer=optimizer,
                               mirror_perm=gpu_perm if resident else perm, gen=generator,
                               device=device, shared=True, async_loss=asynchronous, on_batch=record)
        if args.warmup:
            run(order[:args.warmup * args.batch], 0)
        sync()
        started = time.perf_counter()
        run(order[args.warmup * args.batch:needed], args.warmup)
        sync()
        step_ms = (time.perf_counter() - started) * 1000 / args.steps
        cumulative = curve.cpu()
        losses = []
        for lo, hi in ((0, args.warmup), (args.warmup, len(curve))):
            if hi > lo:
                segment = cumulative[lo:hi]
                losses.extend(((segment - torch.cat([segment.new_zeros(1), segment[:-1]]))
                               / args.batch).tolist())
        state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        if reference is None:
            reference = state
        diff = difference(reference, state)
        # Decision (Task 135, fixed before the GPU rerun): variants that do not reorder any float
        # sum (repeat, compact, async_loss) must be bit-exact vs default. Variants that do
        # (sparse_ft, fused_adam, all) drift chaotically like any summation-order change, so they
        # are judged against `reorder` (same math, reversed feature order): feature-table RMS drift
        # <= 3x reorder's, and |mean loss over the last 500 steps - default's| <= 3x reorder's.
        dr = drift(reference, state)
        tail = float(np.mean(losses[-500:]))
        tail_d = abs(tail - results["default"]["tail_loss"]) if results else 0.0
        if variant in ("repeat", "compact", "async_loss"):
            passed = all(v["max_abs"] == 0 for v in diff.values())
        elif variant in ("sparse_ft", "fused_adam", "all"):
            fl = results["reorder"]
            passed = (dr["ft_rms"] <= 3 * fl["drift"]["ft_rms"]
                      and tail_d <= 3 * max(fl["tail_loss_diff"], 1e-9))
        else:
            passed = True
        loss_max = max(abs(a - b) for a, b in zip(losses, results["default"]["loss_curve"])) if results else 0
        results[variant] = dict(step_ms=step_ms, differences=diff, loss_curve=losses, drift=dr,
                                tail_loss=tail, tail_loss_diff=tail_d,
                                max_loss_diff=loss_max, generator_sha256=hashlib.sha256(
                                    generator.get_state().numpy().tobytes()).hexdigest(), passed=passed)
        print(f"{variant}: {step_ms:.3f} ms/step, tail500_loss={tail:.2f} (diff {tail_d:.2f}), "
              f"ft_rms_drift={dr['ft_rms']:.3g} ft_max_drift={dr['ft_max']:.3g}, "
              f"{'PASS' if passed else 'FAIL'}", flush=True)
        for name, value in diff.items():
            print(f"  {name}: max_abs={value['max_abs']:.9g} relative={value['relative']:.9g}", flush=True)
        if results[variant]["generator_sha256"] != results["default"]["generator_sha256"]:
            raise AssertionError("CPU generator stream changed")
        if variant == "all" and device == "cuda" and args.profile_steps:
            # Profile more steps after measurement; exclude run_batches' permitted
            # final .item() by allowing exactly one local scalar readback.
            from torch.profiler import ProfilerActivity, profile
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                run(order[:min(args.profile_steps, args.steps) * args.batch], 0)
                sync()
            events = prof.key_averages()
            scalars = sum(e.count for e in events if e.key == "aten::_local_scalar_dense")
            stream_syncs = sum(e.count for e in events if e.key == "cudaStreamSynchronize")
            results[variant]["profile"] = dict(scalar_readbacks=scalars, stream_syncs=stream_syncs)
            print(f"all-fast profiler: scalar_readbacks={scalars} stream_syncs={stream_syncs} "
                  "(torch.unique: one per step by design)", flush=True)
            if args.trace:
                prof.export_chrome_trace(args.trace)
        del model, optimizer, state, order, curve
    output = dict(args=vars(args), torch=torch.__version__,
                  device=torch.cuda.get_device_name() if device == "cuda" else "cpu",
                  decision="non-reordering variants bit-exact; reordering ones <= 3x the reorder floor "
                           "(ft RMS drift, last-500 mean loss)", first_step=checks, results=results)
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
    passed = all(result["passed"] for result in results.values())
    print(f"decision={'PASS' if passed else 'FAIL'} all_fast_under_10ms={results['all']['step_ms'] < 10} "
          f"report={args.output}")
    return 0 if passed else 1


def main():
    data = Path(os.environ.get("DATA_ROOT", Path.home() / "binbo-shogi-data"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-checks", action="store_true")
    parser.add_argument("--cli-checks", action="store_true")
    parser.add_argument("--cache", default=str(data / "caches/shared_64m_qs129"))
    parser.add_argument("--compact-cache", default=str(data / "caches/compact_qs129"))
    parser.add_argument("--eval-npy")
    parser.add_argument("--mirror-cache", default=str(data / "caches/shared_8m_mirror_perm_shared.npy"))
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--batch", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=20260621)
    parser.add_argument("--limit-train", type=int, default=63950000)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--profile-steps", type=int, default=10)
    parser.add_argument("--trace", help="optional all-fast Chrome trace path")
    parser.add_argument("--output", default="experiments/alphazero/t135_equiv_result.json")
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 0 or args.batch < 1 or args.profile_steps < 0:
        parser.error("steps/batch must be positive, warmup/profile-steps nonnegative")
    return cpu_checks() if args.cpu_checks else cli_checks(args) if args.cli_checks else benchmark(args)


if __name__ == "__main__":
    raise SystemExit(main())
