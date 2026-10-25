"""Task 149: random wide checkpoint and CPU compatibility checks, no training."""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from trainer import kp
from trainer.binbo_to_yane import build_collapse_map
from trainer.compact_cache import CompactCache, convert
from trainer.export_nnue import export_qat
from trainer.nnue import QATSharedNNUE
from trainer.quant_format import header, read_header


def sha(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def cpu_checks(work, data):
    """Task 149: real cache decoding, asymmetric dense/sparse forward and gradients."""
    source, compact = work / "tiny_cache", work / "compact"
    source.mkdir(exist_ok=True)
    meta = dict(features_per_position=77, total_features=kp.CANONICAL_SHARED_TOTAL,
                canonical_shared=True, val_shards=["Task 149 tiny CPU fixture"])
    (source / "meta.json").write_text(json.dumps(meta) + "\n")
    for split, n in (("train", 16), ("val", 8)):
        a = np.load(data / f"caches/shared_64m_qs129/{split}_indices.npy", mmap_mode="r")
        np.save(source / f"{split}_indices.npy", a[:n])
        labels = np.load(data / f"work/task129/labels/{split}_dlshogi_evals.npy", mmap_mode="r")
        np.save(source / f"{split}_evals.npy", labels[:n])
    if compact.exists():  # Task 149: this tiny generated fixture is safe to regenerate.
        shutil.rmtree(compact)
    convert(source, compact, chunk_rows=8)
    cache = CompactCache(compact, "train", device="cpu", source=source)
    rows = torch.arange(4)
    decoded = cache[rows]
    assert np.array_equal(decoded.numpy(), np.load(source / "train_indices.npy")[:4])

    # Keep the gradient oracle tiny, with actual wide dense layers and all FT options.
    with contextlib.ExitStack() as stack:
        for name, value in dict(SQUARE_NB=3, FEATURES_PER_KING=7, VIEW_DIM=21,
                                CANONICAL_SHARED_TOTAL=23).items():
            stack.enter_context(patch.object(kp, name, value))
        tie = np.arange(23)
        tie[[4, 5]] = 3
        tie[21:] = 23
        torch.manual_seed(149)
        dense = QATSharedNNUE(total_features=23, feature_dim=512, hidden_dim=8, hidden_dim2=96,
                             fake_quant=True, factorize=True, tie_collapse_map=tie)
        sparse = copy.deepcopy(dense)
        sparse.sparse_ft = True
        own, opp = torch.tensor([1, 4, 5, 21, 9, 3]), torch.tensor([2, 3, 7, 22, 5, 11])
        off = torch.tensor([0, 4])
        a, b = dense(own, off, opp, off), sparse(own, off, opp, off)
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
        a.sum().backward()
        b.sum().backward()
        for (_, p), (_, q) in zip(dense.named_parameters(), sparse.named_parameters()):
            torch.testing.assert_close(p.grad, q.grad, atol=2e-6, rtol=2e-5)
        # Explicit symmetric width must consume the identical RNG stream/default bytes.
        torch.manual_seed(149)
        old = QATSharedNNUE(total_features=23)
        torch.manual_seed(149)
        explicit = QATSharedNNUE(total_features=23, hidden_dim2=32)
        assert all(torch.equal(t, explicit.state_dict()[n]) for n, t in old.state_dict().items())
    for dims in ((176744, 256, 32, 32), (176744, 512, 8, 96)):
        meta, offset = read_header(header(*dims))
        assert (meta["total"], meta["fd"], meta["hidden"], meta["hidden2"]) == dims
        assert offset == (24 if dims[2] == dims[3] else 32)
    bad = bytearray(header(176744, 512, 8, 96))
    bad[28:32] = (2).to_bytes(4, "little")
    try:
        read_header(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown features accepted")
    print("CPU checks GREEN: compact decode, 512-8-96 dense/sparse outputs+gradients, legacy RNG, headers")


def prepare(work, data):
    """Task 149: export random tied/factorized weights and byte-check adopted 256."""
    model_dir = work / "random512"
    model_dir.mkdir(exist_ok=True)
    torch.manual_seed(149)
    model = QATSharedNNUE(feature_dim=512, hidden_dim=8, hidden_dim2=96,
                         fake_quant=True, factorize=True, tie_collapse_map=build_collapse_map())
    with torch.no_grad():
        model.ft_factor.normal_(0, 0.001)
        model.ft_bias.uniform_(0.05, 0.25)
    config = dict(total_features=kp.CANONICAL_SHARED_TOTAL, feature_dim=512, hidden_dim=8,
                  hidden_dim2=96, scale_cp=600., qat=True, qat_tie=True, qat_fake_quant=True,
                  factorize=True, vm=False)
    (model_dir / "model_config.json").write_text(json.dumps(config, indent=2) + "\n")
    torch.save(model.state_dict(), model_dir / "nnue_final.pt")
    del model
    export_qat(SimpleNamespace(model_dir=str(model_dir), weights="nnue_final.pt", out="nnue_final.bnnq"), config)
    adopted = data / "models/task142_finetune"
    cfg = json.loads((adopted / "model_config.json").read_text())
    target = work / "task142_reexport.bnnq"
    export_qat(SimpleNamespace(model_dir=str(adopted), weights="nnue_final.pt", out=str(target)), cfg)
    expected = adopted / "nnue_final.bnnq"
    assert sha(target) == sha(expected), "legacy exported bytes changed"
    print(f"default 256 byte-identical GREEN: {sha(target)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work-dir", required=True)
    ap.add_argument("--checks-only", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(2)
    work = Path(args.work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    data = Path(os.environ.get("DATA_ROOT", Path.home() / "binbo-shogi-data"))
    cpu_checks(work, data)
    if not args.checks_only:
        prepare(work, data)


if __name__ == "__main__":
    main()
