from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

try:
    from . import kp
except ImportError:
    import kp


DEFAULT_EVAL_SCALE_CP = 600.0

# --- Task 058: quantization-aware (QAT) shared HalfKP, matching YaneuraOu's int forward ---
# Scales below are shared by the model's fake-quant (training) AND the BNNQ export
# (deployment), so the float forward and the deployed int forward agree. They mirror
# the int arithmetic verified bit-exact in trainer/yane_to_binbo.py::yane_forward_int:
#   activation [0,1] real  <-> [0,127] int        (ClippedReLU, ACT_SCALE=127)
#   FT weight/bias         -> int16 (round * 127)
#   hidden affine weight   -> int8  (round * 64 = 2^kWeightScaleBits)
#   hidden affine bias     -> int32 (round * 127*64)
#   output is descaled by  >>6 between layers and / FV_SCALE(16) at the end.
QAT_ACT_SCALE = 127
QAT_FT_W_SCALE = 127
QAT_FT_B_SCALE = 127
QAT_HID_W_SCALE = 64
QAT_HID_B_SCALE = 127 * 64
QAT_WEIGHT_SCALE_BITS = 6
QAT_FV_SCALE = 16
QAT_I16_LIM = 32767
QAT_I8_LIM = 127
QAT_I32_LIM = 2 ** 31 - 1


def qat_out_scales(scale_cp: float) -> tuple[float, float]:
    """Output-layer (logit->cp) int scales. They fold FV_SCALE and scale_cp in so the
    deployed int eval ``o2 / FV_SCALE`` equals ``forward_logit * scale_cp`` (centipawns):
    o2 = FV_SCALE*scale_cp*logit, and h1_int = ACT_SCALE*h1_real, so
    w2_int = (FV_SCALE*scale_cp/ACT_SCALE)*w2_real, b2_int = (FV_SCALE*scale_cp)*b2_real."""
    return QAT_FV_SCALE * scale_cp / QAT_ACT_SCALE, QAT_FV_SCALE * scale_cp


def _fake_quant(t: torch.Tensor, scale: float, lim: int) -> torch.Tensor:
    """Straight-through fake quantization of a weight tensor: snap to the int grid
    (round * scale, clamp +/-lim, / scale) on the forward, identity gradient on the
    backward so the optimizer learns weights robust to the deployed rounding."""
    q = torch.clamp(torch.round(t * scale), -lim, lim) / scale
    return t + (q - t).detach()


def tied_embedding_weight(weight: torch.Tensor, collapse_map) -> torch.Tensor:
    """Gather a tied QATSharedNNUE embedding [total+1, fd] into the deployed [total, fd]
    (Task 059): row b = weight[collapse_map[b]], so collapsed groups become identical and
    the side rows (mapped to the pad row) become 0. The BNNQ/Rust forward then needs no
    knowledge of the collapse -- equal rows give the YaneuraOu-folded behaviour for free."""
    cm = torch.as_tensor(collapse_map, dtype=torch.long, device=weight.device)
    return weight[cm]


def fold_ft_factor(state: dict) -> dict:
    """Fold the factorizer (Task 125) into the real feature table: row k*FPK+p of the
    shared band gets ft_factor[p] added (the king-independent "P" part), the side / pad
    rows are untouched. Returns a new state without ``ft_factor``; a state without it is
    returned unchanged. Must run BEFORE the tie gather (training looks up the merged
    table at the representative rows)."""
    if "ft_factor" not in state:
        return state
    state = dict(state)
    factor = state.pop("ft_factor")
    w = state["feature_transform.weight"]
    fd = w.shape[1]
    band = w[:kp.VIEW_DIM].reshape(kp.SQUARE_NB, kp.FEATURES_PER_KING, fd) + factor.unsqueeze(0)
    state["feature_transform.weight"] = torch.cat([band.reshape(kp.VIEW_DIM, fd), w[kp.VIEW_DIM:]], dim=0)
    return state


def quantize_qat_state(state: dict, scale_cp: float) -> dict:
    """Quantize a QATSharedNNUE state_dict to the BNNQ integer grids (Task 058). Uses
    the SAME scales as the model's fake-quant, so the int forward reproduces the float
    forward. Returns numpy int arrays for the BNNQ writer (export_nnue.export_qat)."""
    if "ft_factor" in state:
        raise ValueError("fold_ft_factor() the state (and tie-gather it) before quantizing")
    def q(key: str, scale: float, lim: int, dt) -> np.ndarray:
        a = np.rint(state[key].detach().cpu().numpy().astype(np.float64) * scale)
        return np.clip(a, -lim, lim).astype(dt)

    ow, ob = qat_out_scales(scale_cp)
    return {
        "emb": q("feature_transform.weight", QAT_FT_W_SCALE, QAT_I16_LIM, np.int16),
        "ft_bias": q("ft_bias", QAT_FT_B_SCALE, QAT_I16_LIM, np.int16),
        "w0": q("l0.weight", QAT_HID_W_SCALE, QAT_I8_LIM, np.int8),
        "b0": q("l0.bias", QAT_HID_B_SCALE, QAT_I32_LIM, np.int32),
        "w1": q("l1.weight", QAT_HID_W_SCALE, QAT_I8_LIM, np.int8),
        "b1": q("l1.bias", QAT_HID_B_SCALE, QAT_I32_LIM, np.int32),
        "w2": q("l2.weight", ow, QAT_I8_LIM, np.int8),
        "b2": q("l2.bias", ob, QAT_I32_LIM, np.int32),
    }


def eval_to_winrate(eval_cp: torch.Tensor, scale_cp: float = DEFAULT_EVAL_SCALE_CP) -> torch.Tensor:
    return torch.sigmoid(eval_cp.to(torch.float32) / scale_cp)


@dataclass(frozen=True)
class SparseBatch:
    flat_indices: torch.Tensor
    offsets: torch.Tensor
    eval_cp: torch.Tensor


class MinimalNNUE(nn.Module):
    def __init__(
        self,
        total_features: int = kp.TOTAL_FEATURES,
        feature_dim: int = 32,
        hidden_dim: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.feature_transform = nn.EmbeddingBag(
            total_features,
            feature_dim,
            mode="sum",
            sparse=False,
        )
        # Dropout on the accumulator (embedding output), before the network. The
        # overfit lives in the huge embedding/representation (Task 035), so this
        # regularizes the representation that drives the eval and breaks
        # co-adaptation of its dimensions. It is training-only (identity in eval
        # mode) and has no parameters, so the exported network Linear layers
        # (network.0/2/4) and the Rust forward are unchanged.
        self.dropout = nn.Dropout(dropout)
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.feature_transform.weight, mean=0.0, std=0.01)
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, flat_indices: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        hidden = self.feature_transform(flat_indices, offsets)
        hidden = self.dropout(hidden)  # training-only; identity in eval mode
        return self.network(hidden).squeeze(-1)

    def predict_winrate(self, flat_indices: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(flat_indices, offsets))


class CanonicalNNUE(nn.Module):
    """Canonical HalfKP NNUE (Task 046, docs/kp_spec.md §7).

    Two perspectives are kept in *separate* accumulators and concatenated, instead
    of summed into one (MinimalNNUE). The own half (side-to-move view + side
    feature) and the opponent half (the canonical 180-rotated + owner-flipped view)
    each get their own EmbeddingBag-sum over the SAME shared embedding table, then
    the two ``feature_dim`` vectors are concatenated into a ``2*feature_dim`` input
    to the first Linear. This lets the first layer treat own/opponent features
    distinctly (a qualitative change, unlike widening feature_dim).

    The embedding table is the same shape as MinimalNNUE; only ``network.0``'s input
    width is ``2*feature_dim``. Exported as magic ``BNN2`` for the Rust 2-accumulator
    forward. Dropout (eval-mode identity) sits on the concatenated accumulator, so
    the exported Linear layers and Rust inference are unaffected.
    """

    def __init__(
        self,
        total_features: int = kp.TOTAL_FEATURES,
        feature_dim: int = 32,
        hidden_dim: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.feature_transform = nn.EmbeddingBag(
            total_features,
            feature_dim,
            mode="sum",
            sparse=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.network = nn.Sequential(
            nn.Linear(2 * feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.feature_transform.weight, mean=0.0, std=0.01)
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(
        self,
        own_flat: torch.Tensor,
        own_offsets: torch.Tensor,
        opp_flat: torch.Tensor,
        opp_offsets: torch.Tensor,
    ) -> torch.Tensor:
        own = self.feature_transform(own_flat, own_offsets)
        opp = self.feature_transform(opp_flat, opp_offsets)
        hidden = torch.cat([own, opp], dim=1)
        hidden = self.dropout(hidden)  # training-only; identity in eval mode
        return self.network(hidden).squeeze(-1)


class QATSharedNNUE(nn.Module):
    """Quantization-aware shared HalfKP (Task 058).

    Same own/opp shared-EmbeddingBag-sum as ``CanonicalNNUE`` (BNN3), but rebuilt to
    match YaneuraOu's *integer* forward so a trained eval survives quantization undistorted:

    - an explicit ``ft_bias`` is added to EACH perspective BEFORE a ClippedReLU[0,1]
      (YaneuraOu's FT clamp[0,127]); unlike the float path it cannot be folded into the
      first affine, because the clamp is non-linear;
    - ClippedReLU[0,1] (not plain ReLU) between affines -> the activations live in the
      same bounded range as the deployed int8 [0,127];
    - optional ``fake_quant`` snaps weights (and the bounded activations) to the int grids
      from ``quantize_qat_state`` with straight-through gradients (QAT). With it off the
      forward is the continuous float reference used to measure the quantization residual
      (post-training quantization).
    - optional ``factorize`` (Task 125): a king-independent table ``ft_factor[p]``
      (FEATURES_PER_KING x fd, init 0) is added to every king's row for p, so the
      effective weight is W[k, p] + V[p]. V learns from every position with p, W learns
      the king-specific remainder. Fake quant and the export act on the merged table
      (``fold_ft_factor``), so the deployed network and its forward are unchanged.

    ``forward`` returns a logit like ``CanonicalNNUE``; ``pred = logit * scale_cp`` is the
    centipawn eval, equal to the deployed int forward's ``o2 / FV_SCALE``. The network
    signature (own_flat, own_off, opp_flat, opp_off) matches the trainer's ``shared`` path,
    so it drops into ``run_batches`` unchanged. Exported as magic ``BNNQ``.
    """

    def __init__(
        self,
        total_features: int = kp.CANONICAL_SHARED_TOTAL,
        feature_dim: int = 256,
        hidden_dim: int = 32,
        dropout: float = 0.0,
        fake_quant: bool = False,
        scale_cp: float = DEFAULT_EVAL_SCALE_CP,
        tie_collapse_map=None,
        factorize: bool = False,
        sparse_ft: bool = False,
        hidden_dim2: int | None = None,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.hidden_dim2 = hidden_dim if hidden_dim2 is None else hidden_dim2  # Task 149
        if min(feature_dim, hidden_dim, self.hidden_dim2) < 1:
            raise ValueError("QAT dimensions must be positive")
        self.fake_quant = fake_quant
        self.scale_cp = scale_cp
        self.sparse_ft = sparse_ft  # Task 135: build the effective table for this batch only.
        # Tie (Task 059): features that share a YaneuraOu HalfKP slot (the gold collapse)
        # are mapped to one representative row, and the 2 side rows to a frozen PAD row, so
        # the trained eval is exactly representable in YaneuraOu's fe_end=1548 / no-side
        # format (bit-exact writable). collapse_map[binbo_idx] -> row in [0, total]; pad=total.
        self.tied = tie_collapse_map is not None
        emb_rows = total_features + 1 if self.tied else total_features
        pad_idx = total_features if self.tied else None
        self.pad_idx = pad_idx
        if self.tied:
            self.register_buffer("collapse_map", torch.as_tensor(tie_collapse_map, dtype=torch.long))
        self.feature_transform = nn.EmbeddingBag(emb_rows, feature_dim, mode="sum",
                                                 sparse=False, padding_idx=pad_idx)
        self.ft_bias = nn.Parameter(torch.zeros(feature_dim))
        self.factorize = factorize
        if factorize:
            self.ft_factor = nn.Parameter(torch.zeros(kp.FEATURES_PER_KING, feature_dim))
        self.dropout = nn.Dropout(dropout)
        self.l0 = nn.Linear(2 * feature_dim, hidden_dim)
        self.l1 = nn.Linear(hidden_dim, self.hidden_dim2)
        self.l2 = nn.Linear(self.hidden_dim2, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.feature_transform.weight, mean=0.0, std=0.01)
        if self.pad_idx is not None:
            self.feature_transform.weight.data[self.pad_idx].zero_()  # PAD stays 0 (side feature)
        nn.init.zeros_(self.ft_bias)
        for lin in (self.l0, self.l1, self.l2):
            nn.init.kaiming_uniform_(lin.weight, nonlinearity="relu")
            nn.init.zeros_(lin.bias)

    def ft_weight(self) -> torch.Tensor:
        """The effective feature table (merged with the factorizer when on)."""
        w = self.feature_transform.weight
        if not self.factorize:
            return w
        fd = w.shape[1]
        band = w[:kp.VIEW_DIM].reshape(kp.SQUARE_NB, kp.FEATURES_PER_KING, fd) + self.ft_factor.unsqueeze(0)
        return torch.cat([band.reshape(kp.VIEW_DIM, fd), w[kp.VIEW_DIM:]], dim=0)

    def _qw(self, w: torch.Tensor, scale: float, lim: int) -> torch.Tensor:
        return _fake_quant(w, scale, lim) if self.fake_quant else w

    def _clamped_act(self, pre: torch.Tensor) -> torch.Tensor:
        """ClippedReLU[0,1] (proper clamp gradient), then, when fake_quant, snap to the
        1/ACT_SCALE grid with a straight-through gradient. Uses FLOOR (not round) to model
        YaneuraOu's `>>weight_scale_bits` exactly, so the trainer optimizes the deployed
        behaviour (no systematic offset between the QAT forward and the int deployment)."""
        x = torch.clamp(pre, 0.0, 1.0)
        if self.fake_quant:
            q = torch.floor(x * QAT_ACT_SCALE) / QAT_ACT_SCALE
            x = x + (q - x).detach()
        return x

    def _ft_half(self, flat: torch.Tensor, off: torch.Tensor, ftw: torch.Tensor, ftb: torch.Tensor) -> torch.Tensor:
        if self.tied:
            flat = self.collapse_map[flat]  # fold collapsed/side features onto shared rows
        acc = F.embedding_bag(flat, ftw, off, mode="sum", padding_idx=self.pad_idx) + ftb
        # FT activation is exact on the grid once ftw/ftb are quantized, so just clamp.
        return torch.clamp(acc, 0.0, 1.0)

    def forward_features(
        self,
        own_flat: torch.Tensor,
        own_off: torch.Tensor,
        opp_flat: torch.Tensor,
        opp_off: torch.Tensor,
    ) -> torch.Tensor:
        if self.sparse_ft:
            return self._touched_features(own_flat, own_off, opp_flat, opp_off)
        ftw = self._qw(self.ft_weight(), QAT_FT_W_SCALE, QAT_I16_LIM)
        ftb = self._qw(self.ft_bias, QAT_FT_B_SCALE, QAT_I16_LIM)
        own = self._ft_half(own_flat, own_off, ftw, ftb)
        opp = self._ft_half(opp_flat, opp_off, ftw, ftb)
        return self.dropout(torch.cat([own, opp], dim=1))

    def _touched_features(self, own_flat, own_off, opp_flat, opp_off):
        """Task 135: build the effective table only for the batch's unique touched rows.

        torch.unique costs one small host sync per step, but a 1024-row batch touches ~14k
        unique rows of its 78,848 feature slots, so a fixed-capacity table would do ~5.6x
        the gather / merge / fake-quant work (measured: forward 8.4 ms with capacity)."""
        flat = torch.cat([own_flat, opp_flat])
        if self.tied:
            flat = self.collapse_map[flat]
        rows, inverse = torch.unique(flat, return_inverse=True)
        w = F.embedding(rows, self.feature_transform.weight)
        if self.factorize:
            factor = F.embedding(rows.remainder(kp.FEATURES_PER_KING), self.ft_factor)
            w = w + factor * (rows < kp.VIEW_DIM).unsqueeze(1)
        w = self._qw(w, QAT_FT_W_SCALE, QAT_I16_LIM)
        pad = rows.numel()  # the appended zero row
        w = torch.cat([w, w.new_zeros((1, w.shape[1]))])
        if self.pad_idx is not None:
            inverse = torch.where(flat == self.pad_idx, pad, inverse)
        ftb = self._qw(self.ft_bias, QAT_FT_B_SCALE, QAT_I16_LIM)
        cut = own_flat.numel()
        own = F.embedding_bag(inverse[:cut], w, own_off, mode="sum", padding_idx=pad)
        opp = F.embedding_bag(inverse[cut:], w, opp_off, mode="sum", padding_idx=pad)
        return self.dropout(torch.cat([torch.clamp(own + ftb, 0.0, 1.0),
                                       torch.clamp(opp + ftb, 0.0, 1.0)], dim=1))

    def forward_hidden_from_features(self, features: torch.Tensor) -> torch.Tensor:
        w0 = self._qw(self.l0.weight, QAT_HID_W_SCALE, QAT_I8_LIM)
        b0 = self._qw(self.l0.bias, QAT_HID_B_SCALE, QAT_I32_LIM)
        h0 = self._clamped_act(F.linear(features, w0, b0))
        w1 = self._qw(self.l1.weight, QAT_HID_W_SCALE, QAT_I8_LIM)
        b1 = self._qw(self.l1.bias, QAT_HID_B_SCALE, QAT_I32_LIM)
        return self._clamped_act(F.linear(h0, w1, b1))

    def forward_hidden(
        self,
        own_flat: torch.Tensor,
        own_off: torch.Tensor,
        opp_flat: torch.Tensor,
        opp_off: torch.Tensor,
    ) -> torch.Tensor:
        features = self.forward_features(own_flat, own_off, opp_flat, opp_off)
        return self.forward_hidden_from_features(features)

    def forward(
        self,
        own_flat: torch.Tensor,
        own_off: torch.Tensor,
        opp_flat: torch.Tensor,
        opp_off: torch.Tensor,
    ) -> torch.Tensor:
        h1 = self.forward_hidden(own_flat, own_off, opp_flat, opp_off)
        return self.forward_output(h1)

    def forward_output(self, h1: torch.Tensor) -> torch.Tensor:
        ow, ob = qat_out_scales(self.scale_cp)
        w2 = self._qw(self.l2.weight, ow, QAT_I8_LIM)
        b2 = self._qw(self.l2.bias, ob, QAT_I32_LIM)
        return F.linear(h1, w2, b2).squeeze(-1)


class KppNNUE(nn.Module):
    """HalfKP shared + king-zone KPP (Task 053). Two embedding tables (the shared KP
    band and the KPP-zone pair table) feed FOUR EmbeddingBag-sum accumulators
    (KP own/opp, KPP own/opp) concatenated into a 4*feature_dim input. Empty KPP
    bags (no in-zone pairs) sum to zero, so no padding row is needed."""

    def __init__(
        self,
        kp_total: int = kp.CANONICAL_SHARED_TOTAL,
        kpp_total: int = kp.KPP_ZONE_TOTAL,
        feature_dim: int = 8,
        hidden_dim: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.kp_emb = nn.EmbeddingBag(kp_total, feature_dim, mode="sum", sparse=False)
        self.kpp_emb = nn.EmbeddingBag(kpp_total, feature_dim, mode="sum", sparse=False)
        self.dropout = nn.Dropout(dropout)
        self.network = nn.Sequential(
            nn.Linear(4 * feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.kp_emb.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.kpp_emb.weight, mean=0.0, std=0.01)
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(
        self,
        kp_own_flat: torch.Tensor, kp_own_off: torch.Tensor,
        kp_opp_flat: torch.Tensor, kp_opp_off: torch.Tensor,
        pp_own_flat: torch.Tensor, pp_own_off: torch.Tensor,
        pp_opp_flat: torch.Tensor, pp_opp_off: torch.Tensor,
    ) -> torch.Tensor:
        kp_own = self.kp_emb(kp_own_flat, kp_own_off)
        kp_opp = self.kp_emb(kp_opp_flat, kp_opp_off)
        pp_own = self.kpp_emb(pp_own_flat, pp_own_off)
        pp_opp = self.kpp_emb(pp_opp_flat, pp_opp_off)
        hidden = torch.cat([kp_own, kp_opp, pp_own, pp_opp], dim=1)
        hidden = self.dropout(hidden)
        return self.network(hidden).squeeze(-1)


def collate_sparse_batch(
    examples: list[kp.KpExample],
    *,
    device: torch.device | str = "cpu",
) -> SparseBatch:
    if not examples:
        raise ValueError("empty batch")

    flat_indices: list[int] = []
    offsets: list[int] = []
    cursor = 0
    evals: list[int] = []
    for example in examples:
        offsets.append(cursor)
        flat_indices.extend(example.indices)
        cursor += len(example.indices)
        evals.append(example.eval)

    return SparseBatch(
        flat_indices=torch.tensor(flat_indices, dtype=torch.long, device=device),
        offsets=torch.tensor(offsets, dtype=torch.long, device=device),
        eval_cp=torch.tensor(evals, dtype=torch.float32, device=device),
    )
