#!/usr/bin/env python3
"""Task 048 efficacy probe (Python-only, NO Rust parity yet): does adding
king-zone piece-PAIR features (a feasible reduced KPP) improve val 度合い over the
true-canonical HalfKP (shared) baseline, on the SAME positions?

- KP part: true canonical shared HalfKP (own 39, opp 38) over kp.CANONICAL_SHARED_TOTAL.
- KPP-zone part: for each perspective king, board pieces within Chebyshev<=2 (5x5
  zone); unordered pairs of their zone-state (zone_rel_sq 0..24 x ptype 0..12 x
  owner 0..1 = 650), shared single table (opp half = 180-rotated zone + owner flip).
  No king-square dimension (zone-relative already encodes position rel. to king) ->
  tiny table C(650,2)=210925. This is a SUBSET of full king-zone KPP (which keeps the
  king square, 17M rows, 0.51GB) sufficient to test efficacy cheaply.
- Model: KP own/opp + KPP own/opp = 4 accumulators concat -> Linear(4*fd,h)->...->1.
  Baseline ablation: zero the KPP accumulators (== KP-shared only) by --no-kpp.
"""
import sys, os, time, json
import numpy as np, torch
from torch import nn
sys.path.insert(0, os.path.expanduser("~/GitHub/shogi/binbo-shogi"))
import cshogi
from trainer import kp

ZS = 25 * 13 * 2          # 650 zone-states
KPP_TOTAL = ZS * (ZS - 1) // 2   # 210925 unordered pairs
SCALE = 600.0

def pair_index(a, b):     # unordered, a!=b, 0-based combinatorial rank
    if a > b: a, b = b, a
    return a * ZS - (a * (a + 1)) // 2 + (b - a - 1)

def zone_state(zone_rel, ptype_id, rel_owner):
    return (zone_rel * 13 + ptype_id) * 2 + rel_owner

def cheb(a, b):
    return max(abs(a // 9 - b // 9), abs(a % 9 - b % 9))

BOARD_ID = kp.BOARD_PIECE_TO_ID  # piece_type -> 0..12

def kpp_zone_halves(board):
    """Return (own_pairs, opp_pairs) lists of KPP-zone pair indices."""
    turn = board.turn
    res = []
    for anchor_color in (turn, kp.opposite_color(turn)):
        king = board.king_square(anchor_color)
        kf, kr = king // 9, king % 9
        states = []
        for sq, p in enumerate(board.pieces):
            if p == cshogi.NONE or (p & 0x0F) == cshogi.KING:
                continue
            if cheb(sq, king) > 2:
                continue
            df, dr = sq // 9 - kf, sq % 9 - kr           # -2..2
            zone_rel = (df + 2) * 5 + (dr + 2)           # 0..24
            ptype = p & 0x0F
            pid = BOARD_ID[ptype]
            color = cshogi.WHITE if p >= cshogi.WPAWN else cshogi.BLACK
            rel_owner = 0 if color == turn else 1
            # opponent perspective: rotate zone 180 + flip owner (shared table)
            if anchor_color != turn:
                zone_rel = 24 - zone_rel
                rel_owner = 1 - rel_owner
            states.append(zone_state(zone_rel, pid, rel_owner))
        pairs = [pair_index(states[i], states[j])
                 for i in range(len(states)) for j in range(i + 1, len(states))]
        res.append(sorted(set(pairs)))
    return res[0], res[1]


def load(files, stride=1):
    """Pre-extract ragged KP + KPP features + eval for all positions."""
    kp_own, kp_opp, pp_own, pp_opp, evals = [], [], [], [], []
    for f in files:
        for e in np.fromfile(f, dtype=cshogi.HuffmanCodedPosAndEval)[::stride]:
            b = cshogi.Board(); b.set_hcp(e['hcp'])
            o, p = kp.canonical_shared_halves(b)
            po, pp = kpp_zone_halves(b)
            kp_own.append(np.array(o, np.int64)); kp_opp.append(np.array(p, np.int64))
            pp_own.append(np.array(po, np.int64)); pp_opp.append(np.array(pp, np.int64))
            evals.append(int(e['eval']))
    return kp_own, kp_opp, pp_own, pp_opp, np.array(evals, np.float32)


class KPPModel(nn.Module):
    def __init__(self, fd=8, h=16, use_kpp=True):
        super().__init__()
        self.use_kpp = use_kpp
        self.kp_emb = nn.EmbeddingBag(kp.CANONICAL_SHARED_TOTAL, fd, mode="sum")
        self.pp_emb = nn.EmbeddingBag(KPP_TOTAL, fd, mode="sum")
        nin = 4 * fd if use_kpp else 2 * fd
        self.net = nn.Sequential(nn.Linear(nin, h), nn.ReLU(),
                                 nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
        nn.init.normal_(self.kp_emb.weight, 0, 0.01); nn.init.normal_(self.pp_emb.weight, 0, 0.01)
        for m in self.net:
            if isinstance(m, nn.Linear): nn.init.kaiming_uniform_(m.weight, nonlinearity="relu"); nn.init.zeros_(m.bias)

    def forward(self, kof, kos, kpf, kps, pof, pos, ppf, pps):
        ko = self.kp_emb(kof, kos); kpv = self.kp_emb(kpf, kps)
        if self.use_kpp:
            po = self.pp_emb(pof, pos); pp = self.pp_emb(ppf, pps)
            x = torch.cat([ko, kpv, po, pp], 1)
        else:
            x = torch.cat([ko, kpv], 1)
        return self.net(x).squeeze(-1)


def batch(idx_lists, rows, dev):
    flat = np.concatenate([idx_lists[r] for r in rows]) if len(rows) else np.zeros(0, np.int64)
    offs = np.zeros(len(rows), np.int64)
    c = 0
    for i, r in enumerate(rows):
        offs[i] = c; c += len(idx_lists[r])
    return (torch.from_numpy(flat).to(dev), torch.from_numpy(offs).to(dev))


def main():
    use_kpp = "--no-kpp" not in sys.argv
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    teach = os.path.expanduser("~/binbo-shogi-data/teacher/run1m")
    import glob
    shards = sorted(glob.glob(teach + "/shard_00*.hcpe"))
    train_files, val_files = shards[:95], shards[95:100]
    t0 = time.time()
    tr = load(train_files); va = load(val_files)
    print(f"extract: train {len(tr[4])} val {len(va[4])} in {time.time()-t0:.0f}s "
          f"(~{(time.time()-t0)/(len(tr[4])+len(va[4]))*1e6:.0f} us/pos)")
    ppn = np.array([len(x) for x in tr[2]] + [len(x) for x in tr[3]])
    print(f"KPP-zone active/half: mean {ppn.mean():.1f} max {ppn.max()}  (KP/half ~38)")
    tr_e = torch.from_numpy(np.clip(tr[4], -5000, 5000)).to(dev)
    va_e = torch.from_numpy(np.clip(va[4], -5000, 5000)).to(dev)
    va_raw = va[4]
    model = KPPModel(use_kpp=use_kpp).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.MSELoss()
    n = len(tr[4]); bs = 1024
    g = torch.Generator().manual_seed(20260621)
    print(f"use_kpp={use_kpp} device={dev} n_train={n} KPP_TOTAL={KPP_TOTAL}")
    best = 1e18
    for ep in range(1, 13):
        model.train(); order = torch.randperm(n, generator=g).numpy(); t1 = time.time()
        for s in range(0, n, bs):
            rows = order[s:s+bs]
            args = (*batch(tr[0], rows, dev), *batch(tr[1], rows, dev),
                    *batch(tr[2], rows, dev), *batch(tr[3], rows, dev))
            opt.zero_grad(set_to_none=True)
            pred = model(*args) * SCALE
            loss = crit(pred, tr_e[rows]); loss.backward(); opt.step()
        # val
        model.eval(); preds = np.zeros(len(va[4]), np.float32)
        with torch.no_grad():
            for s in range(0, len(va[4]), 4096):
                rows = np.arange(s, min(s+4096, len(va[4])))
                args = (*batch(va[0], rows, dev), *batch(va[1], rows, dev),
                        *batch(va[2], rows, dev), *batch(va[3], rows, dev))
                preds[rows] = (model(*args)*SCALE).cpu().numpy()
        vl = float(np.mean((preds - va_e.cpu().numpy())**2))
        dec = np.abs(va_raw) >= 1500
        pdec = np.abs(preds[dec]).mean(); tdec = np.abs(va_raw[dec]).mean()
        corr = np.corrcoef(preds, va_e.cpu().numpy())[0,1]
        best = min(best, vl)
        print(f"ep{ep} val_loss={vl:.0f} val_dec_PRED={pdec:.0f}/{tdec:.0f} ({100*pdec/tdec:.1f}%) corr={corr:.4f} {time.time()-t1:.1f}s")
    print(f"BEST val_loss={best:.0f}")

if __name__ == "__main__":
    main()
