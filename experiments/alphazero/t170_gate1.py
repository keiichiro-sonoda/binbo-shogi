#!/usr/bin/env python3
"""Task 170: gate 1 of the GPU idea -- does a small learned predictor of "value of extra search" beat search-only features?

Data (no new search): the stable, non-mate roots of Task 169 (work/strategy/gate0, round-1 games) and Task 167
(work/strategy/probe13/main, round-2 games); all roots are from distinct games. Target per root: gain = loss_n - loss_live_4n
(N = 100k nodes, then +3N keeping the hash table), clipped to [-300, 300] cp for fitting only.
Allocation: every root gets N; the 20% with the highest predicted gain in each test fold get the +3N continuation
(mean budget 1.6N, the same as the uniform 1.6N baseline). Loss = mean reference loss (cp).
Predictors (fixed, nothing tuned on test data): ridge regression on standardised features, lambda = 10.
  S  search-only, free from the N search: depth, PV changes, |N score|, score swing over the last 4 iterations,
     legal move count, ply
  G  S + GPU policy (old WS-T): top-1 agrees with the N move, top-1 probability, probability of the N move,
     entropy of the top-8, rank of the N move in the policy (9 if absent)
5-fold cross-validation split by game (seed 20271700); out-of-fold predictions; game-level bootstrap (2000, seed 20271701).
PASS (fixed before running): G's reduction vs uniform >= 4 cp with 95% CI lower > 0, AND G - S paired difference > 0 with
95% CI lower > 0. WEAK: G passes the first part only. FAIL otherwise.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

DATA = Path.home() / "binbo-shogi-data/work/strategy"
SOURCES = {"r1": DATA / "gate0", "r2": DATA / "probe13/main"}
FRAC, LAM, FOLDS, BOOT = 0.2, 10.0, 5, 2000


def rows():
    out = []
    for tag, d in SOURCES.items():
        rep = json.loads((d / "report.json").read_text())["roots"]
        pol = json.loads((d / "policy.json").read_text())["roots"]
        roots = {r["root_id"]: r for r in json.loads((d / "roots.json").read_text())["roots"]}
        for r in rep:
            if not r["stable"] or r["mate"]:
                continue
            s = json.loads((d / "search" / f"{r['root_id']}.json").read_text())
            b = s["budgets"]["100000"]
            its = b["iterations"] if isinstance(b["iterations"], list) else []
            sc = [x["score_cp"] for x in its[-4:] if x.get("score_cp") is not None]
            p = pol[r["root_id"]]
            tm = p["top_moves"]
            probs = {m["move"]: m["probability"] for m in tm}
            pv = np.array([m["probability"] for m in tm])
            pv = pv / pv.sum()
            rank = next((i + 1 for i, m in enumerate(tm) if m["move"] == b["bestmove"]), 9)
            search = [b["depth"], r["pv_changes"], min(abs(b["score_cp"] or 0), 2000) / 100,
                      (max(sc) - min(sc)) / 100 if len(sc) > 1 else 0.0, p["legal_count"], roots[r["root_id"]]["ply"]]
            gpu = [float(tm[0]["move"] == b["bestmove"]), tm[0]["probability"], probs.get(b["bestmove"], 0.0),
                   float(-(pv * np.log(pv + 1e-12)).sum()), rank]
            out.append(dict(game=f"{tag}:{r['game_id']}", ln=r["loss_n"], lu=r["loss_uniform"], ll=r["loss_live_4n"],
                            S=search, G=search + gpu))
    return out


def ridge_fit(x, y):
    mu, sd = x.mean(0), x.std(0) + 1e-9
    z = (x - mu) / sd
    w = np.linalg.solve(z.T @ z + LAM * np.eye(z.shape[1]), z.T @ (y - y.mean()))
    return lambda q: ((q - mu) / sd) @ w + y.mean()


def alloc_loss(pred, ln, ll):
    k = int(round(FRAC * len(pred)))
    top = np.argsort(-pred)[:k]
    loss = ln.copy()
    loss[top] = ll[top]
    return loss


def main():
    R = rows()
    games = sorted({r["game"] for r in R})
    rng = np.random.default_rng(20271700)
    fold_of = dict(zip(rng.permutation(games), [i % FOLDS for i in range(len(games))]))
    ln = np.array([r["ln"] for r in R], float)
    lu = np.array([r["lu"] for r in R], float)
    ll = np.array([r["ll"] for r in R], float)
    y = np.clip(ln - ll, -300, 300)
    folds = np.array([fold_of[r["game"]] for r in R])
    loss = {}
    for key in ("S", "G"):
        X = np.array([r[key] for r in R], float)
        out = np.empty(len(R))
        for f in range(FOLDS):
            tr, te = folds != f, folds == f
            pred = ridge_fit(X[tr], y[tr])(X[te])
            out[te] = alloc_loss(pred, ln[te], ll[te])
        loss[key] = out
    # gate 0 rule for reference: agree -> N, else extra to disagreements (same mean budget)
    agree = np.array([r["G"][6] for r in R]) > 0.5
    q = (1.6 * len(R) - agree.sum() - 1.6 * (~agree).sum()) / (2.4 * (~agree).sum())
    rng2 = np.random.default_rng(20271701)
    idx = np.arange(len(R))
    gid = np.array([games.index(r["game"]) for r in R])
    by_game = [idx[gid == g] for g in range(len(games))]
    def stats(f):
        pt = f(idx)
        bs = []
        for _ in range(BOOT):
            pick = np.concatenate([by_game[g] for g in rng2.integers(0, len(games), len(games))])
            bs.append(f(pick))
        bs = np.sort(bs)
        return pt, bs[int(0.025 * BOOT)], bs[int(0.975 * BOOT) - 1]
    res = {
        "S_reduction": stats(lambda i: lu[i].mean() - loss["S"][i].mean()),
        "G_reduction": stats(lambda i: lu[i].mean() - loss["G"][i].mean()),
        "G_minus_S": stats(lambda i: loss["S"][i].mean() - loss["G"][i].mean()),
        "oracle_live_reduction": stats(lambda i: lu[i].mean() - alloc_loss(ln[i] - ll[i], ln[i], ll[i]).mean()),
    }
    res["uniform_loss"] = float(lu.mean())
    res["roots"] = len(R)
    res["gate0_rule_reference_q"] = float(q)
    g, d = res["G_reduction"], res["G_minus_S"]
    verdict = "PASS" if (g[0] >= 4 and g[1] > 0 and d[0] > 0 and d[1] > 0) else "WEAK" if (g[0] >= 4 and g[1] > 0) else "FAIL"
    res["verdict"] = verdict
    out = DATA / "gate1_results.json"
    out.write_text(json.dumps(res, indent=2))
    for k, v in res.items():
        print(k, [round(x, 2) for x in v] if isinstance(v, tuple) else v)
    print(f"T170_VERDICT {verdict} results={out}")


if __name__ == "__main__":
    sys.exit(main())
