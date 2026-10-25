# Task 078 M3: faithful Python port of src/sprt.rs (trinomial BayesElo SPRT +
# trinomial score CI + logistic elo). Same H0/H1 semantics as the 066-072 adoption
# gates. Running `python m3_sprt.py` reproduces the two Rust unit tests.

import math

ACCEPT_H0 = "accept_h0"
ACCEPT_H1 = "accept_h1"
CONTINUE = "continue"


def sprt(w, d, l, elo0, elo1, alpha=0.05, beta=0.05):
    """Return (llr, verdict). BayesElo drawelo nuisance estimated from data with
    Haldane (+0.5) smoothing, shared by both hypotheses (== sprt.rs)."""
    n = float(w + d + l)
    if n < 1.0:
        return 0.0, CONTINUE
    denom = n + 1.5
    pw = (w + 0.5) / denom
    pl = (l + 0.5) / denom
    drawelo = 200.0 * math.log10(((1.0 - pl) / pl) * ((1.0 - pw) / pw))

    def probs(elo):
        pw_ = 1.0 / (1.0 + 10.0 ** ((drawelo - elo) / 400.0))
        pl_ = 1.0 / (1.0 + 10.0 ** ((drawelo + elo) / 400.0))
        return max(pw_, 1e-12), max(1.0 - pw_ - pl_, 1e-12), max(pl_, 1e-12)

    w0, d0, l0 = probs(elo0)
    w1, d1, l1 = probs(elo1)
    llr = w * math.log(w1 / w0) + d * math.log(d1 / d0) + l * math.log(l1 / l0)
    lower, upper = bounds(alpha, beta)
    verdict = ACCEPT_H1 if llr >= upper else (ACCEPT_H0 if llr <= lower else CONTINUE)
    return llr, verdict


def bounds(alpha=0.05, beta=0.05):
    return math.log(beta / (1.0 - alpha)), math.log((1.0 - beta) / alpha)


def score_ci(w, d, l):
    """Score (W+0.5D)/N with 95% CI from the trinomial per-game variance."""
    n = float(w + d + l)
    if n < 1.0:
        return 0.5, 0.0, 1.0
    score = (w + 0.5 * d) / n
    e_x2 = (w + 0.25 * d) / n
    var = max(e_x2 - score * score, 0.0)
    se = math.sqrt(var / n)
    return score, max(score - 1.96 * se, 0.0), min(score + 1.96 * se, 1.0)


def elo_from_score(score):
    s = min(max(score, 1e-6), 1.0 - 1e-6)
    return -400.0 * math.log10(1.0 / s - 1.0)


if __name__ == "__main__":
    # Rust unit test 1: even_score_is_zero_elo_and_continues
    s, lo, hi = score_ci(10, 0, 10)
    assert abs(s - 0.5) < 1e-9 and lo < 0.5 < hi
    assert abs(elo_from_score(0.5)) < 1e-9
    _, v = sprt(10, 0, 10, elo0=0.0, elo1=-50.0)
    assert v == CONTINUE, v
    # Rust unit test 2: lopsided_loss_accepts_h1_worse
    llr, v = sprt(1, 4, 60, elo0=0.0, elo1=-50.0)
    assert llr > 0.0 and v == ACCEPT_H1, (llr, v)
    assert elo_from_score(score_ci(1, 4, 60)[0]) < -50.0
    print(f"m3_sprt: both sprt.rs unit tests reproduced (llr example: {llr:.3f})")
