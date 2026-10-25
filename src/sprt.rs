//! Trinomial (BayesElo) SPRT + Elo/CI for the search-engine match (Task 060).
//!
//! A sequential probability ratio test that models Win/Draw/Loss (so draws are handled
//! properly, not collapsed to 0.5) and stops as soon as the log-likelihood ratio crosses
//! a Wald bound — no naive peeking. All counts are from binbo's perspective.

#[derive(Debug, Clone, Copy)]
pub struct SprtConfig {
    pub elo0: f64,
    pub elo1: f64,
    pub alpha: f64,
    pub beta: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SprtVerdict {
    AcceptH0,
    AcceptH1,
    Continue,
}

impl SprtVerdict {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::AcceptH0 => "accept_h0",
            Self::AcceptH1 => "accept_h1",
            Self::Continue => "continue",
        }
    }
}

/// Log-likelihood ratio and verdict for the current (W, D, L). The BayesElo nuisance
/// parameter `drawelo` is estimated from the data (shared by both hypotheses), with
/// Haldane (+0.5) smoothing so it stays finite when a result class is still empty.
pub fn sprt(w: u64, d: u64, l: u64, cfg: &SprtConfig) -> (f64, SprtVerdict) {
    let n = (w + d + l) as f64;
    if n < 1.0 {
        return (0.0, SprtVerdict::Continue);
    }
    let denom = n + 1.5;
    let pw = (w as f64 + 0.5) / denom;
    let pl = (l as f64 + 0.5) / denom;
    let drawelo = 200.0 * (((1.0 - pl) / pl) * ((1.0 - pw) / pw)).log10();
    let probs = |elo: f64| -> (f64, f64, f64) {
        let pw = 1.0 / (1.0 + 10f64.powf((drawelo - elo) / 400.0));
        let pl = 1.0 / (1.0 + 10f64.powf((drawelo + elo) / 400.0));
        (pw.max(1e-12), (1.0 - pw - pl).max(1e-12), pl.max(1e-12))
    };
    let (w0, d0, l0) = probs(cfg.elo0);
    let (w1, d1, l1) = probs(cfg.elo1);
    let llr = w as f64 * (w1 / w0).ln() + d as f64 * (d1 / d0).ln() + l as f64 * (l1 / l0).ln();
    let upper = ((1.0 - cfg.beta) / cfg.alpha).ln();
    let lower = (cfg.beta / (1.0 - cfg.alpha)).ln();
    let verdict = if llr >= upper {
        SprtVerdict::AcceptH1
    } else if llr <= lower {
        SprtVerdict::AcceptH0
    } else {
        SprtVerdict::Continue
    };
    (llr, verdict)
}

/// Wald LLR bounds (lower, upper) for logging.
pub fn bounds(cfg: &SprtConfig) -> (f64, f64) {
    ((cfg.beta / (1.0 - cfg.alpha)).ln(), ((1.0 - cfg.beta) / cfg.alpha).ln())
}

/// Score rate `(W + 0.5 D)/N` and a 95% CI using the trinomial per-game variance (each
/// game scores 0/0.5/1), so draws reduce the variance correctly.
pub fn score_ci(w: u64, d: u64, l: u64) -> (f64, f64, f64) {
    let n = (w + d + l) as f64;
    if n < 1.0 {
        return (0.5, 0.0, 1.0);
    }
    let score = (w as f64 + 0.5 * d as f64) / n;
    let e_x2 = (w as f64 + 0.25 * d as f64) / n; // E[x^2], x in {0,0.5,1}
    let var = (e_x2 - score * score).max(0.0);
    let se = (var / n).sqrt();
    let lo = (score - 1.96 * se).max(0.0);
    let hi = (score + 1.96 * se).min(1.0);
    (score, lo, hi)
}

/// Logistic Elo point estimate from a score in (0,1): `-400*log10(1/s - 1)`.
pub fn elo_from_score(score: f64) -> f64 {
    let s = score.clamp(1e-6, 1.0 - 1e-6);
    -400.0 * (1.0 / s - 1.0).log10()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn even_score_is_zero_elo_and_continues() {
        let (s, lo, hi) = score_ci(10, 0, 10);
        assert!((s - 0.5).abs() < 1e-9);
        assert!(lo < 0.5 && hi > 0.5);
        assert!((elo_from_score(0.5)).abs() < 1e-9);
        let cfg = SprtConfig { elo0: 0.0, elo1: -50.0, alpha: 0.05, beta: 0.05 };
        let (_, v) = sprt(10, 0, 10, &cfg);
        assert_eq!(v, SprtVerdict::Continue);
    }

    #[test]
    fn lopsided_loss_accepts_h1_worse() {
        // binbo loses almost everything -> "binbo >=50 elo worse" (H1) accepted.
        let cfg = SprtConfig { elo0: 0.0, elo1: -50.0, alpha: 0.05, beta: 0.05 };
        let (llr, v) = sprt(1, 4, 60, &cfg);
        assert!(llr > 0.0); // LLR accumulates toward H1 (elo1 = worse) as binbo loses
        assert_eq!(v, SprtVerdict::AcceptH1);
        assert!(elo_from_score(score_ci(1, 4, 60).0) < -50.0);
    }
}
