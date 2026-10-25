//! NNUE forward pass for inference, numerically matching the Python
//! `trainer/nnue.py::MinimalNNUE` it loads its weights from.
//!
//! Architecture (re-confirmed from the trained checkpoint, not assumed):
//! `EmbeddingBag(total_features, feature_dim, mode=sum)` -> `Linear(feature_dim,
//! hidden_dim)` -> ReLU -> `Linear(hidden_dim, hidden_dim)` -> ReLU ->
//! `Linear(hidden_dim, 1)`. The output is a logit (sigmoid -> win rate).
//!
//! Weights are loaded from the self-describing binary written by
//! `trainer/export_nnue.py` (see that file for the exact byte layout). All
//! computation is f32 to mirror the Python model's dtype; cross-checked against
//! Python in `trainer/verify_nnue_rust.py`.

use std::fs::File;
use std::io::Read;
use std::path::Path;

use yasai::Position;

use crate::eval::kp;

const MAGIC_NAIVE: [u8; 4] = *b"BNN1";
const MAGIC_TWOBAND: [u8; 4] = *b"BNN2"; // Task 046: 2-band, non-shared W1 (kept for comparison)
const MAGIC_SHARED: [u8; 4] = *b"BNN3"; // Task 047: true canonical, shared W1 (single band)
const MAGIC_SHARED_KA: [u8; 4] = *b"BNN4"; // Task 052: HalfKA shared band (kings as features)
const MAGIC_SHARED_KAV2: [u8; 4] = *b"BNN5"; // Task 053: HalfKAv2 (KA minus anchor-king redundancy)
const MAGIC_KPP_ZONE: [u8; 4] = *b"BNN6"; // Task 053: HalfKP shared + king-zone KPP (2 tables)
const MAGIC_QUANT: [u8; 4] = *b"BNNQ"; // Task 058: quantization-aware shared HalfKP (int16 FT / int8 affine)

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// One summed accumulator over all features (`feature_dim` input).
    Naive,
    /// Two accumulators (own/opp) concatenated, split by index range over the
    /// two-band numbering (Task 046; W1 NOT shared between perspectives).
    TwoBand,
    /// Two accumulators (own/opp) over a SINGLE shared band (Task 047, the
    /// original paper's shared W1). Both halves index `[0, VIEW_DIM)`.
    Shared,
    /// Like `Shared` but HalfKA: kings are included as board features (Task 052).
    /// Both halves index `[0, VIEW_DIM_KA)`.
    SharedKA,
    /// HalfKAv2: HalfKA minus each half's redundant anchor king (Task 053). Same
    /// KA table; only the active-feature extraction differs.
    SharedKAV2,
    /// HalfKP shared + king-zone KPP (Task 053): a second embedding table for piece
    /// pairs; four accumulators (KP own/opp, KPP own/opp) -> 4*feature_dim input.
    KppZone,
    /// Quantization-aware shared HalfKP (Task 058): the SAME own/opp feature
    /// extraction as `Shared`, but the forward is YaneuraOu's exact integer NNUE
    /// arithmetic (int16 FT + per-perspective bias -> ClippedReLU[0,127] -> int8
    /// affine -> `>>weight_scale_bits` ClippedReLU -> out/FV_SCALE). `forward()`
    /// returns the integer eval in centipawns (NOT a logit), so it matches the
    /// engine's `e` directly. Weights live in `quant`, not the f32 vectors.
    Quant,
}

/// Integer weights for `Mode::Quant`, matching YaneuraOu's halfkp_*x2-32-32 layout
/// (see `trainer/yane_to_binbo.py::yane_forward_int`, verified bit-exact vs `e`).
struct QuantWeights {
    /// `out / fv_scale` (C truncation toward zero) maps the final accumulator to cp.
    fv_scale: i32,
    /// Right-shift between affine layers (`kWeightScaleBits`, normally 6 => /64).
    weight_scale_bits: u32,
    /// FT embedding, row-major `[total_features][feature_dim]` (int16).
    emb: Vec<i16>,
    /// FT bias `[feature_dim]`, added to each perspective's accumulator BEFORE the
    /// clamp (so it cannot be folded into the first affine like the f32 path does).
    ft_bias: Vec<i16>,
    /// affine0 weight `[hidden_dim][2*feature_dim]` (int8), bias `[hidden_dim]` (int32).
    w0: Vec<i8>,
    b0: Vec<i32>,
    /// affine1 weight `[hidden_dim][hidden_dim]`, bias `[hidden_dim]`.
    w1: Vec<i8>,
    b1: Vec<i32>,
    /// affine2 (output) weight `[1][hidden_dim]`, bias `[1]`.
    w2: Vec<i8>,
    b2: Vec<i32>,
}

pub struct Nnue {
    #[allow(dead_code)] // validated on load; metadata for future use
    pub total_features: usize,
    pub feature_dim: usize,
    pub hidden_dim: usize,
    #[allow(dead_code)] // for future logit -> centipawn conversion
    pub scale_cp: f32,
    /// Feature/accumulator scheme (selected by file magic).
    mode: Mode,
    /// Input width of `network.0` = `feature_dim` (naive) or `2*feature_dim` (2-acc).
    net_in_dim: usize,
    /// Embedding matrix, row-major `[total_features][feature_dim]`.
    emb: Vec<f32>,
    /// KPP-zone pair embedding (KppZone only), row-major `[kpp_total][feature_dim]`.
    kpp_emb: Vec<f32>,
    /// Linear(net_in_dim -> hidden_dim), row-major `[hidden_dim][net_in_dim]` (out, in).
    w0: Vec<f32>,
    b0: Vec<f32>,
    /// Linear(hidden_dim -> hidden_dim), row-major `[hidden_dim][hidden_dim]`.
    w2: Vec<f32>,
    b2: Vec<f32>,
    /// Linear(hidden_dim -> 1), row-major `[1][hidden_dim]`.
    w4: Vec<f32>,
    b4: f32,
    /// Integer weights for `Mode::Quant` (None for the f32 modes BNN1..BNN6).
    quant: Option<QuantWeights>,
}

impl Nnue {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let mut bytes = Vec::new();
        File::open(path.as_ref())
            .map_err(|e| format!("open {}: {e}", path.as_ref().display()))?
            .read_to_end(&mut bytes)
            .map_err(|e| format!("read weights: {e}"))?;

        if bytes.len() >= 4 && bytes[0..4] == MAGIC_QUANT {
            return Self::load_quant(&bytes);
        }
        let mode = if bytes.len() < 4 {
            return Err("weights file too short".to_string());
        } else if bytes[0..4] == MAGIC_KPP_ZONE {
            Mode::KppZone
        } else if bytes[0..4] == MAGIC_SHARED_KAV2 {
            Mode::SharedKAV2
        } else if bytes[0..4] == MAGIC_SHARED_KA {
            Mode::SharedKA
        } else if bytes[0..4] == MAGIC_SHARED {
            Mode::Shared
        } else if bytes[0..4] == MAGIC_TWOBAND {
            Mode::TwoBand
        } else if bytes[0..4] == MAGIC_NAIVE {
            Mode::Naive
        } else {
            return Err("bad magic (expected BNN1..BNN6)".to_string());
        };
        if bytes.len() < 20 {
            return Err("weights file too short".to_string());
        }
        let total_features = read_u32(&bytes, 4) as usize;
        let feature_dim = read_u32(&bytes, 8) as usize;
        let hidden_dim = read_u32(&bytes, 12) as usize;
        let scale_cp = read_f32(&bytes, 16);
        // net_in: naive sums one accumulator -> feature_dim; 2-acc modes -> 2*fd;
        // KppZone has four accumulators (KP own/opp + KPP own/opp) -> 4*fd.
        let net_in_dim = match mode {
            Mode::Naive => feature_dim,
            Mode::KppZone => 4 * feature_dim,
            _ => 2 * feature_dim,
        };

        // KppZone (BNN6) has an extra u32 header field (kpp_total) before the tensors,
        // and a second embedding table (kpp_emb) after the KP table.
        let (mut cursor, kpp_total) = if mode == Mode::KppZone {
            if bytes.len() < 24 {
                return Err("weights file too short (BNN6 header)".to_string());
            }
            (24usize, read_u32(&bytes, 20) as usize)
        } else {
            (20usize, 0usize)
        };
        let emb = read_f32s(&bytes, &mut cursor, total_features * feature_dim)?;
        let kpp_emb = if mode == Mode::KppZone {
            read_f32s(&bytes, &mut cursor, kpp_total * feature_dim)?
        } else {
            Vec::new()
        };
        let w0 = read_f32s(&bytes, &mut cursor, hidden_dim * net_in_dim)?;
        let b0 = read_f32s(&bytes, &mut cursor, hidden_dim)?;
        let w2 = read_f32s(&bytes, &mut cursor, hidden_dim * hidden_dim)?;
        let b2 = read_f32s(&bytes, &mut cursor, hidden_dim)?;
        let w4 = read_f32s(&bytes, &mut cursor, hidden_dim)?;
        let b4 = read_f32s(&bytes, &mut cursor, 1)?[0];
        if cursor != bytes.len() {
            return Err(format!("trailing bytes: used {cursor} of {}", bytes.len()));
        }
        if mode == Mode::KppZone && kpp_total != kp::KPP_ZONE_TOTAL {
            return Err(format!(
                "kpp_total {kpp_total} != expected {}", kp::KPP_ZONE_TOTAL
            ));
        }

        let expected_total = match mode {
            Mode::SharedKA | Mode::SharedKAV2 => kp::CANONICAL_SHARED_KA_TOTAL,
            Mode::Shared | Mode::KppZone => kp::CANONICAL_SHARED_TOTAL,
            _ => kp::TOTAL_FEATURES,
        };
        if total_features != expected_total {
            return Err(format!(
                "total_features {total_features} != expected {expected_total} for this mode"
            ));
        }

        Ok(Self {
            total_features,
            feature_dim,
            hidden_dim,
            scale_cp,
            mode,
            net_in_dim,
            emb,
            kpp_emb,
            w0,
            b0,
            w2,
            b2,
            w4,
            b4,
            quant: None,
        })
    }

    /// Load a `Mode::Quant` (BNNQ) net: header
    /// `magic|total|feature_dim|hidden_dim|fv_scale(i32)|weight_scale_bits(u32)`
    /// then int tensors `emb(i16) ft_bias(i16) w0(i8) b0(i32) w1(i8) b1(i32) w2(i8) b2(i32)`.
    fn load_quant(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 24 {
            return Err("weights file too short (BNNQ header)".to_string());
        }
        let total_features = read_u32(bytes, 4) as usize;
        let feature_dim = read_u32(bytes, 8) as usize;
        let hidden_dim = read_u32(bytes, 12) as usize;
        let fv_scale = read_u32(bytes, 16) as i32;
        let weight_scale_bits = read_u32(bytes, 20);
        if total_features != kp::CANONICAL_SHARED_TOTAL {
            return Err(format!(
                "BNNQ total_features {total_features} != expected {} (shared HalfKP)",
                kp::CANONICAL_SHARED_TOTAL
            ));
        }
        if fv_scale == 0 {
            return Err("BNNQ fv_scale must be non-zero".to_string());
        }
        let net_in_dim = 2 * feature_dim;
        let mut cur = 24usize;
        let emb = read_i16s(bytes, &mut cur, total_features * feature_dim)?;
        let ft_bias = read_i16s(bytes, &mut cur, feature_dim)?;
        let w0 = read_i8s(bytes, &mut cur, hidden_dim * net_in_dim)?;
        let b0 = read_i32s(bytes, &mut cur, hidden_dim)?;
        let w1 = read_i8s(bytes, &mut cur, hidden_dim * hidden_dim)?;
        let b1 = read_i32s(bytes, &mut cur, hidden_dim)?;
        let w2 = read_i8s(bytes, &mut cur, hidden_dim)?;
        let b2 = read_i32s(bytes, &mut cur, 1)?;
        if cur != bytes.len() {
            return Err(format!("BNNQ trailing bytes: used {cur} of {}", bytes.len()));
        }
        Ok(Self {
            total_features,
            feature_dim,
            hidden_dim,
            scale_cp: 1.0, // unused: Quant forward returns cp directly, not a logit
            mode: Mode::Quant,
            net_in_dim,
            emb: Vec::new(),
            kpp_emb: Vec::new(),
            w0: Vec::new(),
            b0: Vec::new(),
            w2: Vec::new(),
            b2: Vec::new(),
            w4: Vec::new(),
            b4: 0.0,
            quant: Some(QuantWeights {
                fv_scale,
                weight_scale_bits,
                emb,
                ft_bias,
                w0,
                b0,
                w1,
                b1,
                w2,
                b2,
            }),
        })
    }

    /// Sum the rows of `emb` for `indices` into `acc` (EmbeddingBag mode=sum).
    fn accumulate_into(&self, indices: &[u32], acc: &mut [f32]) {
        self.accumulate_emb(&self.emb, indices, acc);
    }

    /// Sum the rows of an explicit embedding table for `indices` into `acc`.
    fn accumulate_emb(&self, emb: &[f32], indices: &[u32], acc: &mut [f32]) {
        for &index in indices {
            let base = index as usize * self.feature_dim;
            let row = &emb[base..base + self.feature_dim];
            for (a, &w) in acc.iter_mut().zip(row) {
                *a += w;
            }
        }
    }

    /// Concatenate two accumulators (own, opp) into the `2*feature_dim` input.
    fn concat_input(&self, own: &[u32], opp: &[u32]) -> Vec<f32> {
        let mut acc = vec![0.0f32; self.net_in_dim];
        let (own_acc, opp_acc) = acc.split_at_mut(self.feature_dim);
        self.accumulate_into(own, own_acc);
        self.accumulate_into(opp, opp_acc);
        acc
    }

    /// Linear -> ReLU -> Linear -> ReLU -> Linear(1), returning the logit.
    fn network_forward(&self, acc: &[f32]) -> f32 {
        let h0 = relu_linear(&self.w0, &self.b0, acc, self.hidden_dim, self.net_in_dim);
        let h2 = relu_linear(&self.w2, &self.b2, &h0, self.hidden_dim, self.hidden_dim);
        // Final layer: Linear(hidden_dim -> 1), no ReLU.
        let mut out = self.b4;
        for (j, &x) in h2.iter().enumerate() {
            out += self.w4[j] * x;
        }
        out
    }

    /// Forward pass from a position. For the f32 modes this is a logit (sigmoid is
    /// the win rate); for `Mode::Quant` it is the integer eval in centipawns (== the
    /// engine's `e`), returned as f32 so the `nnue-eval` CLI is unchanged.
    pub fn forward(&self, pos: &Position) -> f32 {
        if self.mode == Mode::Quant {
            let (own, opp) = kp::kp_indices_canonical_shared(pos);
            return self.quant_forward(&own, &opp);
        }
        let acc = match self.mode {
            Mode::Naive => {
                let mut a = vec![0.0f32; self.net_in_dim];
                self.accumulate_into(&kp::kp_indices(pos), &mut a);
                a
            }
            Mode::TwoBand => {
                // own = view-0 body + side; opp = view-1 body, split by index range.
                let indices = kp::kp_indices_canonical(pos);
                let view_dim = kp::VIEW_DIM as u32;
                let side_base = kp::SIDE_FEATURE_BASE as u32;
                let (own, opp): (Vec<u32>, Vec<u32>) = indices
                    .iter()
                    .partition(|&&i| i < view_dim || i >= side_base);
                self.concat_input(&own, &opp)
            }
            Mode::Shared => {
                let (own, opp) = kp::kp_indices_canonical_shared(pos);
                self.concat_input(&own, &opp)
            }
            Mode::SharedKA => {
                let (own, opp) = kp::kp_indices_canonical_shared_ka(pos);
                self.concat_input(&own, &opp)
            }
            Mode::SharedKAV2 => {
                let (own, opp) = kp::kp_indices_canonical_shared_kav2(pos);
                self.concat_input(&own, &opp)
            }
            Mode::KppZone => {
                // Four accumulators: KP own/opp (shared band) + KPP own/opp (pair
                // table), concatenated as [kp_own | kp_opp | pp_own | pp_opp].
                let (kp_own, kp_opp) = kp::kp_indices_canonical_shared(pos);
                let (pp_own, pp_opp) = kp::kpp_zone_halves(pos);
                let fd = self.feature_dim;
                let mut acc = vec![0.0f32; self.net_in_dim]; // 4*fd
                self.accumulate_emb(&self.emb, &kp_own, &mut acc[0..fd]);
                self.accumulate_emb(&self.emb, &kp_opp, &mut acc[fd..2 * fd]);
                self.accumulate_emb(&self.kpp_emb, &pp_own, &mut acc[2 * fd..3 * fd]);
                self.accumulate_emb(&self.kpp_emb, &pp_opp, &mut acc[3 * fd..4 * fd]);
                acc
            }
            Mode::Quant => unreachable!("Mode::Quant returns before this match"),
        };
        self.network_forward(&acc)
    }

    /// YaneuraOu's exact integer NNUE forward (Task 058). `own`/`opp` are the shared
    /// HalfKP active features (same extraction as `Mode::Shared`). Returns the eval
    /// in centipawns as f32. Mirrors `trainer/yane_to_binbo.py::yane_forward_int`:
    /// per-perspective `ft_bias + sum(emb)` clamped to [0,127]; each affine is
    /// `bias + W*x` then `ClippedReLU((>>weight_scale_bits), 0, 127)` (output layer
    /// raw); final `out / fv_scale` truncates toward zero (C division).
    fn quant_forward(&self, own: &[u32], opp: &[u32]) -> f32 {
        let q = self
            .quant
            .as_ref()
            .expect("quant_forward requires Mode::Quant weights");
        let fd = self.feature_dim;
        let net_in = self.net_in_dim; // 2*fd
        let h = self.hidden_dim;
        let shift = q.weight_scale_bits;

        // FeatureTransformer: own/opp accumulators, seeded with ft_bias, clamped.
        let mut x = vec![0i32; net_in];
        for (half, feats) in [(0usize, own), (fd, opp)] {
            for k in 0..fd {
                x[half + k] = q.ft_bias[k] as i32;
            }
            for &f in feats {
                let base = f as usize * fd;
                let row = &q.emb[base..base + fd];
                for k in 0..fd {
                    x[half + k] += row[k] as i32;
                }
            }
            for k in 0..fd {
                x[half + k] = x[half + k].clamp(0, 127);
            }
        }

        // affine0: [hidden][net_in] -> ClippedReLU(>>shift).
        let mut h0 = vec![0i32; h];
        for i in 0..h {
            let mut s = q.b0[i];
            let row = &q.w0[i * net_in..i * net_in + net_in];
            for j in 0..net_in {
                s += row[j] as i32 * x[j];
            }
            h0[i] = (s >> shift).clamp(0, 127);
        }
        // affine1: [hidden][hidden] -> ClippedReLU(>>shift).
        let mut h1 = vec![0i32; h];
        for i in 0..h {
            let mut s = q.b1[i];
            let row = &q.w1[i * h..i * h + h];
            for j in 0..h {
                s += row[j] as i32 * h0[j];
            }
            h1[i] = (s >> shift).clamp(0, 127);
        }
        // affine2 (output): [1][hidden], raw; then out / fv_scale (trunc toward 0).
        let mut o2 = q.b2[0];
        for j in 0..h {
            o2 += q.w2[j] as i32 * h1[j];
        }
        (o2 / q.fv_scale) as f32
    }

    #[allow(dead_code)] // used by the search/eval stage next
    pub fn predict_winrate(&self, pos: &Position) -> f32 {
        1.0 / (1.0 + (-self.forward(pos)).exp())
    }

    /// True for `Mode::Quant`, whose `forward()` returns centipawns directly (not a
    /// logit). The search uses this to skip the logit->cp `EVAL_SCALE` multiply, so a
    /// BNNQ eval drives the search at the same scale YaneuraOu uses (Task 060).
    pub fn is_quant(&self) -> bool {
        self.mode == Mode::Quant
    }
}

/// `y = relu(W x + b)` where `w` is row-major `[out][in]` and matches PyTorch's
/// Linear weight layout `(out_features, in_features)` exactly (no transpose).
fn relu_linear(w: &[f32], b: &[f32], x: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; out_dim];
    for i in 0..out_dim {
        let mut sum = b[i];
        let row = &w[i * in_dim..i * in_dim + in_dim];
        for (j, &xj) in x.iter().enumerate() {
            sum += row[j] * xj;
        }
        y[i] = if sum > 0.0 { sum } else { 0.0 };
    }
    y
}

fn read_u32(bytes: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

fn read_f32(bytes: &[u8], at: usize) -> f32 {
    f32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
}

fn read_f32s(bytes: &[u8], cursor: &mut usize, count: usize) -> Result<Vec<f32>, String> {
    let end = *cursor + count * 4;
    if end > bytes.len() {
        return Err(format!(
            "truncated weights: need {end} bytes, have {}",
            bytes.len()
        ));
    }
    let mut out = Vec::with_capacity(count);
    let mut at = *cursor;
    for _ in 0..count {
        out.push(read_f32(bytes, at));
        at += 4;
    }
    *cursor = end;
    Ok(out)
}

fn read_i16s(bytes: &[u8], cursor: &mut usize, count: usize) -> Result<Vec<i16>, String> {
    let end = *cursor + count * 2;
    if end > bytes.len() {
        return Err(format!("truncated i16: need {end} bytes, have {}", bytes.len()));
    }
    let mut out = Vec::with_capacity(count);
    let mut at = *cursor;
    for _ in 0..count {
        out.push(i16::from_le_bytes([bytes[at], bytes[at + 1]]));
        at += 2;
    }
    *cursor = end;
    Ok(out)
}

fn read_i8s(bytes: &[u8], cursor: &mut usize, count: usize) -> Result<Vec<i8>, String> {
    let end = *cursor + count;
    if end > bytes.len() {
        return Err(format!("truncated i8: need {end} bytes, have {}", bytes.len()));
    }
    let out = bytes[*cursor..end].iter().map(|&b| b as i8).collect();
    *cursor = end;
    Ok(out)
}

fn read_i32s(bytes: &[u8], cursor: &mut usize, count: usize) -> Result<Vec<i32>, String> {
    let end = *cursor + count * 4;
    if end > bytes.len() {
        return Err(format!("truncated i32: need {end} bytes, have {}", bytes.len()));
    }
    let mut out = Vec::with_capacity(count);
    let mut at = *cursor;
    for _ in 0..count {
        out.push(i32::from_le_bytes([
            bytes[at],
            bytes[at + 1],
            bytes[at + 2],
            bytes[at + 3],
        ]));
        at += 4;
    }
    *cursor = end;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny() -> Nnue {
        // feature_dim=2, hidden_dim=2, total_features=3, naive (net_in_dim=2).
        Nnue {
            total_features: 3,
            feature_dim: 2,
            hidden_dim: 2,
            scale_cp: 600.0,
            mode: Mode::Naive,
            net_in_dim: 2,
            // rows: f0=[1,2] f1=[3,4] f2=[5,6]
            emb: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            kpp_emb: Vec::new(),
            // [[1,0],[0,1]] identity
            w0: vec![1.0, 0.0, 0.0, 1.0],
            b0: vec![0.0, 0.0],
            // [[1,1],[-1,0]]
            w2: vec![1.0, 1.0, -1.0, 0.0],
            b2: vec![0.0, 1.0],
            // [2,3]
            w4: vec![2.0, 3.0],
            b4: 1.0,
            quant: None,
        }
    }

    #[test]
    fn accumulate_sums_active_rows() {
        let net = tiny();
        // active {0,2} -> [1+5, 2+6] = [6,8]
        let mut acc = vec![0.0f32; 2];
        net.accumulate_into(&[0, 2], &mut acc);
        assert_eq!(acc, vec![6.0, 8.0]);
    }

    #[test]
    fn network_forward_matches_hand_computation() {
        let net = tiny();
        // acc=[6,8]; h0=identity=[6,8]; h2=relu([6+8, -6+1])=relu([14,-5])=[14,0];
        // out = 1 + 2*14 + 3*0 = 29
        assert_eq!(net.network_forward(&[6.0, 8.0]), 29.0);
    }

    #[test]
    fn concat_input_sums_own_and_opp_separately() {
        // feature_dim=2, net_in_dim=4 (concat). own/opp accumulators are separate.
        let mut emb = vec![0.0f32; 6 * 2];
        emb[5 * 2] = 1.0; emb[5 * 2 + 1] = 1.0; // own index 5
        emb[2 * 2] = 100.0; emb[2 * 2 + 1] = 100.0; // own index 2 (e.g. side)
        emb[3 * 2] = 10.0; emb[3 * 2 + 1] = 10.0; // opp index 3
        let net = Nnue {
            total_features: 6, feature_dim: 2, hidden_dim: 2, scale_cp: 600.0,
            mode: Mode::Shared, net_in_dim: 4,
            emb,
            kpp_emb: Vec::new(),
            w0: vec![0.0; 8], b0: vec![0.0, 0.0],
            w2: vec![0.0; 4], b2: vec![0.0, 0.0],
            w4: vec![0.0, 0.0], b4: 0.0,
            quant: None,
        };
        // own = {5, 2} -> [1+100, 1+100]; opp = {3} -> [10,10].
        assert_eq!(net.concat_input(&[5, 2], &[3]), vec![101.0, 101.0, 10.0, 10.0]);
    }

    /// Hand-computed integer forward (Task 058): feature_dim=2, hidden_dim=2,
    /// fv_scale=2, shift=1. Exercises ft_bias-before-clamp, the >>shift ClippedReLU,
    /// and truncating final division.
    fn tiny_quant() -> Nnue {
        Nnue {
            total_features: 3,
            feature_dim: 2,
            hidden_dim: 2,
            scale_cp: 1.0,
            mode: Mode::Quant,
            net_in_dim: 4,
            emb: Vec::new(),
            kpp_emb: Vec::new(),
            w0: Vec::new(),
            b0: Vec::new(),
            w2: Vec::new(),
            b2: Vec::new(),
            w4: Vec::new(),
            b4: 0.0,
            quant: Some(QuantWeights {
                fv_scale: 2,
                weight_scale_bits: 1,
                // rows: f0=[10,20] f1=[3,4] f2=[200,0]
                emb: vec![10, 20, 3, 4, 200, 0],
                ft_bias: vec![1, 2],
                // affine0 [2][4] identity-ish: out0 = x0+x1, out1 = x2+x3
                w0: vec![1, 1, 0, 0, 0, 0, 1, 1],
                b0: vec![0, 0],
                // affine1 [2][2] identity
                w1: vec![1, 0, 0, 1],
                b1: vec![0, 0],
                // output [1][2] = h0 + h1
                w2: vec![1, 1],
                b2: vec![0],
            }),
        }
    }

    #[test]
    fn quant_forward_matches_hand_computation() {
        let net = tiny_quant();
        // own={0,1}: acc=[1+10+3, 2+20+4]=[14,26] clamp->[14,26]
        // opp={2}:   acc=[1+200, 2+0]=[201,2] clamp->[127,2]
        // x=[14,26,127,2]
        // o0_0 = (14+26)>>1 = 20 ; o0_1 = (127+2)>>1 = 64 ; h0=[20,64]
        // o1=h0 (identity) >>1 -> [10,32]
        // o2 = 10+32 = 42 ; eval = 42/2 = 21
        assert_eq!(net.quant_forward(&[0, 1], &[2]), 21.0);
    }
}
