use crate::biquad_response::biquad_response;
use crate::compensate::{center, compensate};
use crate::equalize::equalize;
use crate::interpolate::{build_grid, interpolate};
use crate::peak_finding::find_peaks_full;
use crate::types::{
    BiquadError, Constraints, Filter, FilterSpec, FilterType, FreqPoint, InterpolateOptions,
    MinStd, OptimizeResult,
};
use slsqp::minimize;
use std::cell::{Cell, RefCell};

// ────────────────────────────────────────────────────────────────────────────
// Resolved filter spec (internal; fully expanded from FilterSpec)
// ────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
struct ResolvedSpec {
    filter_type: FilterType,
    fc_init: Option<f64>,
    q_init: Option<f64>,
    gain_init: Option<f64>,
    optimize_fc: bool,
    optimize_q: bool,
    optimize_gain: bool,
    fc_range: (f64, f64),
    q_range: (f64, f64),
    gain_range: (f64, f64),
}

fn default_q_range(ft: FilterType) -> (f64, f64) {
    match ft {
        FilterType::PK => (0.18248, 6.0),
        FilterType::LSQ | FilterType::HSQ => (0.4, 0.7),
    }
}

fn resolve_specs(specs: &[FilterSpec]) -> Result<Vec<ResolvedSpec>, BiquadError> {
    specs
        .iter()
        .map(|s| {
            let ft = s.filter_type.unwrap_or(FilterType::PK);
            let opt_fc = s.optimize_fc.unwrap_or(true);
            let opt_q = s.optimize_q.unwrap_or(true);
            let opt_gain = s.optimize_gain.unwrap_or(true);

            if s.fc.is_none() && !opt_fc {
                return Err(BiquadError::InvalidFilterSpec(
                    "fc=None with optimize_fc=false: cannot lock an unspecified fc".into(),
                ));
            }
            if s.q.is_none() && !opt_q {
                return Err(BiquadError::InvalidFilterSpec(
                    "q=None with optimize_q=false: cannot lock an unspecified q".into(),
                ));
            }
            if s.gain.is_none() && !opt_gain {
                return Err(BiquadError::InvalidFilterSpec(
                    "gain=None with optimize_gain=false: cannot lock an unspecified gain".into(),
                ));
            }

            Ok(ResolvedSpec {
                filter_type: ft,
                fc_init: s.fc,
                q_init: s.q,
                gain_init: s.gain,
                optimize_fc: opt_fc,
                optimize_q: opt_q,
                optimize_gain: opt_gain,
                fc_range: s.fc_range.unwrap_or((20.0, 10000.0)),
                q_range: s.q_range.unwrap_or_else(|| default_q_range(ft)),
                gain_range: s.gain_range,
            })
        })
        .collect()
}

// ────────────────────────────────────────────────────────────────────────────
// Parameter encoding / decoding / bounds  (peq.py:567-583, 642-655)
// Layout per filter: [log10(fc)?, q?, gain?] — locked params skipped.
// ────────────────────────────────────────────────────────────────────────────

fn encode_params(filters: &[Filter], specs: &[ResolvedSpec]) -> Vec<f64> {
    let mut v = Vec::new();
    for (f, s) in filters.iter().zip(specs) {
        if s.optimize_fc {
            v.push(f.fc.log10());
        }
        if s.optimize_q {
            v.push(f.q);
        }
        if s.optimize_gain {
            v.push(f.gain);
        }
    }
    v
}

fn decode_params(x: &[f64], filters: &mut [Filter], specs: &[ResolvedSpec]) {
    let mut i = 0;
    for (f, s) in filters.iter_mut().zip(specs) {
        if s.optimize_fc {
            f.fc = 10_f64.powf(x[i]);
            i += 1;
        }
        if s.optimize_q {
            f.q = x[i];
            i += 1;
        }
        if s.optimize_gain {
            f.gain = x[i];
            i += 1;
        }
    }
}

fn build_bounds(filters: &[Filter], specs: &[ResolvedSpec]) -> Vec<(f64, f64)> {
    let mut bounds = Vec::new();
    for (_, s) in filters.iter().zip(specs) {
        if s.optimize_fc {
            bounds.push((s.fc_range.0.log10(), s.fc_range.1.log10()));
        }
        if s.optimize_q {
            bounds.push(s.q_range);
        }
        if s.optimize_gain {
            bounds.push(s.gain_range);
        }
    }
    bounds
}

// ────────────────────────────────────────────────────────────────────────────
// Filter cascade response (dB, summed)
// ────────────────────────────────────────────────────────────────────────────

pub fn total_response(filters: &[Filter], freqs: &[f64], fs: f64) -> Vec<f64> {
    let mut out = vec![0.0f64; freqs.len()];
    for f in filters {
        let fr = biquad_response(f.filter_type, f.fc, f.gain, f.q, freqs, fs);
        for (o, v) in out.iter_mut().zip(fr) {
            *o += v;
        }
    }
    out
}

// ────────────────────────────────────────────────────────────────────────────
// Sharpness penalty  (peq.py:252-266 — PK only; shelves return 0)
// ────────────────────────────────────────────────────────────────────────────

fn sharpness_penalty(filter: &Filter, freqs: &[f64], fs: f64) -> f64 {
    if filter.filter_type != FilterType::PK {
        return 0.0;
    }
    if filter.q.abs() < 1e-12 {
        return 0.0;
    }
    let gain_limit = -0.09503189270199464 + 20.575128011847003 / filter.q;
    if gain_limit.abs() < 1e-12 {
        return 0.0;
    }
    let x = filter.gain / gain_limit - 1.0;
    // Numerically stable sigmoid: 1/(1+exp(-100x))
    let z = -100.0 * x;
    let coeff = if z >= 0.0 {
        let e = z.min(500.0).exp();
        1.0 / (1.0 + e)
    } else {
        let e = (-z).min(500.0).exp();
        e / (1.0 + e)
    };
    let fr = biquad_response(filter.filter_type, filter.fc, filter.gain, filter.q, freqs, fs);
    fr.iter().map(|&v| (v * coeff).powi(2)).sum::<f64>() / fr.len() as f64
}

// ────────────────────────────────────────────────────────────────────────────
// Loss function  (peq.py:585-600)
// ────────────────────────────────────────────────────────────────────────────

fn mean_slice(s: &[f64]) -> f64 {
    if s.is_empty() {
        return 0.0;
    }
    s.iter().sum::<f64>() / s.len() as f64
}

fn joint_loss(
    filters: &[Filter],
    freqs: &[f64],
    correction: &[f64],
    fs: f64,
    min_f_ix: usize,
    max_f_ix: usize,
    ten_k_ix: usize,
) -> f64 {
    let mut target = correction.to_vec();
    let mut fr = total_response(filters, freqs, fs);

    // Flatten above 10 kHz (inclusive)  [ten_k_ix:]
    let mt = mean_slice(&target[ten_k_ix..]);
    let mf_val = mean_slice(&fr[ten_k_ix..]);
    for v in target[ten_k_ix..].iter_mut() {
        *v = mt;
    }
    for v in fr[ten_k_ix..].iter_mut() {
        *v = mf_val;
    }

    let mse = target[min_f_ix..max_f_ix]
        .iter()
        .zip(&fr[min_f_ix..max_f_ix])
        .map(|(t, r)| (t - r).powi(2))
        .sum::<f64>()
        / (max_f_ix - min_f_ix) as f64;

    let penalty: f64 = filters.iter().map(|f| sharpness_penalty(f, freqs, fs)).sum();
    (mse + penalty).sqrt()
}

fn eval_loss(
    x: &[f64],
    base: &[Filter],
    specs: &[ResolvedSpec],
    freqs: &[f64],
    correction: &[f64],
    fs: f64,
    min_f_ix: usize,
    max_f_ix: usize,
    ten_k_ix: usize,
) -> f64 {
    let mut filters = base.to_vec();
    decode_params(x, &mut filters, specs);
    joint_loss(&filters, freqs, correction, fs, min_f_ix, max_f_ix, ten_k_ix)
}

// ────────────────────────────────────────────────────────────────────────────
// Filter initialization  (peq.py:165-401)
// ────────────────────────────────────────────────────────────────────────────

fn init_peaking(freqs: &[f64], correction: &[f64], spec: &ResolvedSpec, _fs: f64) -> Filter {
    if !spec.optimize_fc && !spec.optimize_q && !spec.optimize_gain {
        return Filter {
            filter_type: FilterType::PK,
            fc: spec.fc_init.unwrap_or(spec.fc_range.0),
            q: spec.q_init.unwrap_or(1.0),
            gain: spec.gain_init.unwrap_or(0.0),
        };
    }

    let min_fc_ix = freqs.iter().take_while(|&&f| f < spec.fc_range.0).count();
    let max_fc_ix = freqs.iter().take_while(|&&f| f < spec.fc_range.1).count().min(freqs.len() - 1);

    let pos: Vec<f64> = correction.iter().map(|&v| v.max(0.0)).collect();
    let neg: Vec<f64> = correction.iter().map(|&v| (-v).max(0.0)).collect();
    let (pos_inds, pos_props) = find_peaks_full(&pos, 0.0, 0.0, 0.0);
    let (neg_inds, neg_props) = find_peaks_full(&neg, 0.0, 0.0, 0.0);

    struct Candidate {
        idx: usize,
        height: f64,
        width: f64,
        positive: bool,
    }

    let mut candidates: Vec<Candidate> = Vec::new();
    for (&ix, (&h, &w)) in pos_inds
        .iter()
        .zip(pos_props.peak_heights.iter().zip(pos_props.widths.iter()))
    {
        if ix >= min_fc_ix && ix <= max_fc_ix {
            candidates.push(Candidate { idx: ix, height: h, width: w, positive: true });
        }
    }
    for (&ix, (&h, &w)) in neg_inds
        .iter()
        .zip(neg_props.peak_heights.iter().zip(neg_props.widths.iter()))
    {
        if ix >= min_fc_ix && ix <= max_fc_ix {
            candidates.push(Candidate { idx: ix, height: h, width: w, positive: false });
        }
    }

    let fc;
    let q;
    let gain;

    if candidates.is_empty() {
        let mid_ix = (min_fc_ix + max_fc_ix) / 2;
        fc = freqs[mid_ix.min(freqs.len() - 1)];
        q = 2.0_f64.sqrt();
        gain = 0.0;
    } else {
        let best = candidates
            .iter()
            .max_by(|a, b| (a.height * a.width).partial_cmp(&(b.height * b.width)).unwrap())
            .unwrap();
        fc = freqs[best.idx];
        let f_step = (freqs[1] / freqs[0]).log2();
        let bw = f_step * best.width;
        let ratio = 2_f64.powf(bw);
        let raw_q = if (ratio - 1.0).abs() < 1e-12 {
            spec.q_range.1
        } else {
            ratio.sqrt() / (ratio - 1.0)
        };
        q = raw_q.clamp(spec.q_range.0, spec.q_range.1);
        gain = if best.positive { best.height } else { -best.height };
    }

    Filter {
        filter_type: FilterType::PK,
        fc: spec.fc_init.unwrap_or(fc),
        q: spec.q_init.unwrap_or(q),
        gain: spec.gain_init.unwrap_or(gain).clamp(spec.gain_range.0, spec.gain_range.1),
    }
}

fn init_low_shelf(freqs: &[f64], correction: &[f64], spec: &ResolvedSpec, fs: f64) -> Filter {
    if !spec.optimize_fc && !spec.optimize_q && !spec.optimize_gain {
        return Filter {
            filter_type: FilterType::LSQ,
            fc: spec.fc_init.unwrap_or(spec.fc_range.0),
            q: spec.q_init.unwrap_or(0.7),
            gain: spec.gain_init.unwrap_or(0.0),
        };
    }

    // Shelf fc search clamp: [max(40, min_fc), min(10000, max_fc)]  (peq.py:334-335)
    let min_ix = freqs.iter().take_while(|&&f| f < spec.fc_range.0.max(40.0)).count();
    let max_ix = freqs.iter().take_while(|&&f| f < spec.fc_range.1.min(10000.0)).count();
    let safe_max = max_ix.max(min_ix + 1).min(freqs.len());

    // argmax of |mean(correction[:ix+1])| for ix in [min_ix, max_ix)
    // Then add min_ix back  (peq.py:389)
    let best_offset = (min_ix..safe_max)
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            let ma = mean_slice(&correction[..*a + 1]).abs();
            let mb = mean_slice(&correction[..*b + 1]).abs();
            ma.partial_cmp(&mb).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(offset, _)| offset)
        .unwrap_or(0);
    let fc_ix = (best_offset + min_ix).min(freqs.len() - 1);
    let fc = freqs[fc_ix];

    let q_raw = 0.7_f64.clamp(spec.q_range.0, spec.q_range.1);
    let shelf_fr = biquad_response(FilterType::LSQ, fc, 1.0, q_raw, freqs, fs);
    let sum_fr: f64 = shelf_fr.iter().sum();
    let gain = if sum_fr.abs() < 1e-12 {
        0.0
    } else {
        let dot: f64 = correction.iter().zip(&shelf_fr).map(|(c, s)| c * s).sum();
        (dot / sum_fr).clamp(spec.gain_range.0, spec.gain_range.1)
    };

    Filter {
        filter_type: FilterType::LSQ,
        fc: spec.fc_init.unwrap_or(fc),
        q: spec.q_init.unwrap_or(q_raw),
        gain: spec.gain_init.unwrap_or(gain),
    }
}

fn init_high_shelf(freqs: &[f64], correction: &[f64], spec: &ResolvedSpec, fs: f64) -> Filter {
    if !spec.optimize_fc && !spec.optimize_q && !spec.optimize_gain {
        return Filter {
            filter_type: FilterType::HSQ,
            fc: spec.fc_init.unwrap_or(spec.fc_range.0),
            q: spec.q_init.unwrap_or(0.7),
            gain: spec.gain_init.unwrap_or(0.0),
        };
    }

    let min_ix = freqs.iter().take_while(|&&f| f < spec.fc_range.0.max(40.0)).count();
    let max_ix = freqs.iter().take_while(|&&f| f < spec.fc_range.1.min(10000.0)).count();
    let safe_max = max_ix.max(min_ix + 1).min(freqs.len());

    // argmax of |mean(correction[ix:])| for ix in [min_ix, max_ix)
    // NOTE: AutoEQ bug — does NOT add min_ix (peq.py:337); argmax result used directly as fc_ix.
    let fc_ix = (min_ix..safe_max)
        .enumerate()
        .max_by(|(_, a), (_, b)| {
            let ma = mean_slice(&correction[*a..]).abs();
            let mb = mean_slice(&correction[*b..]).abs();
            ma.partial_cmp(&mb).unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(offset, _)| offset) // offset from 0, NOT +min_ix — mirrors AutoEQ bug
        .unwrap_or(0)
        .min(freqs.len() - 1);
    let fc = freqs[fc_ix];

    let q_raw = 0.7_f64.clamp(spec.q_range.0, spec.q_range.1);
    let shelf_fr = biquad_response(FilterType::HSQ, fc, 1.0, q_raw, freqs, fs);
    let sum_fr: f64 = shelf_fr.iter().sum();
    let gain = if sum_fr.abs() < 1e-12 {
        0.0
    } else {
        let dot: f64 = correction.iter().zip(&shelf_fr).map(|(c, s)| c * s).sum();
        (dot / sum_fr).clamp(spec.gain_range.0, spec.gain_range.1)
    };

    Filter {
        filter_type: FilterType::HSQ,
        fc: spec.fc_init.unwrap_or(fc),
        q: spec.q_init.unwrap_or(q_raw),
        gain: spec.gain_init.unwrap_or(gain),
    }
}

// 12-entry priority table index (peq.py:608-619). optimize_gain is NOT part of the key.
fn priority_table_index(ft: FilterType, optimize_fc: bool, optimize_q: bool) -> usize {
    match (ft, optimize_fc, optimize_q) {
        (FilterType::PK, true, true) => 0,
        (FilterType::LSQ, true, true) => 1,
        (FilterType::HSQ, true, true) => 2,
        (FilterType::PK, true, false) => 3,
        (FilterType::LSQ, true, false) => 4,
        (FilterType::HSQ, true, false) => 5,
        (FilterType::PK, false, true) => 6,
        (FilterType::LSQ, false, true) => 7,
        (FilterType::HSQ, false, true) => 8,
        (FilterType::PK, false, false) => 9,
        (FilterType::LSQ, false, false) => 10,
        (FilterType::HSQ, false, false) => 11,
    }
}

fn init_priority(spec: &ResolvedSpec) -> f64 {
    let ix = priority_table_index(spec.filter_type, spec.optimize_fc, spec.optimize_q);
    let val = (ix * 100) as f64;
    if spec.optimize_fc && (spec.fc_range.1 / spec.fc_range.0) > 1.0 + 1e-12 {
        val + 1.0 / (spec.fc_range.1 / spec.fc_range.0).log2()
    } else {
        val
    }
}

/// Initialize all filters in priority order with subtractive seeding (peq.py:628-638).
fn init_filters(
    specs: &[ResolvedSpec],
    freqs: &[f64],
    correction: &[f64],
    fs: f64,
) -> Vec<Filter> {
    let n = specs.len();
    let mut filters: Vec<Option<Filter>> = vec![None; n];

    // Descending priority: most-constrained (locked) first
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        init_priority(&specs[b])
            .partial_cmp(&init_priority(&specs[a]))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut remaining = correction.to_vec();
    for ix in order {
        let s = &specs[ix];
        let filt = match s.filter_type {
            FilterType::PK => init_peaking(freqs, &remaining, s, fs),
            FilterType::LSQ => init_low_shelf(freqs, &remaining, s, fs),
            FilterType::HSQ => init_high_shelf(freqs, &remaining, s, fs),
        };
        let fr = biquad_response(filt.filter_type, filt.fc, filt.gain, filt.q, freqs, fs);
        for (r, v) in remaining.iter_mut().zip(fr) {
            *r -= v;
        }
        filters[ix] = Some(filt);
    }

    filters.into_iter().map(|f| f.unwrap()).collect()
}

// ────────────────────────────────────────────────────────────────────────────
// Convergence helpers  (peq.py:657-698)
// ────────────────────────────────────────────────────────────────────────────

fn pop_std(xs: &[f64]) -> f64 {
    let n = xs.len() as f64;
    let mean = xs.iter().sum::<f64>() / n;
    (xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n).sqrt()
}

fn convergence_triggered(losses: &[f64], min_std: f64) -> bool {
    let n = 8;
    let big = losses.len() > n && pop_std(&losses[losses.len() - n..]) < min_std;
    let small = losses.len() > n / 2 && pop_std(&losses[losses.len() - n / 2..]) < min_std / 2.0;
    big || small
}

// ────────────────────────────────────────────────────────────────────────────
// SLSQP driver  (peq.py:700-719)
// ────────────────────────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn joint_optimize(
    initial_filters: &[Filter],
    specs: &[ResolvedSpec],
    freqs: &[f64],
    correction: &[f64],
    fs: f64,
    min_f_ix: usize,
    max_f_ix: usize,
    ten_k_ix: usize,
    stop_threshold: Option<f64>,
    x0_override: Option<Vec<f64>>,
) -> Result<Vec<Filter>, BiquadError> {
    let x0 = x0_override.unwrap_or_else(|| encode_params(initial_filters, specs));
    if x0.is_empty() {
        return Ok(initial_filters.to_vec()); // no free variables
    }
    let bounds = build_bounds(initial_filters, specs);

    let history: RefCell<Vec<(Vec<f64>, f64)>> = RefCell::new(Vec::new());
    let early_stop = Cell::new(false);
    let best_x_on_stop: RefCell<Option<Vec<f64>>> = RefCell::new(None);
    let frozen_loss = Cell::new(f64::INFINITY);

    let objective = |x: &[f64], grad: Option<&mut [f64]>, _: &mut ()| -> f64 {
        if early_stop.get() {
            if let Some(g) = grad {
                g.fill(0.0);
            }
            return frozen_loss.get();
        }

        let f = eval_loss(
            x, initial_filters, specs, freqs, correction, fs, min_f_ix, max_f_ix, ten_k_ix,
        );

        if let Some(g) = grad {
            let eps = f64::EPSILON.sqrt();
            let mut x_pert = x.to_vec();
            for i in 0..x.len() {
                let h = eps * (1.0 + x[i].abs());
                x_pert[i] = x[i] + h;
                let f1 = eval_loss(
                    &x_pert, initial_filters, specs, freqs, correction, fs, min_f_ix, max_f_ix,
                    ten_k_ix,
                );
                g[i] = (f1 - f) / h;
                x_pert[i] = x[i];
            }
        }

        {
            let mut h = history.borrow_mut();
            h.push((x.to_vec(), f));

            if let Some(min_std_val) = stop_threshold {
                let losses: Vec<f64> = h.iter().map(|(_, l)| *l).collect();
                if convergence_triggered(&losses, min_std_val) {
                    let best_idx = h
                        .iter()
                        .enumerate()
                        .min_by(|a, b| a.1.1.partial_cmp(&b.1.1).unwrap())
                        .map(|(i, _)| i)
                        .unwrap();
                    let best_x = h[best_idx].0.clone();
                    let best_f = h[best_idx].1;
                    drop(h); // release borrow before writing to other cells
                    *best_x_on_stop.borrow_mut() = Some(best_x);
                    frozen_loss.set(best_f);
                    early_stop.set(true);
                }
            }
        }

        f
    };

    let no_cons: Vec<fn(&[f64], Option<&mut [f64]>, &mut ()) -> f64> = Vec::new();
    let result = minimize(objective, &x0, &bounds, &no_cons, (), 150, None);

    let final_x = if early_stop.get() {
        best_x_on_stop.borrow().clone().unwrap()
    } else {
        match result {
            Ok((_, x, _)) => x,
            Err((_, x, _)) => x,
        }
    };

    let mut filters_out = initial_filters.to_vec();
    decode_params(&final_x, &mut filters_out, specs);
    Ok(filters_out)
}

// ────────────────────────────────────────────────────────────────────────────
// Pregain  (frequency_response.py:129, constants.py PREAMP_HEADROOM=0.2)
// ────────────────────────────────────────────────────────────────────────────

fn compute_pregain(filters: &[Filter], freqs: &[f64], fs: f64) -> f64 {
    let resp = total_response(filters, freqs, fs);
    let max_boost = resp.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max_boost > 0.0 {
        -(max_boost + 0.2)
    } else {
        0.0
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Grid index helpers
// ────────────────────────────────────────────────────────────────────────────

fn ten_k_ix(freqs: &[f64]) -> usize {
    freqs
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| {
            (*a - 10000.0).abs().partial_cmp(&(*b - 10000.0).abs()).unwrap()
        })
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn loss_min_f_ix(freqs: &[f64]) -> usize {
    freqs.iter().take_while(|&&f| f < 20.0).count()
}

fn loss_max_f_ix(freqs: &[f64]) -> usize {
    // Half-open: count of freqs strictly < 20000 (mirrors np.sum(f < 20000))
    freqs.iter().take_while(|&&f| f < 20000.0).count()
}

// ────────────────────────────────────────────────────────────────────────────
// Public entry point
// ────────────────────────────────────────────────────────────────────────────

pub fn optimize(
    measured: &[FreqPoint],
    target: &[FreqPoint],
    constraints: &Constraints,
) -> Result<OptimizeResult, BiquadError> {
    if constraints.filter_specs.is_empty() {
        return Ok(OptimizeResult { pregain: 0.0, filters: vec![] });
    }

    let fs = constraints.fs.unwrap_or(44100.0);
    let stop_threshold = match constraints.min_std.as_ref().unwrap_or(&MinStd::Default) {
        MinStd::Default => Some(0.002),
        MinStd::Custom(v) => Some(*v),
        MinStd::Disabled => None,
    };

    // 1. Interpolate measured to 1.01 grid
    let measured_i = interpolate(
        measured,
        &InterpolateOptions { step: Some(1.01), f_min: Some(20.0), f_max: Some(20000.0) },
    );

    // 2. Center measured at 1 kHz
    let measured_c = center(&measured_i);

    // 3. Compensate (four-step pipeline)
    let error = compensate(&measured_c, target);

    // 4. Equalize (slope-limited correction curve)
    let eq_curve = equalize(&error);

    // 5. Re-interpolate to optimizer grid (1.02 step)
    let eq_on_opt = interpolate(
        &eq_curve,
        &InterpolateOptions { step: Some(1.02), f_min: Some(20.0), f_max: Some(20000.0) },
    );
    let correction: Vec<f64> = eq_on_opt.iter().map(|p| p.db).collect();
    let opt_freqs = build_grid(20.0, 20000.0, 1.02);

    // 6. Resolve specs
    let specs = resolve_specs(&constraints.filter_specs)?;

    // 7. Pre-compute loss indices
    let mf = loss_min_f_ix(&opt_freqs);
    let xf = loss_max_f_ix(&opt_freqs);
    let tk = ten_k_ix(&opt_freqs);

    // 8. Initialize filters
    let initial = init_filters(&specs, &opt_freqs, &correction, fs);

    // 9. Optimize
    let filters = joint_optimize(
        &initial, &specs, &opt_freqs, &correction, fs, mf, xf, tk, stop_threshold, None,
    )?;

    // 10. Pregain
    let pregain = compute_pregain(&filters, &opt_freqs, fs);

    Ok(OptimizeResult { pregain, filters })
}

/// Like `optimize` but uses a caller-supplied x0 instead of running init.
/// x0 must be encoded identically to `encode_params`: log10(fc), q, gain per free param.
/// Exposed for diagnostic tests that want to feed AutoEQ's x0 directly.
pub fn optimize_from_x0(
    measured: &[FreqPoint],
    target: &[FreqPoint],
    constraints: &Constraints,
    x0: Vec<f64>,
) -> Result<OptimizeResult, BiquadError> {
    let fs = constraints.fs.unwrap_or(44100.0);
    let stop_threshold = match constraints.min_std.as_ref().unwrap_or(&MinStd::Default) {
        MinStd::Default => Some(0.002),
        MinStd::Custom(v) => Some(*v),
        MinStd::Disabled => None,
    };
    let measured_i = interpolate(measured, &InterpolateOptions { step: Some(1.01), f_min: Some(20.0), f_max: Some(20000.0) });
    let measured_c = center(&measured_i);
    let error = compensate(&measured_c, target);
    let eq_curve = equalize(&error);
    let eq_on_opt = interpolate(&eq_curve, &InterpolateOptions { step: Some(1.02), f_min: Some(20.0), f_max: Some(20000.0) });
    let correction: Vec<f64> = eq_on_opt.iter().map(|p| p.db).collect();
    let opt_freqs = build_grid(20.0, 20000.0, 1.02);
    let specs = resolve_specs(&constraints.filter_specs)?;
    let mf = loss_min_f_ix(&opt_freqs);
    let xf = loss_max_f_ix(&opt_freqs);
    let tk = ten_k_ix(&opt_freqs);
    // Use zero-gain placeholders so decode_params has a filter list to splice locked params into.
    let initial = init_filters(&specs, &opt_freqs, &correction, fs);
    let filters = joint_optimize(&initial, &specs, &opt_freqs, &correction, fs, mf, xf, tk, stop_threshold, Some(x0))?;
    let pregain = compute_pregain(&filters, &opt_freqs, fs);
    Ok(OptimizeResult { pregain, filters })
}

/// Returns the encoded x0 that `optimize` would feed to SLSQP (log10(fc), q, gain per free param).
pub fn compute_x0(
    measured: &[FreqPoint],
    target: &[FreqPoint],
    constraints: &Constraints,
) -> Result<(Vec<f64>, Vec<Filter>), BiquadError> {
    let fs = constraints.fs.unwrap_or(44100.0);
    let measured_i = interpolate(measured, &InterpolateOptions { step: Some(1.01), f_min: Some(20.0), f_max: Some(20000.0) });
    let measured_c = center(&measured_i);
    let error = compensate(&measured_c, target);
    let eq_curve = equalize(&error);
    let eq_on_opt = interpolate(&eq_curve, &InterpolateOptions { step: Some(1.02), f_min: Some(20.0), f_max: Some(20000.0) });
    let correction: Vec<f64> = eq_on_opt.iter().map(|p| p.db).collect();
    let opt_freqs = build_grid(20.0, 20000.0, 1.02);
    let specs = resolve_specs(&constraints.filter_specs)?;
    let initial = init_filters(&specs, &opt_freqs, &correction, fs);
    let x0 = encode_params(&initial, &specs);
    Ok((x0, initial))
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn pk(fc: f64, gain: f64, q: f64) -> Filter {
        Filter { filter_type: FilterType::PK, fc, gain, q }
    }
    fn lsq(fc: f64, gain: f64, q: f64) -> Filter {
        Filter { filter_type: FilterType::LSQ, fc, gain, q }
    }
    fn hsq(fc: f64, gain: f64, q: f64) -> Filter {
        Filter { filter_type: FilterType::HSQ, fc, gain, q }
    }

    fn free_spec(ft: FilterType) -> FilterSpec {
        FilterSpec {
            filter_type: Some(ft),
            fc: None,
            q: None,
            gain: None,
            optimize_fc: None,
            optimize_q: None,
            optimize_gain: None,
            fc_range: None,
            q_range: None,
            gain_range: (-12.0, 12.0),
        }
    }

    #[test]
    fn test_resolve_specs_defaults_pk() {
        let specs = resolve_specs(&[free_spec(FilterType::PK)]).unwrap();
        assert!(specs[0].optimize_fc && specs[0].optimize_q && specs[0].optimize_gain);
        assert_abs_diff_eq!(specs[0].q_range.0, 0.18248, epsilon = 1e-12);
        assert_abs_diff_eq!(specs[0].fc_range.0, 20.0, epsilon = 1e-12);
    }

    #[test]
    fn test_resolve_specs_defaults_lsq() {
        let specs = resolve_specs(&[free_spec(FilterType::LSQ)]).unwrap();
        assert_abs_diff_eq!(specs[0].q_range.0, 0.4, epsilon = 1e-12);
        assert_abs_diff_eq!(specs[0].q_range.1, 0.7, epsilon = 1e-12);
    }

    #[test]
    fn test_resolve_specs_locked_without_value_fails() {
        let spec = FilterSpec {
            filter_type: Some(FilterType::PK),
            fc: None,
            q: Some(1.0),
            gain: Some(0.0),
            optimize_fc: Some(false),
            optimize_q: Some(false),
            optimize_gain: Some(false),
            fc_range: None,
            q_range: None,
            gain_range: (-12.0, 12.0),
        };
        assert!(resolve_specs(&[spec]).is_err());
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let specs = resolve_specs(&[free_spec(FilterType::PK), free_spec(FilterType::LSQ)]).unwrap();
        let filters = vec![pk(1000.0, -3.0, 2.0), lsq(100.0, 2.5, 0.6)];
        let x = encode_params(&filters, &specs);
        assert_eq!(x.len(), 6);
        let mut decoded = filters.clone();
        decode_params(&x, &mut decoded, &specs);
        assert_abs_diff_eq!(decoded[0].fc, 1000.0, epsilon = 1e-9);
        assert_abs_diff_eq!(decoded[0].gain, -3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(decoded[1].q, 0.6, epsilon = 1e-12);
    }

    #[test]
    fn test_encode_locked_params_skipped() {
        let spec = FilterSpec {
            filter_type: Some(FilterType::PK),
            fc: Some(1000.0),
            q: Some(2.0),
            gain: None,
            optimize_fc: Some(false),
            optimize_q: Some(false),
            optimize_gain: None,
            fc_range: None,
            q_range: None,
            gain_range: (-12.0, 12.0),
        };
        let specs = resolve_specs(&[spec]).unwrap();
        let filters = vec![pk(1000.0, -3.0, 2.0)];
        let x = encode_params(&filters, &specs);
        assert_eq!(x.len(), 1);
        assert_abs_diff_eq!(x[0], -3.0, epsilon = 1e-12);
    }

    #[test]
    fn test_build_bounds_len_matches_encode() {
        let specs = resolve_specs(&[free_spec(FilterType::PK), free_spec(FilterType::LSQ)]).unwrap();
        let filters = vec![pk(1000.0, 0.0, 2.0), lsq(100.0, 0.0, 0.6)];
        let x = encode_params(&filters, &specs);
        let bounds = build_bounds(&filters, &specs);
        assert_eq!(bounds.len(), x.len());
        assert_abs_diff_eq!(bounds[0].0, (20.0f64).log10(), epsilon = 1e-12);
        assert_abs_diff_eq!(bounds[0].1, (10000.0f64).log10(), epsilon = 1e-12);
    }

    #[test]
    fn test_total_response_single_pk() {
        let freqs = build_grid(20.0, 20000.0, 1.02);
        let f = pk(1000.0, 6.0, 2.0);
        let resp = total_response(&[f.clone()], &freqs, 44100.0);
        let direct = biquad_response(f.filter_type, f.fc, f.gain, f.q, &freqs, 44100.0);
        for (a, b) in resp.iter().zip(direct.iter()) {
            assert_abs_diff_eq!(a, b, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_sharpness_penalty_shelves_zero() {
        let freqs = build_grid(20.0, 20000.0, 1.02);
        assert_abs_diff_eq!(sharpness_penalty(&lsq(100.0, 3.0, 0.6), &freqs, 44100.0), 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(sharpness_penalty(&hsq(8000.0, -3.0, 0.6), &freqs, 44100.0), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_sharpness_penalty_pk_small_gain() {
        let freqs = build_grid(20.0, 20000.0, 1.02);
        let p = sharpness_penalty(&pk(1000.0, 0.1, 2.0), &freqs, 44100.0);
        assert!(p >= 0.0 && p < 0.01, "expected tiny penalty, got {p}");
    }

    #[test]
    fn test_joint_loss_perfect_match() {
        let freqs = build_grid(20.0, 20000.0, 1.02);
        let f = pk(1000.0, 3.0, 2.0);
        let correction = biquad_response(f.filter_type, f.fc, f.gain, f.q, &freqs, 44100.0);
        let mf = loss_min_f_ix(&freqs);
        let xf = loss_max_f_ix(&freqs);
        let tk = ten_k_ix(&freqs);
        let loss = joint_loss(&[f], &freqs, &correction, 44100.0, mf, xf, tk);
        assert_abs_diff_eq!(loss, 0.0, epsilon = 1e-9);
    }

    #[test]
    fn test_10k_averaging_invariant() {
        let freqs = build_grid(20.0, 20000.0, 1.02);
        let f = pk(1000.0, 3.0, 2.0);
        let mut correction = biquad_response(f.filter_type, f.fc, f.gain, f.q, &freqs, 44100.0);
        let mf = loss_min_f_ix(&freqs);
        let xf = loss_max_f_ix(&freqs);
        let tk = ten_k_ix(&freqs);
        let loss_before = joint_loss(&[f.clone()], &freqs, &correction, 44100.0, mf, xf, tk);
        if tk + 5 < correction.len() {
            correction.swap(tk, tk + 5);
        }
        let loss_after = joint_loss(&[f], &freqs, &correction, 44100.0, mf, xf, tk);
        assert_abs_diff_eq!(loss_before, loss_after, epsilon = 1e-9);
    }

    #[test]
    fn test_init_peaking_no_peaks_fallback() {
        let freqs = build_grid(20.0, 20000.0, 1.02);
        let correction = vec![0.0f64; freqs.len()];
        let spec = resolve_specs(&[free_spec(FilterType::PK)]).unwrap().remove(0);
        let f = init_peaking(&freqs, &correction, &spec, 44100.0);
        assert_eq!(f.filter_type, FilterType::PK);
        assert_abs_diff_eq!(f.gain, 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(f.q, 2.0_f64.sqrt(), epsilon = 1e-9);
    }

    #[test]
    fn test_convergence_triggered_constant_history() {
        let losses = vec![1.0f64; 10];
        assert!(convergence_triggered(&losses, 0.002));
    }

    #[test]
    fn test_convergence_not_triggered_too_few() {
        let losses = vec![1.0f64; 4]; // 4 values: 4 > 4 is false, 4 > 8 is false → no trigger
        assert!(!convergence_triggered(&losses, 0.002));
    }

    #[test]
    fn test_convergence_not_triggered_noisy() {
        let losses: Vec<f64> = (0..12).map(|i| if i % 2 == 0 { 1.0 } else { 2.0 }).collect();
        assert!(!convergence_triggered(&losses, 0.002));
    }

    #[test]
    fn test_compute_pregain_positive_boost() {
        let freqs = build_grid(20.0, 20000.0, 1.02);
        let f = pk(1000.0, 5.0, 2.0);
        let resp = total_response(&[f.clone()], &freqs, 44100.0);
        let max_boost = resp.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let pre = compute_pregain(&[f], &freqs, 44100.0);
        assert!(pre < 0.0);
        assert_abs_diff_eq!(pre, -(max_boost + 0.2), epsilon = 1e-9);
    }

    #[test]
    fn test_compute_pregain_all_cuts() {
        let freqs = build_grid(20.0, 20000.0, 1.02);
        assert_abs_diff_eq!(compute_pregain(&[pk(1000.0, -5.0, 2.0)], &freqs, 44100.0), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_priority_order() {
        // Free HSQ > LSQ > PK (descending sort should give HSQ first)
        let specs =
            resolve_specs(&[free_spec(FilterType::PK), free_spec(FilterType::LSQ), free_spec(FilterType::HSQ)])
                .unwrap();
        let p: Vec<f64> = specs.iter().map(init_priority).collect();
        assert!(p[2] > p[1], "HSQ priority ({}) should exceed LSQ ({})", p[2], p[1]);
        assert!(p[1] > p[0], "LSQ priority ({}) should exceed PK ({})", p[1], p[0]);
    }

    #[test]
    fn test_locked_fc_unchanged_after_optimize() {
        let freqs = build_grid(20.0, 20000.0, 1.02);
        let locked_fc = 500.0;
        let spec = FilterSpec {
            filter_type: Some(FilterType::PK),
            fc: Some(locked_fc),
            q: None,
            gain: None,
            optimize_fc: Some(false),
            optimize_q: None,
            optimize_gain: None,
            fc_range: None,
            q_range: None,
            gain_range: (-12.0, 12.0),
        };
        let specs = resolve_specs(&[spec]).unwrap();
        let correction = vec![2.0f64; freqs.len()];
        let initial = init_filters(&specs, &freqs, &correction, 44100.0);
        let mf = loss_min_f_ix(&freqs);
        let xf = loss_max_f_ix(&freqs);
        let tk = ten_k_ix(&freqs);
        let result = joint_optimize(
            &initial, &specs, &freqs, &correction, 44100.0, mf, xf, tk, Some(0.002),
        )
        .unwrap();
        assert_abs_diff_eq!(result[0].fc, locked_fc, epsilon = 1e-9);
    }
}
