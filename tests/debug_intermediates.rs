/// Diagnostic test: dump intermediate values for the 5 failing golden combinations.
/// Run with: cargo test --test debug_intermediates -- --nocapture
mod helpers;

use biquad_rust::optimize::{optimize, total_response};
use biquad_rust::types::{Constraints, FilterSpec, FilterType, FreqPoint, MinStd};
use biquad_rust::{compensate, equalize, interpolate};
use biquad_rust::types::InterpolateOptions;
use helpers::{load_fr, load_golden, load_target, optimizer_grid, rmse};

const FAILING: &[(&str, &str, &str)] = &[
    ("hexa",     "bass_heavy", "qudelix_10"),
    ("hexa",     "bass_heavy", "standard"),
    ("origin_s", "bright",     "qudelix_10"),
    ("origin_s", "flat",       "qudelix_10"),
    ("zero2",    "bass_heavy", "restricted"),
];

fn pk_spec(gain_range: (f64, f64), q_range: (f64, f64)) -> FilterSpec {
    FilterSpec {
        filter_type: Some(FilterType::PK),
        fc: None, q: None, gain: None,
        optimize_fc: None, optimize_q: None, optimize_gain: None,
        fc_range: None,
        q_range: Some(q_range),
        gain_range,
    }
}

fn shelf_spec(ft: FilterType, gain_range: (f64, f64), q_range: (f64, f64)) -> FilterSpec {
    FilterSpec {
        filter_type: Some(ft),
        fc: None, q: None, gain: None,
        optimize_fc: None, optimize_q: None, optimize_gain: None,
        fc_range: None,
        q_range: Some(q_range),
        gain_range,
    }
}

fn constraints_for(name: &str) -> Constraints {
    match name {
        "standard" => Constraints {
            filter_specs: vec![
                shelf_spec(FilterType::LSQ, (-12.0, 12.0), (0.5, 10.0)),
                pk_spec((-12.0, 12.0), (0.5, 10.0)),
                pk_spec((-12.0, 12.0), (0.5, 10.0)),
                pk_spec((-12.0, 12.0), (0.5, 10.0)),
                shelf_spec(FilterType::HSQ, (-12.0, 12.0), (0.5, 10.0)),
            ],
            freq_range: None, fs: Some(44100.0), min_std: Some(MinStd::Default),
        },
        "restricted" => Constraints {
            filter_specs: vec![
                pk_spec((-6.0, 6.0), (1.0, 5.0)),
                pk_spec((-6.0, 6.0), (1.0, 5.0)),
                pk_spec((-6.0, 6.0), (1.0, 5.0)),
            ],
            freq_range: None, fs: Some(44100.0), min_std: Some(MinStd::Default),
        },
        "qudelix_10" => {
            let mut specs = vec![shelf_spec(FilterType::LSQ, (-12.0, 12.0), (0.5, 10.0))];
            for _ in 0..8 { specs.push(pk_spec((-12.0, 12.0), (0.5, 10.0))); }
            specs.push(shelf_spec(FilterType::HSQ, (-12.0, 12.0), (0.5, 10.0)));
            Constraints { filter_specs: specs, freq_range: None, fs: Some(44100.0), min_std: Some(MinStd::Default) }
        }
        other => panic!("unknown constraint: {other}"),
    }
}

fn band_stats(label: &str, fr: &[FreqPoint]) {
    let bands = [(20.0, 200.0, "sub-bass"), (200.0, 2000.0, "mid"),
                 (2000.0, 8000.0, "upper-mid"), (8000.0, 20000.0, "treble")];
    let parts: Vec<String> = bands.iter().filter_map(|&(lo, hi, name)| {
        let vals: Vec<f64> = fr.iter().filter(|p| p.freq >= lo && p.freq <= hi).map(|p| p.db).collect();
        if vals.is_empty() { return None; }
        let mn = vals.iter().cloned().fold(f64::INFINITY, f64::min);
        let mx = vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        Some(format!("{name}:[{mn:.2},{mx:.2}]"))
    }).collect();
    println!("  {label}: {}", parts.join(" "));
}

fn spacing_report(label: &str, filters: &[biquad_rust::types::Filter]) {
    let mut pk_fcs: Vec<f64> = filters.iter()
        .filter(|f| f.filter_type == FilterType::PK)
        .map(|f| f.fc)
        .collect();
    pk_fcs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if pk_fcs.len() < 2 { return; }
    let octs: Vec<f64> = pk_fcs.windows(2).map(|w| (w[1] / w[0]).log2()).collect();
    println!("  {label} PK fc (Hz): {:?}", pk_fcs.iter().map(|f| format!("{f:.0}")).collect::<Vec<_>>());
    println!("  {label} PK spacing (oct): {:?}  min={:.2}",
             octs.iter().map(|o| format!("{o:.2}")).collect::<Vec<_>>(),
             octs.iter().cloned().fold(f64::INFINITY, f64::min));
    if octs.iter().cloned().fold(f64::INFINITY, f64::min) < 0.5 {
        println!("  *** CLUSTER detected ***");
    }
}

#[test]
fn debug_pipeline_intermediates() {
    let grid = optimizer_grid();

    for &(iem, target_name, constraint_name) in FAILING {
        let case = format!("{iem}__{target_name}__{constraint_name}");
        println!("\n{}", "=".repeat(68));
        println!("CASE: {case}");
        println!("{}", "=".repeat(68));

        let measured = load_fr(iem);
        let target   = load_target(target_name);

        // Stage 1: interpolate + compensate (produces error curve)
        let opts = InterpolateOptions { step: Some(1.01), f_min: None, f_max: None };
        let interp = interpolate(&measured, &opts);
        let error  = compensate(&interp, &target);

        println!("\n[1] Our equalization curve (post-equalize, 1.02-grid excerpt):");
        let eq = equalize(&error);
        // Re-interpolate to 1.02 grid for direct comparison with AutoEQ
        let opts102 = InterpolateOptions { step: Some(1.02), f_min: None, f_max: None };
        let eq_102 = interpolate(&eq, &opts102);
        band_stats("eq", &eq_102);

        // Stage 2: optimize and print final filters
        let constraints = constraints_for(constraint_name);
        let result = optimize(&measured, &target, &constraints).expect("optimize failed");

        println!("\n[2] Our final filters:");
        for f in &result.filters {
            println!("  {:<3?}  fc={:8.1} Hz  gain={:7.3} dB  q={:.3}",
                     f.filter_type, f.fc, f.gain, f.q);
        }
        println!("  pregain: {:.3} dB", result.pregain);
        spacing_report("FINAL", &result.filters);

        // Stage 3: our MSE on optimizer grid
        let our_cascade: Vec<f64> = total_response(&result.filters, &grid, 44100.0)
            .iter().map(|&v| v + result.pregain).collect();

        // Golden cascade for RMSE comparison
        let golden = load_golden(&format!("{case}.json"));
        let golden_cascade: Vec<f64> = total_response(&golden.filters, &grid, 44100.0)
            .iter().map(|&v| v + golden.pregain).collect();

        let err = rmse(&our_cascade, &golden_cascade);
        println!("\n[3] RMSE vs golden on optimizer grid: {err:.4} dB");

        // Stage 4: MSE of our cascade vs equalization target (approximates optimizer loss)
        let eq_on_grid: Vec<f64> = grid.iter().map(|&f| {
            // linear interpolation of eq_102 at f
            let pts = &eq_102;
            match pts.binary_search_by(|p| p.freq.partial_cmp(&f).unwrap()) {
                Ok(i) => pts[i].db,
                Err(0) => pts[0].db,
                Err(i) if i >= pts.len() => pts[pts.len()-1].db,
                Err(i) => {
                    let lo = &pts[i-1]; let hi = &pts[i];
                    let t = (f - lo.freq) / (hi.freq - lo.freq);
                    lo.db + t * (hi.db - lo.db)
                }
            }
        }).collect();
        let our_mse: f64 = our_cascade.iter().zip(&eq_on_grid)
            .map(|(c, e)| (c - e).powi(2)).sum::<f64>() / grid.len() as f64;
        let golden_mse: f64 = golden_cascade.iter().zip(&eq_on_grid)
            .map(|(c, e)| (c - e).powi(2)).sum::<f64>() / grid.len() as f64;
        println!("[4] Approx MSE vs equalization target:");
        println!("    Our filters: {our_mse:.6}  |  Golden: {golden_mse:.6}");
        if our_mse > golden_mse * 1.1 {
            println!("    *** Our solution is worse — likely suboptimal local minimum ***");
        } else {
            println!("    Our MSE is comparable — likely a different local optimum, not a bug");
        }
    }
}
