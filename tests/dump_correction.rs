/// Dump our equalization curve (on 1.02 optimizer grid) as JSON for diff vs AutoEQ.
/// Run with: cargo test --test dump_correction -- --nocapture
mod helpers;

use biquad_rust::{center, compensate, equalize, interpolate};
use biquad_rust::types::InterpolateOptions;
use helpers::{load_fr, load_target};
use std::fs;

const CASES: &[(&str, &str)] = &[
    ("hexa",     "bass_heavy"),
    ("origin_s", "bright"),
    ("origin_s", "flat"),
    ("zero2",    "bass_heavy"),
];

#[test]
fn dump_correction_arrays() {
    let debug_dir = "tests/fixtures/debug";
    fs::create_dir_all(debug_dir).ok();

    for &(iem, target_name) in CASES {
        let measured = load_fr(iem);
        let target   = load_target(target_name);

        let opts = InterpolateOptions { step: Some(1.01), f_min: None, f_max: None };
        let interp = interpolate(&measured, &opts);
        let interp_c = center(&interp);

        // Dump error (pre-equalize) stats
        let error = compensate(&interp_c, &target);
        let err_min = error.iter().map(|p| p.db).fold(f64::INFINITY, f64::min);
        let err_max = error.iter().map(|p| p.db).fold(f64::NEG_INFINITY, f64::max);
        println!("Error (pre-equalize): {} points, db range [{err_min:.3}, {err_max:.3}]", error.len());
        println!("  First 3: {:?}", &error[..3].iter().map(|p| (p.freq.round(), (p.db * 1000.0).round() / 1000.0)).collect::<Vec<_>>());
        println!("  Last 3:  {:?}", &error[error.len()-3..].iter().map(|p| (p.freq.round(), (p.db * 1000.0).round() / 1000.0)).collect::<Vec<_>>());

        // Equalize
        let eq = equalize(&error);
        let eq_min = eq.iter().map(|p| p.db).fold(f64::INFINITY, f64::min);
        let eq_max = eq.iter().map(|p| p.db).fold(f64::NEG_INFINITY, f64::max);
        println!("Equalize output: {} points, db range [{eq_min:.3}, {eq_max:.3}]", eq.len());
        println!("  First 3: {:?}", &eq[..3].iter().map(|p| (p.freq.round(), (p.db * 1000.0).round() / 1000.0)).collect::<Vec<_>>());

        // Write 1.01-grid equalize output (BEFORE 1.02 interpolation)
        let json_101: Vec<String> = eq.iter()
            .map(|p| format!("  {{\"freq\": {:.6}, \"db\": {:.6}}}", p.freq, p.db))
            .collect();
        let json101 = format!("[\n{}\n]", json_101.join(",\n"));
        let path101 = format!("{debug_dir}/{iem}__{target_name}__correction_rust_1p01.json");
        fs::write(&path101, &json101).expect("write failed");
        println!("Wrote {path101} ({} points, 1.01 grid)", eq.len());

        let opts102 = InterpolateOptions { step: Some(1.02), f_min: None, f_max: None };
        let eq_102  = interpolate(&eq, &opts102);
        let eq_102_min = eq_102.iter().map(|p| p.db).fold(f64::INFINITY, f64::min);
        let eq_102_max = eq_102.iter().map(|p| p.db).fold(f64::NEG_INFINITY, f64::max);
        println!("Eq on 1.02 grid: {} points, db range [{eq_102_min:.3}, {eq_102_max:.3}]", eq_102.len());

        // Write as JSON
        let json_points: Vec<String> = eq_102.iter()
            .map(|p| format!("  {{\"freq\": {:.6}, \"db\": {:.6}}}", p.freq, p.db))
            .collect();
        let json = format!("[\n{}\n]", json_points.join(",\n"));
        let path = format!("{debug_dir}/{iem}__{target_name}__correction_rust.json");
        fs::write(&path, &json).expect("write failed");
        println!("Wrote {path}");
    }
}
