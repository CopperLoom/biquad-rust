mod helpers;

use biquad_rust::{
    center,
    compensate::compensate,
    equalize::equalize,
    interpolate::interpolate,
    types::{FreqPoint, InterpolateOptions},
};
use helpers::{load_fr, load_phase4_expected, load_target};

const IEMS: &[&str] = &["blessing3", "hexa", "andromeda", "zero2", "origin_s"];
const TARGETS: &[&str] = &[
    "harman_ie_2019",
    "diffuse_field",
    "flat",
    "v_shaped",
    "bass_heavy",
    "bright",
];

fn rmse(a: &[FreqPoint], b: &[FreqPoint]) -> f64 {
    assert_eq!(a.len(), b.len(), "length mismatch in rmse");
    let sum_sq: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x.db - y.db).powi(2))
        .sum();
    (sum_sq / a.len() as f64).sqrt()
}

fn run_pipeline(iem: &str, target_name: &str) -> Vec<FreqPoint> {
    let measured = load_fr(iem);
    let target = load_target(target_name);
    let opts = InterpolateOptions {
        step: Some(1.01),
        f_min: None,
        f_max: None,
    };
    let meas_interp = center(&interpolate(&measured, &opts));
    let error = compensate(&meas_interp, &target);
    equalize(&error)
}

#[test]
fn test_equalize_all_fixtures() {
    let mut results: Vec<(String, f64)> = Vec::new();
    let mut failures: Vec<String> = Vec::new();

    for &iem in IEMS {
        for &target in TARGETS {
            let eq = run_pipeline(iem, target);
            let expected = load_phase4_expected(iem, target);
            let err = rmse(&eq, &expected);
            let label = format!("{iem}__{target}");
            if err >= 0.5 {
                failures.push(format!("  FAIL {label}: {err:.4} dB RMSE"));
            }
            results.push((label, err));
        }
    }

    let worst = results.iter().map(|(_, e)| *e).fold(0.0_f64, f64::max);
    println!(
        "equalize parity vs AutoEQ Python ({} pairs):",
        results.len()
    );
    for (label, err) in &results {
        println!("  {label}: {err:.4} dB");
    }
    println!("worst-case RMSE: {worst:.4} dB");

    assert!(
        failures.is_empty(),
        "equalize RMSE exceeded 0.5 dB threshold:\n{}",
        failures.join("\n")
    );
}
