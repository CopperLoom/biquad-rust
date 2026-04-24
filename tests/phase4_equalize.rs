mod helpers;

use biquad_rust::{
    center,
    compensate::compensate,
    equalize::equalize,
    interpolate::interpolate,
    types::{FreqPoint, InterpolateOptions},
};
use helpers::{load_fr, load_target};

fn rmse(a: &[FreqPoint], b: &[FreqPoint]) -> f64 {
    assert_eq!(a.len(), b.len());
    let sum_sq: f64 = a.iter().zip(b.iter()).map(|(x, y)| (x.db - y.db).powi(2)).sum();
    (sum_sq / a.len() as f64).sqrt()
}

fn load_json_freqpoints(path: &str) -> Vec<FreqPoint> {
    let json = std::fs::read_to_string(path)
        .unwrap_or_else(|_| panic!("Missing {path}"));
    let raw: Vec<serde_json::Value> = serde_json::from_str(&json).unwrap();
    raw.iter()
        .map(|v| FreqPoint { freq: v["freq"].as_f64().unwrap(), db: v["db"].as_f64().unwrap() })
        .collect()
}

#[test]
fn test_equalize_blessing3_harman_rmse() {
    let measured = load_fr("blessing3");
    let target = load_target("harman_ie_2019");

    let opts = InterpolateOptions { step: Some(1.01), f_min: None, f_max: None };
    // Pipeline: interpolate → center measured → compensate (centers target internally)
    let meas_interp = center(&interpolate(&measured, &opts));
    let error = compensate(&meas_interp, &target);
    let eq = equalize(&error);

    let expected = load_json_freqpoints("/tmp/test_eq_expected.json");

    println!("Rust eq[0..3]:  {:?}", eq[..3].iter().map(|p| p.db).collect::<Vec<_>>());
    println!("Python[0..3]:   {:?}", expected[..3].iter().map(|p| p.db).collect::<Vec<_>>());

    let err = rmse(&eq, &expected);
    println!("equalize RMSE vs AutoEQ Python: {:.4} dB", err);
    assert!(err < 0.5, "equalize RMSE {:.4} dB exceeds 0.5 dB threshold", err);
}
