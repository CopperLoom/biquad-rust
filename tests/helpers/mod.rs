use biquad_rust::types::{Filter, FreqPoint, OptimizeResult};
use serde::Deserialize;
use std::fs;
use std::path::Path;

#[derive(Debug, Deserialize)]
pub struct GoldenFile {
    pub iem: String,
    pub target: String,
    pub constraint: String,
    pub fs: f64,
    pub pregain: f64,
    pub filters: Vec<Filter>,
}

pub fn load_golden(name: &str) -> GoldenFile {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/golden")
        .join(name);
    let json = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Failed to read golden file {name}: {e}"));
    serde_json::from_str(&json)
        .unwrap_or_else(|e| panic!("Failed to parse golden file {name}: {e}"))
}

pub fn load_fr(iem: &str) -> Vec<FreqPoint> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/fr")
        .join(format!("{iem}.json"));
    let json = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Failed to read FR fixture {iem}: {e}"));
    serde_json::from_str(&json).unwrap_or_else(|e| panic!("Failed to parse FR fixture {iem}: {e}"))
}

pub fn load_target(target: &str) -> Vec<FreqPoint> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/targets")
        .join(format!("{target}.json"));
    let json = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Failed to read target fixture {target}: {e}"));
    serde_json::from_str(&json)
        .unwrap_or_else(|e| panic!("Failed to parse target fixture {target}: {e}"))
}

/// Build log-spaced frequency grid (1.02 step, 20–20000 Hz) for RMSE comparison.
pub fn optimizer_grid() -> Vec<f64> {
    let mut freqs = Vec::new();
    let mut f = 20.0_f64;
    while f <= 20000.0 {
        freqs.push(f);
        f *= 1.02;
    }
    freqs
}

/// Root mean square error between two equal-length slices.
pub fn rmse(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mse: f64 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>() / a.len() as f64;
    mse.sqrt()
}

pub fn load_phase4_expected(iem: &str, target: &str) -> Vec<FreqPoint> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/phase4_equalize")
        .join(format!("{iem}__{target}.json"));
    let json = fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Missing phase4 fixture {iem}__{target}: {e}\nRun: python3 tests/generate_phase4_expected.py"));
    serde_json::from_str(&json)
        .unwrap_or_else(|e| panic!("Failed to parse phase4 fixture {iem}__{target}: {e}"))
}

// cascade_response and assert_rmse_le are added in Phase 2 when biquad_response is implemented.
