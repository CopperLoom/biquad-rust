/// Feed AutoEQ's x0 to our Rust SLSQP and compare.
/// Same x0 → same result = initialization differs.
/// Same x0 → different result = SLSQP implementations differ.
mod helpers;

use biquad_rust::optimize::{compute_x0, optimize_from_x0};
use biquad_rust::types::{Constraints, FilterSpec, FilterType, MinStd};
use helpers::{load_fr, load_golden, load_target, optimizer_grid, rmse};
use std::path::Path;

fn qudelix_10() -> Constraints {
    let mut specs = vec![FilterSpec {
        filter_type: Some(FilterType::LSQ),
        fc: None,
        q: None,
        gain: None,
        optimize_fc: None,
        optimize_q: None,
        optimize_gain: None,
        fc_range: None,
        q_range: Some((0.5, 10.0)),
        gain_range: (-12.0, 12.0),
    }];
    for _ in 0..8 {
        specs.push(FilterSpec {
            filter_type: Some(FilterType::PK),
            fc: None,
            q: None,
            gain: None,
            optimize_fc: None,
            optimize_q: None,
            optimize_gain: None,
            fc_range: None,
            q_range: Some((0.5, 10.0)),
            gain_range: (-12.0, 12.0),
        });
    }
    specs.push(FilterSpec {
        filter_type: Some(FilterType::HSQ),
        fc: None,
        q: None,
        gain: None,
        optimize_fc: None,
        optimize_q: None,
        optimize_gain: None,
        fc_range: None,
        q_range: Some((0.5, 10.0)),
        gain_range: (-12.0, 12.0),
    });
    Constraints {
        filter_specs: specs,
        freq_range: None,
        fs: Some(44100.0),
        min_std: Some(MinStd::Default),
    }
}

fn restricted() -> Constraints {
    Constraints {
        filter_specs: vec![
            FilterSpec {
                filter_type: Some(FilterType::PK),
                fc: None,
                q: None,
                gain: None,
                optimize_fc: None,
                optimize_q: None,
                optimize_gain: None,
                fc_range: None,
                q_range: Some((1.0, 5.0)),
                gain_range: (-6.0, 6.0),
            },
            FilterSpec {
                filter_type: Some(FilterType::PK),
                fc: None,
                q: None,
                gain: None,
                optimize_fc: None,
                optimize_q: None,
                optimize_gain: None,
                fc_range: None,
                q_range: Some((1.0, 5.0)),
                gain_range: (-6.0, 6.0),
            },
            FilterSpec {
                filter_type: Some(FilterType::PK),
                fc: None,
                q: None,
                gain: None,
                optimize_fc: None,
                optimize_q: None,
                optimize_gain: None,
                fc_range: None,
                q_range: Some((1.0, 5.0)),
                gain_range: (-6.0, 6.0),
            },
        ],
        freq_range: None,
        fs: Some(44100.0),
        min_std: Some(MinStd::Default),
    }
}

fn standard() -> Constraints {
    Constraints {
        filter_specs: vec![
            FilterSpec {
                filter_type: Some(FilterType::LSQ),
                fc: None,
                q: None,
                gain: None,
                optimize_fc: None,
                optimize_q: None,
                optimize_gain: None,
                fc_range: None,
                q_range: Some((0.5, 10.0)),
                gain_range: (-12.0, 12.0),
            },
            FilterSpec {
                filter_type: Some(FilterType::PK),
                fc: None,
                q: None,
                gain: None,
                optimize_fc: None,
                optimize_q: None,
                optimize_gain: None,
                fc_range: None,
                q_range: Some((0.5, 10.0)),
                gain_range: (-12.0, 12.0),
            },
            FilterSpec {
                filter_type: Some(FilterType::PK),
                fc: None,
                q: None,
                gain: None,
                optimize_fc: None,
                optimize_q: None,
                optimize_gain: None,
                fc_range: None,
                q_range: Some((0.5, 10.0)),
                gain_range: (-12.0, 12.0),
            },
            FilterSpec {
                filter_type: Some(FilterType::PK),
                fc: None,
                q: None,
                gain: None,
                optimize_fc: None,
                optimize_q: None,
                optimize_gain: None,
                fc_range: None,
                q_range: Some((0.5, 10.0)),
                gain_range: (-12.0, 12.0),
            },
            FilterSpec {
                filter_type: Some(FilterType::HSQ),
                fc: None,
                q: None,
                gain: None,
                optimize_fc: None,
                optimize_q: None,
                optimize_gain: None,
                fc_range: None,
                q_range: Some((0.5, 10.0)),
                gain_range: (-12.0, 12.0),
            },
        ],
        freq_range: None,
        fs: Some(44100.0),
        min_std: Some(MinStd::Default),
    }
}

fn load_x0(name: &str) -> Vec<f64> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/debug")
        .join(format!("{name}__x0.json"));
    let j: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(&path)
            .unwrap_or_else(|_| panic!("Run same_x0_test.py first to generate {path:?}")),
    )
    .unwrap();
    j["x0"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect()
}

fn report(case: &str, constraints: &Constraints) {
    let iem = case.split("__").next().unwrap();
    let target_name = case.split("__").nth(1).unwrap();
    let constraint_name = case.split("__").nth(2).unwrap();

    println!("\n{}\nCASE: {case}\n{}", "=".repeat(60), "=".repeat(60));

    let measured = load_fr(iem);
    let target = load_target(target_name);

    // Our normal x0 (from our init)
    let (our_x0, our_init) = compute_x0(&measured, &target, constraints).unwrap();
    println!(
        "\nOur x0:    {:?}",
        our_x0.iter().map(|v| format!("{v:.4}")).collect::<Vec<_>>()
    );
    println!(
        "AutoEQ x0: {:?}",
        load_x0(case)
            .iter()
            .map(|v| format!("{v:.4}"))
            .collect::<Vec<_>>()
    );

    println!("\nOur init filters:");
    for f in &our_init {
        println!(
            "  {:3?}  fc={:8.1}  gain={:7.3}  q={:.3}",
            f.filter_type, f.fc, f.gain, f.q
        );
    }

    // Run Rust optimizer from AutoEQ's x0
    let autoeq_x0 = load_x0(case);
    let result_from_autoeq_x0 = optimize_from_x0(&measured, &target, constraints, autoeq_x0)
        .expect("optimize_from_x0 failed");

    println!("\nRust result from AutoEQ x0:");
    for f in &result_from_autoeq_x0.filters {
        println!(
            "  {:3?}  fc={:8.1}  gain={:7.3}  q={:.3}",
            f.filter_type, f.fc, f.gain, f.q
        );
    }

    // Compare cascades
    let freqs = optimizer_grid();
    use biquad_rust::optimize::total_response;
    let golden = load_golden(&format!("{case}.json"));
    let golden_cascade: Vec<f64> = total_response(&golden.filters, &freqs, 44100.0)
        .iter()
        .map(|&v| v + golden.pregain)
        .collect();
    let rust_cascade: Vec<f64> = total_response(&result_from_autoeq_x0.filters, &freqs, 44100.0)
        .iter()
        .map(|&v| v + result_from_autoeq_x0.pregain)
        .collect();
    println!(
        "\nRMSE (Rust-from-AutoEQ-x0 vs golden): {:.4} dB",
        rmse(&rust_cascade, &golden_cascade)
    );

    // Verdict
    let our_x0_val = compute_x0(&measured, &target, constraints).unwrap().0;
    let autoeq_x0_val = load_x0(case);
    let x0_diff: f64 = our_x0_val
        .iter()
        .zip(&autoeq_x0_val)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    println!("\nx0 max diff (ours vs AutoEQ): {x0_diff:.6}");
    if x0_diff > 0.01 {
        println!(">>> INIT DIFFERS — x0 mismatch is the cause");
    } else {
        println!(">>> INIT MATCHES — divergence is in the SLSQP solver");
    }
}

#[test]
fn same_x0_comparison() {
    report("hexa__bass_heavy__qudelix_10", &qudelix_10());
    report("hexa__bass_heavy__standard", &standard());
    report("hexa__diffuse_field__restricted", &restricted());
    report("origin_s__bass_heavy__qudelix_10", &qudelix_10());
    report("zero2__bass_heavy__restricted", &restricted());
}
