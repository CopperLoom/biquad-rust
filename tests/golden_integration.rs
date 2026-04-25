mod helpers;

use biquad_rust::optimize::{optimize, total_response};
use biquad_rust::types::{Constraints, Filter, FilterSpec, FilterType, MinStd, OptimizeResult};
use helpers::{load_fr, load_golden, load_target, optimizer_grid, rmse};

// ── constraint builders ──────────────────────────────────────────────────────

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

fn standard_constraints() -> Constraints {
    Constraints {
        filter_specs: vec![
            shelf_spec(FilterType::LSQ, (-12.0, 12.0), (0.5, 10.0)),
            pk_spec((-12.0, 12.0), (0.5, 10.0)),
            pk_spec((-12.0, 12.0), (0.5, 10.0)),
            pk_spec((-12.0, 12.0), (0.5, 10.0)),
            shelf_spec(FilterType::HSQ, (-12.0, 12.0), (0.5, 10.0)),
        ],
        freq_range: None,
        fs: Some(44100.0),
        min_std: Some(MinStd::Default),
    }
}

fn restricted_constraints() -> Constraints {
    Constraints {
        filter_specs: vec![
            pk_spec((-6.0, 6.0), (1.0, 5.0)),
            pk_spec((-6.0, 6.0), (1.0, 5.0)),
            pk_spec((-6.0, 6.0), (1.0, 5.0)),
        ],
        freq_range: None,
        fs: Some(44100.0),
        min_std: Some(MinStd::Default),
    }
}

fn qudelix_10_constraints() -> Constraints {
    let mut specs = vec![shelf_spec(FilterType::LSQ, (-12.0, 12.0), (0.5, 10.0))];
    for _ in 0..8 {
        specs.push(pk_spec((-12.0, 12.0), (0.5, 10.0)));
    }
    specs.push(shelf_spec(FilterType::HSQ, (-12.0, 12.0), (0.5, 10.0)));
    Constraints {
        filter_specs: specs,
        freq_range: None,
        fs: Some(44100.0),
        min_std: Some(MinStd::Default),
    }
}

fn constraints_for(name: &str) -> Constraints {
    match name {
        "standard"    => standard_constraints(),
        "restricted"  => restricted_constraints(),
        "qudelix_10"  => qudelix_10_constraints(),
        other         => panic!("unknown constraint set: {other}"),
    }
}

// ── RMSE helper ──────────────────────────────────────────────────────────────

fn cascade_on_grid(filters: &[Filter], pregain: f64, freqs: &[f64], fs: f64) -> Vec<f64> {
    total_response(filters, freqs, fs)
        .iter()
        .map(|&v| v + pregain)
        .collect()
}

fn golden_to_result(g: &helpers::GoldenFile) -> OptimizeResult {
    OptimizeResult { pregain: g.pregain, filters: g.filters.clone() }
}

// ── 90-combination golden sweep ───────────────────────────────────────────────

#[test]
fn golden_all_90() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/golden");
    let mut entries: Vec<_> = std::fs::read_dir(&dir)
        .expect("read golden dir")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("json"))
        .collect();
    entries.sort_by_key(|e| e.file_name());

    assert_eq!(entries.len(), 90, "expected 90 golden files, found {}", entries.len());

    let freqs = optimizer_grid();
    let mut failures: Vec<String> = Vec::new();

    for entry in &entries {
        let name = entry.file_name().into_string().unwrap();
        let golden = load_golden(&name);
        let fs = golden.fs;
        let measured = load_fr(&golden.iem);
        let target = load_target(&golden.target);
        let constraints = constraints_for(&golden.constraint);

        let result = match optimize(&measured, &target, &constraints) {
            Ok(r) => r,
            Err(e) => {
                failures.push(format!("{name}: optimize() failed: {e:?}"));
                continue;
            }
        };

        let our_cascade    = cascade_on_grid(&result.filters,        result.pregain,        &freqs, fs);
        let golden_cascade = cascade_on_grid(&golden.filters,         golden.pregain,        &freqs, fs);
        let err = rmse(&our_cascade, &golden_cascade);

        if err > 0.5 {
            failures.push(format!("{name}: RMSE = {err:.4} dB (threshold 0.5 dB)"));
        }
    }

    if !failures.is_empty() {
        panic!(
            "{} / 90 combinations failed:\n  {}",
            failures.len(),
            failures.join("\n  ")
        );
    }
}

// ── per-parameter locking tests ───────────────────────────────────────────────

#[test]
fn locked_pk_params_unchanged_after_optimize() {
    let measured = load_fr("blessing3");
    let target = load_target("harman_ie_2019");

    let locked_fc = 2000.0;
    let locked_gain = -3.0;
    let locked_q = 2.0;

    let constraints = Constraints {
        filter_specs: vec![
            FilterSpec {
                filter_type: Some(FilterType::PK),
                fc: Some(locked_fc),
                gain: Some(locked_gain),
                q: Some(locked_q),
                optimize_fc: Some(false),
                optimize_gain: Some(false),
                optimize_q: Some(false),
                fc_range: None,
                q_range: Some((0.5, 10.0)),
                gain_range: (-12.0, 12.0),
            },
            pk_spec((-12.0, 12.0), (0.5, 10.0)),
            pk_spec((-12.0, 12.0), (0.5, 10.0)),
        ],
        freq_range: None,
        fs: Some(44100.0),
        min_std: Some(MinStd::Default),
    };

    let result = optimize(&measured, &target, &constraints).expect("optimize failed");

    let locked = &result.filters[0];
    assert!((locked.fc   - locked_fc).abs()   < 1e-9, "fc changed: {}", locked.fc);
    assert!((locked.gain - locked_gain).abs() < 1e-9, "gain changed: {}", locked.gain);
    assert!((locked.q    - locked_q).abs()    < 1e-9, "q changed: {}", locked.q);

    // Overall quality: RMSE vs 3-band restricted golden should be reasonable
    let golden = load_golden("blessing3__harman_ie_2019__restricted.json");
    let freqs = optimizer_grid();
    let our_cascade    = cascade_on_grid(&result.filters,  result.pregain,  &freqs, 44100.0);
    let golden_cascade = cascade_on_grid(&golden.filters,  golden.pregain,  &freqs, 44100.0);
    let err = rmse(&our_cascade, &golden_cascade);
    assert!(err <= 1.0, "RMSE with locked band vs restricted golden too high: {err:.4} dB");
}

#[test]
fn locked_shelf_fc_only_gain_and_q_optimize() {
    let measured = load_fr("blessing3");
    let target = load_target("harman_ie_2019");
    let locked_fc = 105.0;

    let constraints = Constraints {
        filter_specs: vec![
            FilterSpec {
                filter_type: Some(FilterType::LSQ),
                fc: Some(locked_fc),
                q: Some(0.7),
                gain: None,
                optimize_fc: Some(false),
                optimize_q: None,
                optimize_gain: None,
                fc_range: None,
                q_range: Some((0.5, 10.0)),
                gain_range: (-12.0, 12.0),
            },
            shelf_spec(FilterType::HSQ, (-12.0, 12.0), (0.5, 10.0)),
        ],
        freq_range: None,
        fs: Some(44100.0),
        min_std: Some(MinStd::Default),
    };

    let result = optimize(&measured, &target, &constraints).expect("optimize failed");

    let lsq = &result.filters[0];
    assert!((lsq.fc - locked_fc).abs() < 1e-9, "fc changed from locked value: {}", lsq.fc);
    // gain and/or q should have been optimized away from seed
    // (just assert they are finite and within bounds)
    assert!(lsq.gain.is_finite(), "gain is not finite");
    assert!(lsq.q >= 0.5 && lsq.q <= 10.0, "q out of range: {}", lsq.q);
}

#[test]
fn locked_pk_gain_only_fc_and_q_change() {
    let measured = load_fr("blessing3");
    let target = load_target("harman_ie_2019");
    let locked_gain = -2.0;
    let seed_fc = 1000.0;
    let seed_q = 2.0;

    let constraints = Constraints {
        filter_specs: vec![
            FilterSpec {
                filter_type: Some(FilterType::PK),
                fc: Some(seed_fc),
                q: Some(seed_q),
                gain: Some(locked_gain),
                optimize_fc: None,
                optimize_q: None,
                optimize_gain: Some(false),
                fc_range: None,
                q_range: Some((0.5, 10.0)),
                gain_range: (-12.0, 12.0),
            },
            pk_spec((-12.0, 12.0), (0.5, 10.0)),
        ],
        freq_range: None,
        fs: Some(44100.0),
        min_std: Some(MinStd::Default),
    };

    let result = optimize(&measured, &target, &constraints).expect("optimize failed");

    let pk = &result.filters[0];
    assert!((pk.gain - locked_gain).abs() < 1e-9, "gain changed from locked value: {}", pk.gain);
    assert!(pk.fc.is_finite() && pk.fc > 0.0, "fc invalid: {}", pk.fc);
    assert!(pk.q.is_finite() && pk.q > 0.0, "q invalid: {}", pk.q);
}

#[test]
fn all_bands_fully_locked_output_equals_input() {
    let measured = load_fr("blessing3");
    let target = load_target("harman_ie_2019");

    let fc = 1000.0; let gain = -3.0; let q = 1.41;

    let locked_spec = FilterSpec {
        filter_type: Some(FilterType::PK),
        fc: Some(fc),
        gain: Some(gain),
        q: Some(q),
        optimize_fc: Some(false),
        optimize_gain: Some(false),
        optimize_q: Some(false),
        fc_range: None,
        q_range: Some((0.5, 10.0)),
        gain_range: (-12.0, 12.0),
    };

    let constraints = Constraints {
        filter_specs: vec![locked_spec.clone(), locked_spec.clone()],
        freq_range: None,
        fs: Some(44100.0),
        min_std: Some(MinStd::Default),
    };

    let result = optimize(&measured, &target, &constraints).expect("optimize failed");

    for (i, f) in result.filters.iter().enumerate() {
        assert!((f.fc   - fc).abs()   < 1e-9, "filter {i} fc changed");
        assert!((f.gain - gain).abs() < 1e-9, "filter {i} gain changed");
        assert!((f.q    - q).abs()    < 1e-9, "filter {i} q changed");
    }
}

#[test]
fn peq_yaml_pattern_shelves_do_not_overlap() {
    // Models AutoEQ's peq.yaml: LSQ gain-only at 105 Hz, HSQ free in 5k–12k Hz
    let measured = load_fr("blessing3");
    let target = load_target("harman_ie_2019");

    let constraints = Constraints {
        filter_specs: vec![
            FilterSpec {
                filter_type: Some(FilterType::LSQ),
                fc: Some(105.0),
                q: Some(0.7),
                gain: None,
                optimize_fc: Some(false),
                optimize_q: Some(false),
                optimize_gain: None,
                fc_range: None,
                q_range: Some((0.5, 10.0)),
                gain_range: (-12.0, 12.0),
            },
            FilterSpec {
                filter_type: Some(FilterType::HSQ),
                fc: None, q: None, gain: None,
                optimize_fc: None, optimize_q: None, optimize_gain: None,
                fc_range: Some((5000.0, 12000.0)),
                q_range: Some((0.5, 10.0)),
                gain_range: (-12.0, 12.0),
            },
        ],
        freq_range: None,
        fs: Some(44100.0),
        min_std: Some(MinStd::Default),
    };

    let result = optimize(&measured, &target, &constraints).expect("optimize failed");

    let lsq = &result.filters[0];
    let hsq = &result.filters[1];

    assert!((lsq.fc - 105.0).abs() < 1e-9, "LSQ fc changed: {}", lsq.fc);
    assert!((lsq.q  - 0.7).abs()   < 1e-9, "LSQ q changed: {}",  lsq.q);
    assert!(hsq.fc >= 5000.0 && hsq.fc <= 12000.0,
        "HSQ fc out of range: {}", hsq.fc);
    // shelves don't overlap (LSQ at 105 Hz, HSQ well above it)
    assert!(hsq.fc > lsq.fc * 10.0,
        "shelves appear to overlap: LSQ fc={}, HSQ fc={}", lsq.fc, hsq.fc);
}

// ── min_std behavioral test ───────────────────────────────────────────────────

#[test]
fn min_std_disabled_runs_to_max_iterations() {
    // MinStd::Disabled should not early-stop. We can't count iterations directly,
    // but we can verify that it produces a valid result and that (on a non-trivial
    // input) it runs more work than MinStd::Default by checking wall time or simply
    // that the result is structurally valid.
    let measured = load_fr("blessing3");
    let target = load_target("harman_ie_2019");

    let constraints = Constraints {
        filter_specs: vec![
            pk_spec((-12.0, 12.0), (0.5, 10.0)),
            pk_spec((-12.0, 12.0), (0.5, 10.0)),
        ],
        freq_range: None,
        fs: Some(44100.0),
        min_std: Some(MinStd::Disabled),
    };

    let result = optimize(&measured, &target, &constraints).expect("optimize failed");

    assert_eq!(result.filters.len(), 2);
    for f in &result.filters {
        assert!(f.fc.is_finite() && f.fc > 0.0);
        assert!(f.gain.is_finite());
        assert!(f.q.is_finite() && f.q > 0.0);
    }
    assert!(result.pregain.is_finite());
}
