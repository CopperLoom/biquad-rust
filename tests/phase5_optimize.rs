mod helpers;

use biquad_rust::optimize::{optimize, total_response};
use biquad_rust::types::{Constraints, FilterSpec, FilterType, MinStd};
use helpers::{load_fr, load_golden, load_target, optimizer_grid, rmse};

fn standard_constraints() -> Constraints {
    let specs = vec![
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
    ];
    Constraints {
        filter_specs: specs,
        freq_range: None,
        fs: Some(44100.0),
        min_std: Some(MinStd::Default),
    }
}

/// Interpolate filter responses + pregain to the optimizer grid; returns total cascade in dB.
fn cascade_on_grid(
    result: &biquad_rust::types::OptimizeResult,
    freqs: &[f64],
    fs: f64,
) -> Vec<f64> {
    let resp = total_response(&result.filters, freqs, fs);
    resp.iter().map(|&v| v + result.pregain).collect()
}

#[test]
fn smoke_blessing3_harman_standard() {
    let measured = load_fr("blessing3");
    let target = load_target("harman_ie_2019");
    let constraints = standard_constraints();

    let result = optimize(&measured, &target, &constraints).expect("optimize returned Err");

    assert_eq!(result.filters.len(), 5, "expected 5 filters");
    assert!(result.pregain.is_finite(), "pregain is not finite");
    assert!(
        result.pregain <= 0.0,
        "pregain should be <= 0, got {}",
        result.pregain
    );

    // Compare filter cascade + pregain against golden on the optimizer grid
    let golden = load_golden("blessing3__harman_ie_2019__standard.json");
    let freqs = optimizer_grid();
    let fs = 44100.0;

    let our_cascade = cascade_on_grid(&result, &freqs, fs);
    let golden_cascade = cascade_on_grid(&golden_to_result(&golden), &freqs, fs);

    let err = rmse(&our_cascade, &golden_cascade);
    assert!(
        err <= 2.0,
        "RMSE vs golden = {:.4} dB exceeds 2.0 dB threshold",
        err
    );
}

fn golden_to_result(g: &helpers::GoldenFile) -> biquad_rust::types::OptimizeResult {
    biquad_rust::types::OptimizeResult {
        pregain: g.pregain,
        filters: g.filters.clone(),
    }
}
