use criterion::{Criterion, black_box, criterion_group, criterion_main};
use std::{fs, path::Path};

use biquad_rust::{
    optimize::optimize,
    types::{Constraints, FilterSpec, FilterType, FreqPoint, MinStd},
};

// ── fixture loading ───────────────────────────────────────────────────────────

fn load_fr(iem: &str) -> Vec<FreqPoint> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/fr")
        .join(format!("{iem}.json"));
    serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap()
}

fn load_target(target: &str) -> Vec<FreqPoint> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/targets")
        .join(format!("{target}.json"));
    serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap()
}

// ── constraint builders ───────────────────────────────────────────────────────

fn pk_spec(gain_range: (f64, f64), q_range: (f64, f64)) -> FilterSpec {
    FilterSpec {
        filter_type: Some(FilterType::PK),
        fc: None,
        q: None,
        gain: None,
        optimize_fc: None,
        optimize_q: None,
        optimize_gain: None,
        fc_range: None,
        q_range: Some(q_range),
        gain_range,
    }
}

fn shelf_spec(ft: FilterType, gain_range: (f64, f64), q_range: (f64, f64)) -> FilterSpec {
    FilterSpec {
        filter_type: Some(ft),
        fc: None,
        q: None,
        gain: None,
        optimize_fc: None,
        optimize_q: None,
        optimize_gain: None,
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

// ── benchmarks ────────────────────────────────────────────────────────────────

fn bench_standard_5band(c: &mut Criterion) {
    let fr = load_fr("blessing3");
    let target = load_target("harman_ie_2019");
    let constraints = standard_constraints();
    c.bench_function(
        "optimize/standard_5band (blessing3 + harman_ie_2019)",
        |b| {
            b.iter(|| {
                optimize(black_box(&fr), black_box(&target), black_box(&constraints)).unwrap()
            })
        },
    );
}

fn bench_qudelix_10band(c: &mut Criterion) {
    let fr = load_fr("blessing3");
    let target = load_target("harman_ie_2019");
    let constraints = qudelix_10_constraints();
    c.bench_function(
        "optimize/qudelix_10band (blessing3 + harman_ie_2019)",
        |b| {
            b.iter(|| {
                optimize(black_box(&fr), black_box(&target), black_box(&constraints)).unwrap()
            })
        },
    );
}

criterion_group!(benches, bench_standard_5band, bench_qudelix_10band);
criterion_main!(benches);
