mod helpers;

use biquad_rust::types::FilterType;

#[test]
fn filter_type_serde_roundtrip() {
    let types = [FilterType::PK, FilterType::LSQ, FilterType::HSQ];
    for ft in types {
        let s = serde_json::to_string(&ft).unwrap();
        let back: FilterType = serde_json::from_str(&s).unwrap();
        assert_eq!(ft, back);
    }
    assert_eq!(serde_json::to_string(&FilterType::PK).unwrap(), "\"PK\"");
    assert_eq!(serde_json::to_string(&FilterType::LSQ).unwrap(), "\"LSQ\"");
    assert_eq!(serde_json::to_string(&FilterType::HSQ).unwrap(), "\"HSQ\"");
}

#[test]
fn golden_files_parse() {
    let files = [
        "blessing3__harman_ie_2019__standard.json",
        "blessing3__harman_ie_2019__restricted.json",
        "blessing3__harman_ie_2019__qudelix_10.json",
    ];
    for name in files {
        let golden = helpers::load_golden(name);
        assert!(!golden.filters.is_empty(), "{name}: no filters");
        assert!(golden.fs > 0.0, "{name}: bad fs");
    }
}

#[test]
fn all_90_golden_files_parse() {
    let iems = ["blessing3", "hexa", "andromeda", "zero2", "origin_s"];
    let targets = [
        "harman_ie_2019",
        "diffuse_field",
        "flat",
        "v_shaped",
        "bass_heavy",
        "bright",
    ];
    let constraints = ["standard", "restricted", "qudelix_10"];
    let mut count = 0;
    for iem in iems {
        for target in targets {
            for constraint in constraints {
                let name = format!("{iem}__{target}__{constraint}.json");
                let golden = helpers::load_golden(&name);
                assert_eq!(golden.iem, iem);
                assert_eq!(golden.target, target);
                assert_eq!(golden.constraint, constraint);
                count += 1;
            }
        }
    }
    assert_eq!(count, 90);
}

#[test]
fn rmse_helper_known_values() {
    let a = [1.0, 2.0, 3.0];
    let b = [1.0, 2.0, 3.0];
    approx::assert_abs_diff_eq!(helpers::rmse(&a, &b), 0.0, epsilon = 1e-12);

    let c = [0.0, 0.0];
    let d = [1.0, 1.0];
    approx::assert_abs_diff_eq!(helpers::rmse(&c, &d), 1.0, epsilon = 1e-12);
}

#[test]
fn optimizer_grid_bounds() {
    let grid = helpers::optimizer_grid();
    assert_eq!(*grid.first().unwrap(), 20.0);
    assert!(*grid.last().unwrap() <= 20000.0);
    assert_eq!(grid.len(), 349);
}
