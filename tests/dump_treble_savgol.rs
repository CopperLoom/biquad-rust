/// Dump our 2-octave savgol pass over the compensate-error, no two-zone blend, no equalize.
/// Run: cargo test --test dump_treble_savgol -- --nocapture
mod helpers;

use biquad_rust::smooth::{savgol_filter, smoothing_window_size};
use biquad_rust::types::InterpolateOptions;
use biquad_rust::{center, compensate, interpolate};
use helpers::{load_fr, load_target};
use std::fs;

const CASES: &[(&str, &str)] = &[
    ("hexa", "bass_heavy"),
    ("origin_s", "bright"),
    ("origin_s", "flat"),
    ("zero2", "bass_heavy"),
];

#[test]
fn dump_treble_savgol() {
    let debug_dir = "tests/fixtures/debug";
    fs::create_dir_all(debug_dir).ok();

    for &(iem, target_name) in CASES {
        let measured = load_fr(iem);
        let target = load_target(target_name);

        let opts = InterpolateOptions { step: Some(1.01), f_min: None, f_max: None };
        let interp = interpolate(&measured, &opts);
        let interp_c = center(&interp);
        let error = compensate(&interp_c, &target);

        let freqs: Vec<f64> = error.iter().map(|p| p.freq).collect();
        let err_db: Vec<f64> = error.iter().map(|p| p.db).collect();

        let win_normal = smoothing_window_size(&freqs, 1.0 / 12.0);
        let win_treble = smoothing_window_size(&freqs, 2.0);

        let y_normal = savgol_filter(&err_db, win_normal);
        let y_treble = savgol_filter(&err_db, win_treble);

        println!(
            "{iem}__{target_name}: n={} win_normal={} win_treble={}",
            err_db.len(),
            win_normal,
            win_treble
        );

        for (suffix, arr) in [
            ("error_in", &err_db),
            ("savgol_normal", &y_normal),
            ("savgol_treble", &y_treble),
        ] {
            let pts: Vec<String> = freqs
                .iter()
                .zip(arr.iter())
                .map(|(f, d)| format!("  {{\"freq\": {f:.6}, \"db\": {d:.6}}}"))
                .collect();
            let json = format!("[\n{}\n]", pts.join(",\n"));
            let path = format!(
                "{debug_dir}/{iem}__{target_name}__{suffix}_rust.json"
            );
            fs::write(&path, &json).expect("write");
            println!("  wrote {path}");
        }
    }
}
