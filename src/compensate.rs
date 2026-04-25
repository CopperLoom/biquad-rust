use crate::types::FreqPoint;

/// Log-linear interpolation with linear extrapolation beyond the source range.
/// Matches AutoEQ's `InterpolatedUnivariateSpline(k=1)` on log10(freq) axis.
fn log_linear_resample(source: &[FreqPoint], target_freqs: &[f64]) -> Vec<f64> {
    assert!(source.len() >= 2);
    let log_src: Vec<f64> = source.iter().map(|p| p.freq.ln()).collect();
    let dbs: Vec<f64> = source.iter().map(|p| p.db).collect();
    let n = source.len();

    target_freqs
        .iter()
        .map(|&f| {
            let lf = f.ln();
            let idx = log_src.partition_point(|&x| x < lf);
            if idx == 0 {
                // Extrapolate left from first two points
                let t = (lf - log_src[0]) / (log_src[1] - log_src[0]);
                dbs[0] + t * (dbs[1] - dbs[0])
            } else if idx >= n {
                // Extrapolate right from last two points
                let t = (lf - log_src[n - 2]) / (log_src[n - 1] - log_src[n - 2]);
                dbs[n - 2] + t * (dbs[n - 1] - dbs[n - 2])
            } else {
                let lo = idx - 1;
                let t = (lf - log_src[lo]) / (log_src[idx] - log_src[lo]);
                dbs[lo] + t * (dbs[idx] - dbs[lo])
            }
        })
        .collect()
}

/// Log-linear interpolation at a single query frequency (with extrapolation).
fn log_linear_at(freqs: &[f64], dbs: &[f64], query: f64) -> f64 {
    assert!(freqs.len() >= 2);
    let n = freqs.len();
    let lq = query.ln();
    let log_freqs: Vec<f64> = freqs.iter().map(|f| f.ln()).collect();
    let idx = log_freqs.partition_point(|&x| x < lq);
    if idx == 0 {
        let t = (lq - log_freqs[0]) / (log_freqs[1] - log_freqs[0]);
        dbs[0] + t * (dbs[1] - dbs[0])
    } else if idx >= n {
        let t = (lq - log_freqs[n - 2]) / (log_freqs[n - 1] - log_freqs[n - 2]);
        dbs[n - 2] + t * (dbs[n - 1] - dbs[n - 2])
    } else {
        let lo = idx - 1;
        let t = (lq - log_freqs[lo]) / (log_freqs[idx] - log_freqs[lo]);
        dbs[lo] + t * (dbs[idx] - dbs[lo])
    }
}

/// Subtract the log-linearly interpolated dB value at 1 kHz from all points.
/// Matches AutoEQ's `FrequencyResponse.center(frequency=1000)`.
/// 1 kHz is off the 1.01 grid, so it must be interpolated, not indexed.
pub fn center(fr: &[FreqPoint]) -> Vec<FreqPoint> {
    let freqs: Vec<f64> = fr.iter().map(|p| p.freq).collect();
    let dbs: Vec<f64> = fr.iter().map(|p| p.db).collect();
    let offset = log_linear_at(&freqs, &dbs, 1000.0);
    fr.iter()
        .map(|p| FreqPoint {
            freq: p.freq,
            db: p.db - offset,
        })
        .collect()
}

/// Compute error curve: error = measured − target.
///
/// Four-step pipeline matching AutoEQ's `compensate()`:
/// 1. Interpolate target onto measured frequency grid (log-linear, extrapolates)
/// 2. Center target at 1 kHz (log-linear interpolation — 1 kHz is off the 1.01 grid)
/// 3. Add `create_target()` contributions (zero for all default parameters)
/// 4. error[i] = measured[i] − target[i]
pub fn compensate(measured: &[FreqPoint], target: &[FreqPoint]) -> Vec<FreqPoint> {
    let measured_freqs: Vec<f64> = measured.iter().map(|p| p.freq).collect();

    // Step 1: interpolate target onto measured grid with linear extrapolation
    let target_dbs = log_linear_resample(target, &measured_freqs);

    // Step 2: center target at 1 kHz via log-linear interpolation (1 kHz is off the 1.01 grid)
    let center_val = log_linear_at(&measured_freqs, &target_dbs, 1000.0);
    let target_centered: Vec<f64> = target_dbs.iter().map(|db| db - center_val).collect();

    // Step 3: create_target() with default parameters contributes all zeros
    // (bass_boost_gain=0, treble_boost_gain=0, tilt=0.0)

    // Step 4: error = measured − target
    measured
        .iter()
        .zip(target_centered.iter())
        .map(|(m, t)| FreqPoint {
            freq: m.freq,
            db: m.db - t,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpolate::build_grid;
    use approx::assert_abs_diff_eq;

    fn canonical_grid() -> Vec<f64> {
        build_grid(20.0, 20000.0, 1.01)
    }

    #[test]
    fn identical_inputs_give_zero_error() {
        let freqs = canonical_grid();
        let fr: Vec<FreqPoint> = freqs
            .iter()
            .map(|&f| FreqPoint { freq: f, db: 0.0 })
            .collect();
        let error = compensate(&fr, &fr);
        for pt in &error {
            assert_abs_diff_eq!(pt.db, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn flat_target_centered_to_zero_gives_measured_as_error() {
        // Flat target at any constant → centers to 0 dB everywhere → error = measured
        let freqs = canonical_grid();
        let measured: Vec<FreqPoint> = freqs
            .iter()
            .enumerate()
            .map(|(i, &f)| FreqPoint {
                freq: f,
                db: i as f64 * 0.01,
            })
            .collect();
        let target: Vec<FreqPoint> = freqs
            .iter()
            .map(|&f| FreqPoint { freq: f, db: 3.0 })
            .collect();
        let error = compensate(&measured, &target);
        for (e, m) in error.iter().zip(&measured) {
            assert_abs_diff_eq!(e.db, m.db, epsilon = 1e-10);
        }
    }

    #[test]
    fn centering_uses_log_linear_interpolation() {
        // Target = log10(f): after centering, target[i] = log10(f/1000).
        // error = 0 - log10(f/1000) = log10(1000/f)
        let freqs = canonical_grid();
        let measured: Vec<FreqPoint> = freqs
            .iter()
            .map(|&f| FreqPoint { freq: f, db: 0.0 })
            .collect();
        let target: Vec<FreqPoint> = freqs
            .iter()
            .map(|&f| FreqPoint {
                freq: f,
                db: f.log10(),
            })
            .collect();
        let error = compensate(&measured, &target);
        for pt in &error {
            let expected = -(pt.freq.log10() - 3.0);
            assert_abs_diff_eq!(pt.db, expected, epsilon = 1e-10);
        }
    }

    #[test]
    fn target_extrapolates_below_range() {
        // Target starting at 100 Hz with a known slope; query at 20 Hz should extrapolate,
        // not clamp to the 100 Hz boundary value.
        let target = vec![
            FreqPoint {
                freq: 100.0,
                db: 4.0,
            },
            FreqPoint {
                freq: 1000.0,
                db: 2.0,
            },
            FreqPoint {
                freq: 10000.0,
                db: 0.0,
            },
        ];
        let measured = vec![
            FreqPoint {
                freq: 20.0,
                db: 0.0,
            },
            FreqPoint {
                freq: 1000.0,
                db: 0.0,
            },
            FreqPoint {
                freq: 10000.0,
                db: 0.0,
            },
        ];
        let error = compensate(&measured, &target);

        // Clamped value at 20 Hz would be 4.0 (boundary). Extrapolated extends the
        // 100–1000 Hz slope leftward, giving a value > 4.0. After centering (subtract
        // ~2.0 at 1 kHz) the 20 Hz target > 2.0, so error < -2.0.
        // With clamping, target at 20 Hz = 4.0, centered ≈ 2.0, error ≈ -2.0.
        assert!(
            error[0].db < -2.0,
            "expected extrapolation (< -2.0), got {}",
            error[0].db
        );
    }
}
