use crate::types::{FreqPoint, InterpolateOptions};

const DEFAULT_STEP: f64 = 1.01;
const DEFAULT_F_MIN: f64 = 20.0;
const DEFAULT_F_MAX: f64 = 20000.0;

/// Build a log-spaced frequency grid (multiplicative steps).
pub fn build_grid(f_min: f64, f_max: f64, step: f64) -> Vec<f64> {
    let mut freqs = Vec::new();
    let mut f = f_min;
    while f <= f_max {
        freqs.push(f);
        f *= step;
    }
    freqs
}

/// Resample a frequency response to a log-spaced grid using log-linear interpolation.
/// Clamps to boundary values outside the measured range (no extrapolation).
pub fn interpolate(fr: &[FreqPoint], options: &InterpolateOptions) -> Vec<FreqPoint> {
    let step = options.step.unwrap_or(DEFAULT_STEP);
    let f_min = options.f_min.unwrap_or(DEFAULT_F_MIN);
    let f_max = options.f_max.unwrap_or(DEFAULT_F_MAX);

    let log_freqs: Vec<f64> = fr.iter().map(|pt| pt.freq.ln()).collect();
    let dbs: Vec<f64> = fr.iter().map(|pt| pt.db).collect();
    let n = log_freqs.len();

    build_grid(f_min, f_max, step)
        .into_iter()
        .map(|freq| {
            let log_f = freq.ln();

            if log_f <= log_freqs[0] {
                return FreqPoint { freq, db: dbs[0] };
            }
            if log_f >= log_freqs[n - 1] {
                return FreqPoint { freq, db: dbs[n - 1] };
            }

            // Binary search for surrounding pair
            let mut lo = 0;
            let mut hi = n - 1;
            while hi - lo > 1 {
                let mid = (lo + hi) / 2;
                if log_freqs[mid] <= log_f {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }

            let t = (log_f - log_freqs[lo]) / (log_freqs[hi] - log_freqs[lo]);
            let db = dbs[lo] + t * (dbs[hi] - dbs[lo]);
            FreqPoint { freq, db }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn grid_1_01_has_695_points() {
        let grid = build_grid(20.0, 20000.0, 1.01);
        assert_eq!(grid.len(), 695);
    }

    #[test]
    fn grid_1_02_has_correct_point_count() {
        let grid = build_grid(20.0, 20000.0, 1.02);
        // log(1000) / log(1.02) ≈ 348.9 → 349 or 350 depending on rounding
        assert!(grid.len() >= 348 && grid.len() <= 351, "got {}", grid.len());
    }

    #[test]
    fn grid_starts_and_ends_at_bounds() {
        let grid = build_grid(20.0, 20000.0, 1.01);
        assert_abs_diff_eq!(grid[0], 20.0, epsilon = 1e-9);
        assert!(*grid.last().unwrap() <= 20000.0 + 1e-9);
    }

    #[test]
    fn interpolate_at_original_points_is_exact() {
        let fr = vec![
            FreqPoint { freq: 100.0, db: 0.0 },
            FreqPoint { freq: 1000.0, db: 3.0 },
            FreqPoint { freq: 10000.0, db: -2.0 },
        ];
        let opts = InterpolateOptions { step: Some(1.01), f_min: Some(100.0), f_max: Some(100.0) };
        let result = interpolate(&fr, &opts);
        assert_eq!(result.len(), 1);
        assert_abs_diff_eq!(result[0].db, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn interpolate_clamps_below() {
        let fr = vec![
            FreqPoint { freq: 100.0, db: 5.0 },
            FreqPoint { freq: 1000.0, db: 0.0 },
        ];
        let opts = InterpolateOptions { step: Some(1.01), f_min: Some(20.0), f_max: Some(20.0) };
        let result = interpolate(&fr, &opts);
        assert_abs_diff_eq!(result[0].db, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn interpolate_clamps_above() {
        let fr = vec![
            FreqPoint { freq: 100.0, db: 0.0 },
            FreqPoint { freq: 1000.0, db: 5.0 },
        ];
        let opts = InterpolateOptions { step: Some(1.01), f_min: Some(5000.0), f_max: Some(5000.0) };
        let result = interpolate(&fr, &opts);
        assert_abs_diff_eq!(result[0].db, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn interpolate_midpoint_is_linear_in_log_freq() {
        // Two points at log10(100)=2 and log10(10000)=4; midpoint log-freq is log10(1000)=3
        let fr = vec![
            FreqPoint { freq: 100.0, db: 0.0 },
            FreqPoint { freq: 10000.0, db: 4.0 },
        ];
        // 1000 Hz is the geometric midpoint of 100–10000
        let opts = InterpolateOptions { step: Some(1.01), f_min: Some(1000.0), f_max: Some(1000.0) };
        let result = interpolate(&fr, &opts);
        // Expect exactly 2.0 dB (midpoint)
        assert_abs_diff_eq!(result[0].db, 2.0, epsilon = 1e-10);
    }
}
