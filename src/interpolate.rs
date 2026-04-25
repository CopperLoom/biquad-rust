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
/// Extrapolates linearly (in log-frequency space) outside the measured range, matching
/// AutoEQ's `InterpolatedUnivariateSpline(log_f, data, k=1)` default behavior.
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

            // Extrapolate using slope of first/last segment when outside the input range.
            let (lo, hi) = if log_f <= log_freqs[0] {
                (0usize, 1usize.min(n - 1))
            } else if log_f >= log_freqs[n - 1] {
                (n - 2, n - 1)
            } else {
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
                (lo, hi)
            };

            let span = log_freqs[hi] - log_freqs[lo];
            let t = if span == 0.0 { 0.0 } else { (log_f - log_freqs[lo]) / span };
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
    fn interpolate_extrapolates_below() {
        // Slope (5 → 0 dB) over (log 100 → log 1000) = -5/log(10) per nat.
        // At 20 Hz: db = 5 + (ln 20 - ln 100) / (ln 1000 - ln 100) * (0 - 5)
        let fr = vec![
            FreqPoint { freq: 100.0, db: 5.0 },
            FreqPoint { freq: 1000.0, db: 0.0 },
        ];
        let opts = InterpolateOptions { step: Some(1.01), f_min: Some(20.0), f_max: Some(20.0) };
        let result = interpolate(&fr, &opts);
        let expected = 5.0 + (20f64.ln() - 100f64.ln()) / (1000f64.ln() - 100f64.ln()) * (0.0 - 5.0);
        assert_abs_diff_eq!(result[0].db, expected, epsilon = 1e-10);
    }

    #[test]
    fn interpolate_extrapolates_above() {
        let fr = vec![
            FreqPoint { freq: 100.0, db: 0.0 },
            FreqPoint { freq: 1000.0, db: 5.0 },
        ];
        let opts = InterpolateOptions { step: Some(1.01), f_min: Some(5000.0), f_max: Some(5000.0) };
        let result = interpolate(&fr, &opts);
        let expected = 0.0 + (5000f64.ln() - 100f64.ln()) / (1000f64.ln() - 100f64.ln()) * (5.0 - 0.0);
        assert_abs_diff_eq!(result[0].db, expected, epsilon = 1e-10);
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
