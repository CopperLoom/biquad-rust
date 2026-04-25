use crate::types::FreqPoint;

const TREBLE_F_LOWER: f64 = 6000.0;
const TREBLE_F_UPPER: f64 = 8000.0;

#[allow(clippy::needless_range_loop)]
fn gauss_jordan_solve(mut aug: Vec<Vec<f64>>) -> Vec<f64> {
    let n = aug.len();
    for col in 0..n {
        let mut max_row = col;
        for row in col + 1..n {
            if aug[row][col].abs() > aug[max_row][col].abs() {
                max_row = row;
            }
        }
        aug.swap(col, max_row);
        let pivot = aug[col][col];
        for j in col..=n {
            aug[col][j] /= pivot;
        }
        for row in 0..n {
            if row != col {
                let factor = aug[row][col];
                for j in col..=n {
                    aug[row][j] -= factor * aug[col][j];
                }
            }
        }
    }
    (0..n).map(|i| aug[i][n]).collect()
}

/// Savitzky-Golay FIR weights for interior smoothing (poly_order=2, deriv=0).
/// Returns coefficients to dot with data[i-m..=i+m] to get smoothed value at i.
pub fn savgol_coeffs(window_size: usize, poly_order: usize) -> Vec<f64> {
    assert!(
        window_size >= 3 && window_size % 2 == 1,
        "window_size must be odd and >= 3"
    );
    assert!(poly_order < window_size);
    let m = (window_size - 1) / 2;
    let np1 = poly_order + 1;

    // Vandermonde A[j][p] = (j - m)^p, centered at 0
    let a: Vec<Vec<f64>> = (0..window_size)
        .map(|j| {
            let x = (j as i64 - m as i64) as f64;
            (0..np1).map(|p| x.powi(p as i32)).collect()
        })
        .collect();

    // A^T A
    let mut ata = vec![vec![0.0f64; np1]; np1];
    #[allow(clippy::needless_range_loop)]
    for k in 0..window_size {
        for i in 0..np1 {
            for j in 0..np1 {
                ata[i][j] += a[k][i] * a[k][j];
            }
        }
    }

    // Solve (A^T A) v = e_0 — first basis vector gives 0th-derivative at center x=0
    let mut aug = vec![vec![0.0f64; np1 + 1]; np1];
    for i in 0..np1 {
        for j in 0..np1 {
            aug[i][j] = ata[i][j];
        }
        aug[i][np1] = if i == 0 { 1.0 } else { 0.0 };
    }
    let v = gauss_jordan_solve(aug);

    // c[j] = A[j] · v
    (0..window_size)
        .map(|j| (0..np1).map(|p| a[j][p] * v[p]).sum::<f64>())
        .collect()
}

/// Fit degree-`poly_order` polynomial to `data` at x = 0..data.len()-1 and sample at `positions`.
fn poly_fit_sample(data: &[f64], poly_order: usize, positions: &[f64]) -> Vec<f64> {
    let w = data.len();
    let np1 = poly_order + 1;

    let a: Vec<Vec<f64>> = (0..w)
        .map(|j| (0..np1).map(|p| (j as f64).powi(p as i32)).collect())
        .collect();

    let mut ata = vec![vec![0.0f64; np1]; np1];
    let mut atb = vec![0.0f64; np1];
    for j in 0..w {
        for i in 0..np1 {
            atb[i] += a[j][i] * data[j];
            for k in 0..np1 {
                ata[i][k] += a[j][i] * a[j][k];
            }
        }
    }

    let mut aug = vec![vec![0.0f64; np1 + 1]; np1];
    for i in 0..np1 {
        for j in 0..np1 {
            aug[i][j] = ata[i][j];
        }
        aug[i][np1] = atb[i];
    }
    let c = gauss_jordan_solve(aug);

    positions
        .iter()
        .map(|&x| (0..np1).map(|p| c[p] * x.powi(p as i32)).sum::<f64>())
        .collect()
}

/// Savitzky-Golay filter, polynomial order 2, scipy mode='interp' edge handling.
pub fn savgol_filter(data: &[f64], window_size: usize) -> Vec<f64> {
    let poly_order = 2usize;
    let m = (window_size - 1) / 2;
    let n = data.len();
    assert!(
        n >= window_size,
        "data length {n} < window_size {window_size}"
    );

    let coeffs = savgol_coeffs(window_size, poly_order);
    let mut out = vec![0.0f64; n];

    // Interior: dot product with precomputed FIR weights
    for i in m..n - m {
        out[i] = (0..window_size).map(|j| coeffs[j] * data[i - m + j]).sum();
    }

    if m > 0 {
        // Left edge (mode='interp'): fit polynomial to first window_size samples,
        // sample at positions 0..m-1
        let left_pos: Vec<f64> = (0..m).map(|i| i as f64).collect();
        let left_vals = poly_fit_sample(&data[..window_size], poly_order, &left_pos);
        out[..m].copy_from_slice(&left_vals);

        // Right edge: fit polynomial to last window_size samples,
        // sample at positions m+1..window_size-1 (local frame of the last w samples)
        let right_pos: Vec<f64> = (m + 1..window_size).map(|i| i as f64).collect();
        let right_vals = poly_fit_sample(&data[n - window_size..], poly_order, &right_pos);
        out[n - m..].copy_from_slice(&right_vals);
    }

    out
}

/// Convert octave smoothing width to an odd sample count for the given frequency grid.
/// Matches AutoEQ's `smoothing_window_size()` exactly (arithmetic mean of ratio steps).
pub fn smoothing_window_size(freqs: &[f64], octaves: f64) -> usize {
    let k = 2.0_f64.powf(octaves);
    let step_size: f64 =
        freqs.windows(2).map(|w| w[1] / w[0]).sum::<f64>() / (freqs.len() - 1) as f64;
    let n = (k.ln() / step_size.ln()).round() as usize;
    let n = if n.is_multiple_of(2) { n + 1 } else { n };
    n.max(3)
}

/// Sigmoid blend weight: 0 below `f_lower`, 1 above `f_upper`, smooth transition between.
/// Matches AutoEQ's `log_f_sigmoid()` with default a_normal=0, a_treble=1.
pub fn log_f_sigmoid(f: f64, f_lower: f64, f_upper: f64) -> f64 {
    let f_center = (f_upper * f_lower).sqrt();
    let half_range = f_upper.log10() - f_center.log10();
    let x = (f.log10() - f_center.log10()) / (half_range / 4.0);
    1.0 / (1.0 + (-x).exp())
}

/// Two-zone smooth: normal window below 6 kHz, treble window above 8 kHz, sigmoid blend.
/// Matches AutoEQ's `_smoothen()`.
pub fn two_zone_smooth(
    fr: &[FreqPoint],
    normal_octaves: f64,
    treble_octaves: f64,
) -> Vec<FreqPoint> {
    let freqs: Vec<f64> = fr.iter().map(|p| p.freq).collect();
    let data: Vec<f64> = fr.iter().map(|p| p.db).collect();

    let w_normal = smoothing_window_size(&freqs, normal_octaves);
    let w_treble = smoothing_window_size(&freqs, treble_octaves);

    let y_normal = savgol_filter(&data, w_normal);
    let y_treble = savgol_filter(&data, w_treble);

    fr.iter()
        .enumerate()
        .map(|(i, p)| {
            let k_treble = log_f_sigmoid(p.freq, TREBLE_F_LOWER, TREBLE_F_UPPER);
            FreqPoint {
                freq: p.freq,
                db: y_normal[i] * (1.0 - k_treble) + y_treble[i] * k_treble,
            }
        })
        .collect()
}

/// Single-zone smooth (used for the re-smooth step in the equalize pipeline).
pub fn smooth(fr: &[FreqPoint], window_octaves: f64) -> Vec<FreqPoint> {
    let freqs: Vec<f64> = fr.iter().map(|p| p.freq).collect();
    let data: Vec<f64> = fr.iter().map(|p| p.db).collect();
    let w = smoothing_window_size(&freqs, window_octaves);
    let smoothed = savgol_filter(&data, w);
    fr.iter()
        .zip(smoothed)
        .map(|(p, db)| FreqPoint { freq: p.freq, db })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interpolate::build_grid;
    use approx::assert_abs_diff_eq;

    #[test]
    fn savgol_preserves_quadratic() {
        // SG filter with poly_order=2 must reproduce any degree-2 polynomial exactly,
        // including at edge samples (mode='interp').
        let n = 50usize;
        let data: Vec<f64> = (0..n)
            .map(|i| 3.0 + 0.5 * i as f64 - 0.02 * (i * i) as f64)
            .collect();
        let out = savgol_filter(&data, 7);
        for i in 0..n {
            assert_abs_diff_eq!(out[i], data[i], epsilon = 1e-9);
        }
    }

    #[test]
    fn savgol_preserves_quadratic_large_window() {
        let n = 200usize;
        let data: Vec<f64> = (0..n).map(|i| 1.0 - 0.001 * (i * i) as f64).collect();
        let out = savgol_filter(&data, 21);
        for i in 0..n {
            assert_abs_diff_eq!(out[i], data[i], epsilon = 1e-8);
        }
    }

    #[test]
    fn smoothing_window_size_1_01_grid() {
        let freqs = build_grid(20.0, 20000.0, 1.01);
        // 1/12 octave: round(ln(2^(1/12)) / ln(1.01)) = round(5.82) = 6 → +1 for even → 7
        assert_eq!(smoothing_window_size(&freqs, 1.0 / 12.0), 7);
        // 2 octaves: round(ln(4) / ln(1.01)) = round(139.3) = 139 (already odd)
        assert_eq!(smoothing_window_size(&freqs, 2.0), 139);
    }

    #[test]
    fn sigmoid_approaches_zero_below_transition() {
        assert!(log_f_sigmoid(5000.0, TREBLE_F_LOWER, TREBLE_F_UPPER) < 0.01);
    }

    #[test]
    fn sigmoid_approaches_one_above_transition() {
        assert!(log_f_sigmoid(10000.0, TREBLE_F_LOWER, TREBLE_F_UPPER) > 0.99);
    }

    #[test]
    fn sigmoid_is_half_at_geometric_mean() {
        // Geometric mean is the sigmoid's inflection point → exactly 0.5
        let f_center = (TREBLE_F_LOWER * TREBLE_F_UPPER).sqrt();
        assert_abs_diff_eq!(
            log_f_sigmoid(f_center, TREBLE_F_LOWER, TREBLE_F_UPPER),
            0.5,
            epsilon = 1e-10
        );
    }

    #[test]
    fn two_zone_blend_is_between_zones_at_7khz() {
        let freqs = build_grid(20.0, 20000.0, 1.01);
        let data: Vec<f64> = freqs
            .iter()
            .enumerate()
            .map(|(i, _)| (i as f64 * 0.05).sin() * 5.0)
            .collect();
        let fr: Vec<FreqPoint> = freqs
            .iter()
            .zip(&data)
            .map(|(&f, &db)| FreqPoint { freq: f, db })
            .collect();

        let smoothed = two_zone_smooth(&fr, 1.0 / 12.0, 2.0);

        let w_normal = smoothing_window_size(&freqs, 1.0 / 12.0);
        let w_treble = smoothing_window_size(&freqs, 2.0);
        let y_normal = savgol_filter(&data, w_normal);
        let y_treble = savgol_filter(&data, w_treble);

        let idx = freqs.iter().position(|&f| f >= 7000.0).unwrap();
        let blended = smoothed[idx].db;
        let lo = y_normal[idx].min(y_treble[idx]) - 1e-10;
        let hi = y_normal[idx].max(y_treble[idx]) + 1e-10;
        assert!(
            blended >= lo && blended <= hi,
            "blend {blended:.4} not in [{lo:.4}, {hi:.4}]"
        );
    }
}
