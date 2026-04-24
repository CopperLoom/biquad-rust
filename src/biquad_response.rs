use crate::types::FilterType;

pub struct BiquadCoeffs {
    pub b0: f64,
    pub b1: f64,
    pub b2: f64,
    pub a1: f64,
    pub a2: f64,
}

pub fn biquad_coeffs(filter_type: FilterType, fc: f64, gain: f64, q: f64, fs: f64) -> BiquadCoeffs {
    let a = 10_f64.powf(gain / 40.0);
    let sqrt_a = a.sqrt();
    let w0 = 2.0 * std::f64::consts::PI * fc / fs;
    let sin_w0 = w0.sin();
    let cos_w0 = w0.cos();
    let alpha = sin_w0 / (2.0 * q);

    let (b0, b1, b2, a1, a2) = match filter_type {
        FilterType::PK => {
            let a0 = 1.0 + alpha / a;
            (
                (1.0 + alpha * a) / a0,
                (-2.0 * cos_w0) / a0,
                (1.0 - alpha * a) / a0,
                (-2.0 * cos_w0) / a0,
                (1.0 - alpha / a) / a0,
            )
        }
        FilterType::LSQ => {
            let a0 = (a + 1.0) + (a - 1.0) * cos_w0 + 2.0 * sqrt_a * alpha;
            (
                a * ((a + 1.0) - (a - 1.0) * cos_w0 + 2.0 * sqrt_a * alpha) / a0,
                2.0 * a * ((a - 1.0) - (a + 1.0) * cos_w0) / a0,
                a * ((a + 1.0) - (a - 1.0) * cos_w0 - 2.0 * sqrt_a * alpha) / a0,
                -2.0 * ((a - 1.0) + (a + 1.0) * cos_w0) / a0,
                ((a + 1.0) + (a - 1.0) * cos_w0 - 2.0 * sqrt_a * alpha) / a0,
            )
        }
        FilterType::HSQ => {
            let a0 = (a + 1.0) - (a - 1.0) * cos_w0 + 2.0 * sqrt_a * alpha;
            (
                a * ((a + 1.0) + (a - 1.0) * cos_w0 + 2.0 * sqrt_a * alpha) / a0,
                -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0) / a0,
                a * ((a + 1.0) + (a - 1.0) * cos_w0 - 2.0 * sqrt_a * alpha) / a0,
                2.0 * ((a - 1.0) - (a + 1.0) * cos_w0) / a0,
                ((a + 1.0) - (a - 1.0) * cos_w0 - 2.0 * sqrt_a * alpha) / a0,
            )
        }
    };

    BiquadCoeffs { b0, b1, b2, a1, a2 }
}

/// Evaluate filter magnitude in dB at a single frequency using the phi identity.
pub fn eval_magnitude(c: &BiquadCoeffs, freq: f64, fs: f64) -> f64 {
    let w = 2.0 * std::f64::consts::PI * freq / fs;
    let phi = 4.0 * (w / 2.0).sin().powi(2);

    let num = (c.b0 + c.b1 + c.b2).powi(2)
        + (c.b0 * c.b2 * phi - (c.b1 * (c.b0 + c.b2) + 4.0 * c.b0 * c.b2)) * phi;
    let den = (1.0 + c.a1 + c.a2).powi(2)
        + (c.a2 * phi - (c.a1 * (1.0 + c.a2) + 4.0 * c.a2)) * phi;

    10.0 * num.max(1e-30).log10() - 10.0 * den.max(1e-30).log10()
}

pub fn biquad_response(
    filter_type: FilterType,
    fc: f64,
    gain: f64,
    q: f64,
    frequencies: &[f64],
    fs: f64,
) -> Vec<f64> {
    let coeffs = biquad_coeffs(filter_type, fc, gain, q, fs);
    frequencies.iter().map(|&f| eval_magnitude(&coeffs, f, fs)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    const FS: f64 = 44100.0;

    #[test]
    fn pk_at_center_freq_equals_gain() {
        let fc = 1000.0;
        let gain = 6.0;
        let q = 1.0;
        let resp = biquad_response(FilterType::PK, fc, gain, q, &[fc], FS);
        assert_abs_diff_eq!(resp[0], gain, epsilon = 1e-6);
    }

    #[test]
    fn pk_far_from_center_approaches_zero() {
        let resp = biquad_response(FilterType::PK, 1000.0, 6.0, 1.0, &[20.0, 20000.0], FS);
        assert_abs_diff_eq!(resp[0], 0.0, epsilon = 0.01);
        assert_abs_diff_eq!(resp[1], 0.0, epsilon = 0.01);
    }

    #[test]
    fn pk_zero_gain_is_flat() {
        let freqs: Vec<f64> = (0..50).map(|i| 20.0 * 1.1_f64.powi(i)).collect();
        let resp = biquad_response(FilterType::PK, 1000.0, 0.0, 1.0, &freqs, FS);
        for db in &resp {
            assert_abs_diff_eq!(*db, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn lsq_zero_gain_is_flat() {
        let freqs: Vec<f64> = (0..50).map(|i| 20.0 * 1.1_f64.powi(i)).collect();
        let resp = biquad_response(FilterType::LSQ, 200.0, 0.0, 0.7, &freqs, FS);
        for db in &resp {
            assert_abs_diff_eq!(*db, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn hsq_zero_gain_is_flat() {
        let freqs: Vec<f64> = (0..50).map(|i| 20.0 * 1.1_f64.powi(i)).collect();
        let resp = biquad_response(FilterType::HSQ, 8000.0, 0.0, 0.7, &freqs, FS);
        for db in &resp {
            assert_abs_diff_eq!(*db, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn lsq_gain_at_dc_approaches_shelf_gain() {
        // At very low frequency, a low shelf should approach its gain value
        let gain = 6.0;
        let resp = biquad_response(FilterType::LSQ, 100.0, gain, 0.7, &[1.0], FS);
        // At f→0, low shelf response → gain dB
        assert_abs_diff_eq!(resp[0], gain, epsilon = 0.05);
    }

    #[test]
    fn hsq_gain_at_nyquist_approaches_shelf_gain() {
        // At very high frequency, a high shelf should approach its gain value
        let gain = 4.0;
        // Use 19000 Hz as proxy for high-frequency plateau
        let resp = biquad_response(FilterType::HSQ, 1000.0, gain, 0.7, &[19000.0], FS);
        assert_abs_diff_eq!(resp[0], gain, epsilon = 0.1);
    }

    /// Reference values traced from AutoEQ peq.py biquad_coefficients() + fr() phi identity.
    #[test]
    fn pk_matches_autoeq_reference() {
        let resp = biquad_response(FilterType::PK, 1000.0, 3.0, 2.0, &[500.0, 2000.0], FS);
        assert_abs_diff_eq!(resp[0], 0.30314119955309593, epsilon = 1e-6);
        assert_abs_diff_eq!(resp[1], 0.29969969796010787, epsilon = 1e-6);
    }

    #[test]
    fn lsq_matches_autoeq_reference() {
        let resp = biquad_response(FilterType::LSQ, 200.0, 6.0, 0.7, &[50.0, 1000.0], FS);
        assert_abs_diff_eq!(resp[0], 5.967104022630323, epsilon = 1e-6);
        assert_abs_diff_eq!(resp[1], 0.015237787440533168, epsilon = 1e-6);
    }

    #[test]
    fn hsq_matches_autoeq_reference() {
        let resp = biquad_response(FilterType::HSQ, 8000.0, -4.0, 0.7, &[1000.0, 16000.0], FS);
        assert_abs_diff_eq!(resp[0], -0.0016569919195106309, epsilon = 1e-6);
        assert_abs_diff_eq!(resp[1], -3.9620696151614445, epsilon = 1e-6);
    }
}
