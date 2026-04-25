use crate::biquad_response::{biquad_coeffs, eval_magnitude};
use crate::types::{Filter, FreqPoint};

pub fn apply_filters(fr: &[FreqPoint], filters: &[Filter], pregain: f64, fs: f64) -> Vec<FreqPoint> {
    fr.iter()
        .map(|p| {
            let correction: f64 = filters.iter().map(|f| {
                let c = biquad_coeffs(f.filter_type, f.fc, f.gain, f.q, fs);
                eval_magnitude(&c, p.freq, fs)
            }).sum();
            FreqPoint { freq: p.freq, db: p.db + pregain + correction }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::FilterType;
    use approx::assert_abs_diff_eq;

    const FS: f64 = 44100.0;

    #[test]
    fn no_filters_applies_pregain_only() {
        let fr = vec![FreqPoint { freq: 1000.0, db: 0.0 }, FreqPoint { freq: 4000.0, db: -3.0 }];
        let result = apply_filters(&fr, &[], -6.0, FS);
        assert_abs_diff_eq!(result[0].db, -6.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[1].db, -9.0, epsilon = 1e-10);
    }

    #[test]
    fn zero_gain_filter_acts_as_pregain_only() {
        let fr = vec![FreqPoint { freq: 1000.0, db: 0.0 }];
        let filters = vec![Filter { filter_type: FilterType::PK, fc: 1000.0, gain: 0.0, q: 1.0 }];
        let result = apply_filters(&fr, &filters, 0.0, FS);
        assert_abs_diff_eq!(result[0].db, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn pk_filter_at_center_freq() {
        // A PK filter at fc should add exactly `gain` dB at that frequency.
        let fr = vec![FreqPoint { freq: 1000.0, db: 0.0 }];
        let filters = vec![Filter { filter_type: FilterType::PK, fc: 1000.0, gain: 3.0, q: 1.0 }];
        let result = apply_filters(&fr, &filters, 0.0, FS);
        assert_abs_diff_eq!(result[0].db, 3.0, epsilon = 1e-6);
    }

    #[test]
    fn multiple_filters_and_pregain_sum() {
        let fr = vec![FreqPoint { freq: 1000.0, db: 0.0 }];
        let filters = vec![
            Filter { filter_type: FilterType::PK, fc: 1000.0, gain: 2.0, q: 1.0 },
            Filter { filter_type: FilterType::PK, fc: 1000.0, gain: 1.0, q: 1.0 },
        ];
        let result = apply_filters(&fr, &filters, -1.0, FS);
        // 0 + (-1) + 2 + 1 = 2.0 at 1 kHz (both PK filters at their center)
        assert_abs_diff_eq!(result[0].db, 2.0, epsilon = 1e-6);
    }
}
