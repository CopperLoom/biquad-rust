use crate::smooth::{log_f_sigmoid, two_zone_smooth};
use crate::types::FreqPoint;

const MAX_GAIN: f64 = 6.0;
const MAX_SLOPE: f64 = 18.0;
const TREBLE_F_LOWER: f64 = 6000.0;
const TREBLE_F_UPPER: f64 = 8000.0;

fn log_log_gradient(x1: f64, x0: f64, y1: f64, y0: f64) -> f64 {
    (y1 - y0) / (x1 / x0).log2()
}

fn local_maxima(y: &[f64]) -> Vec<usize> {
    let n = y.len();
    if n < 3 {
        return vec![];
    }
    let mut peaks = vec![];
    let mut i = 1usize;
    while i < n - 1 {
        if y[i] > y[i - 1] {
            // Walk plateau to find end
            let mut j = i + 1;
            while j < n && y[j] == y[i] {
                j += 1;
            }
            if j == n || y[j] < y[i] {
                peaks.push(i); // left edge of plateau or sharp peak
                i = j;
                continue;
            }
        }
        i += 1;
    }
    peaks
}

fn prominence(y: &[f64], peak: usize) -> f64 {
    let h = y[peak];

    // Scan left: continue while y[i] <= h (equal-height does NOT terminate)
    let mut left_min = h;
    if peak > 0 {
        let mut i = peak - 1;
        loop {
            if y[i] > h {
                break;
            }
            if y[i] < left_min {
                left_min = y[i];
            }
            if i == 0 {
                break;
            }
            i -= 1;
        }
    }

    // Scan right: continue while y[i] <= h
    let mut right_min = h;
    for i in (peak + 1)..y.len() {
        if y[i] > h {
            break;
        }
        if y[i] < right_min {
            right_min = y[i];
        }
    }

    h - left_min.max(right_min)
}

/// Port of scipy.signal.find_peaks with prominence filter.
/// Returns indices of local maxima with prominence >= min_prominence.
pub fn find_peaks(y: &[f64], min_prominence: f64) -> Vec<usize> {
    let candidates = local_maxima(y);
    if min_prominence <= 0.0 {
        return candidates;
    }
    candidates
        .into_iter()
        .filter(|&p| prominence(y, p) >= min_prominence)
        .collect()
}

/// Zones around interior dips lower than neighbors → limit-free.
/// Returns zero mask when synthesized dip count < 3. [Gotcha H]
pub fn protection_mask(y: &[f64], peak_inds: &[usize], dip_inds: &[usize]) -> Vec<bool> {
    let n = y.len();

    let (synth_inds, dip_levels) = if !peak_inds.is_empty()
        && (dip_inds.is_empty()
            || peak_inds[peak_inds.len() - 1] > dip_inds[dip_inds.len() - 1])
    {
        // Last peak after last dip: append real argmin after last peak
        let last_peak = peak_inds[peak_inds.len() - 1];
        let min_ix = y[last_peak..]
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i + last_peak)
            .unwrap_or(last_peak);
        let mut inds = dip_inds.to_vec();
        inds.push(min_ix);
        let levels: Vec<f64> = inds.iter().map(|&i| y[i]).collect();
        (inds, levels)
    } else {
        // Sentinel: level = global minimum (sentinel index never dereferenced in loop)
        let global_min = y.iter().copied().fold(f64::INFINITY, f64::min);
        let mut inds = dip_inds.to_vec();
        let mut levels: Vec<f64> = dip_inds.iter().map(|&i| y[i]).collect();
        inds.push(0); // sentinel index
        levels.push(global_min);
        (inds, levels)
    };

    let mut mask = vec![false; n];
    if synth_inds.len() < 3 {
        return mask; // [Gotcha H]
    }

    for i in 1..synth_inds.len() - 1 {
        let dip_ind = synth_inds[i];
        let target_left = dip_levels[i - 1];
        let target_right = dip_levels[i + 1];

        // Last idx in y[..dip_ind] where y >= target_left, then +1
        let left_ind = y[..dip_ind]
            .iter()
            .enumerate()
            .filter(|&(_, v)| *v >= target_left)
            .map(|(j, _)| j)
            .last()
            .map(|j| j + 1)
            .unwrap_or(0);

        // First idx in y[dip_ind..] where y >= target_right, offset by dip_ind - 1
        let right_ind = y[dip_ind..]
            .iter()
            .enumerate()
            .find(|&(_, v)| *v >= target_right)
            .map(|(j, _)| j + dip_ind.saturating_sub(1))
            .unwrap_or(0);

        for j in left_ind..=right_ind {
            if j < n {
                mask[j] = true;
            }
        }
    }

    mask
}

/// Starting index for the RTL slope-limiting pass. [Gotcha G]
pub fn find_rtl_start(y: &[f64], peak_inds: &[usize], dip_inds: &[usize]) -> usize {
    let n = y.len();

    if !peak_inds.is_empty()
        && (dip_inds.is_empty()
            || peak_inds[peak_inds.len() - 1] > dip_inds[dip_inds.len() - 1])
    {
        let last_peak = peak_inds[peak_inds.len() - 1];
        let threshold = if dip_inds.is_empty() {
            y[0].max(*y.last().unwrap()) // [Gotcha G]
        } else {
            y[dip_inds[dip_inds.len() - 1]]
        };
        // First index in y[last_peak..] where y[i] <= threshold
        y[last_peak..]
            .iter()
            .enumerate()
            .find(|&(_, v)| *v <= threshold)
            .map(|(i, _)| i + last_peak)
            .unwrap_or(n - 1)
    } else {
        dip_inds.last().copied().unwrap_or(n - 1)
    }
}

/// Slope-limited left-to-right pass.
/// Slope baseline is limited[-1], not y[i-1] — clipping cascades. [Gotcha I]
/// Region bounds [region_start, i) exclusive; mask checked at i. [Gotcha F]
pub fn limited_ltr_slope(
    x: &[f64],
    y: &[f64],
    max_slope: f64,
    start_index: usize,
    peak_inds: &[usize],
    limit_free_mask: &[bool],
) -> Vec<f64> {
    let n = x.len();
    let mut limited = Vec::with_capacity(n);
    let mut clipped = Vec::with_capacity(n);
    let mut open_region: Option<usize> = None;

    for i in 0..n {
        if i <= start_index {
            limited.push(y[i]);
            clipped.push(false);
            continue;
        }

        let slope = log_log_gradient(x[i], x[i - 1], y[i], *limited.last().unwrap());
        let is_limit_free = limit_free_mask.get(i).copied().unwrap_or(false); // checked at i [Gotcha F]
        let prev_clipped = clipped[i - 1];

        if slope > max_slope && !is_limit_free {
            if !prev_clipped {
                open_region = Some(i);
            }
            clipped.push(true);
            let octaves = (x[i] / x[i - 1]).ln() / std::f64::consts::LN_2;
            limited.push(*limited.last().unwrap() + max_slope * octaves);
        } else {
            limited.push(y[i]);

            if prev_clipped {
                if let Some(region_start) = open_region.take() {
                    // Peak check: [region_start, i) exclusive [Gotcha F]
                    let has_peak = peak_inds.iter().any(|&p| p >= region_start && p < i);
                    if !has_peak {
                        for j in region_start..i {
                            limited[j] = y[j];
                            clipped[j] = false;
                        }
                    }
                }
            }
            clipped.push(false);
        }
    }

    limited
}

/// Slope-limited right-to-left: flip y/mask/peaks, run LTR, flip back. x is NOT flipped.
pub fn limited_rtl_slope(
    x: &[f64],
    y: &[f64],
    max_slope: f64,
    start_index: usize,
    peak_inds: &[usize],
    limit_free_mask: &[bool],
) -> Vec<f64> {
    let n = x.len();
    let rtl_start = n - start_index - 1;
    let flipped_peaks: Vec<usize> = peak_inds.iter().map(|&p| n - p - 1).collect();
    let flipped_mask: Vec<bool> = limit_free_mask.iter().rev().copied().collect();
    let y_flipped: Vec<f64> = y.iter().rev().copied().collect();

    let mut result =
        limited_ltr_slope(x, &y_flipped, max_slope, rtl_start, &flipped_peaks, &flipped_mask);
    result.reverse();
    result
}

/// Full equalization pipeline. Input: error = measured - target on the pipeline grid.
/// Output: equalization curve on the same frequency grid.
pub fn equalize(error: &[FreqPoint]) -> Vec<FreqPoint> {
    // Two-zone smooth the error (1/12 oct normal, 2 oct treble)
    let smoothed = two_zone_smooth(error, 1.0 / 12.0, 2.0);

    let x: Vec<f64> = smoothed.iter().map(|p| p.freq).collect();
    let y: Vec<f64> = smoothed.iter().map(|p| -p.db).collect(); // negate

    // Find peaks and dips BEFORE any gain/clip
    let peak_inds = find_peaks(&y, 1.0);
    let y_neg: Vec<f64> = y.iter().map(|&v| -v).collect();
    let dip_inds = find_peaks(&y_neg, 1.0);

    // Flat-line early return
    if peak_inds.is_empty() && dip_inds.is_empty() {
        return x
            .iter()
            .zip(y.iter())
            .map(|(&freq, &db)| FreqPoint { freq, db })
            .collect();
    }

    let limit_free_mask = protection_mask(&y, &peak_inds, &dip_inds);
    let rtl_start = find_rtl_start(&y, &peak_inds, &dip_inds);

    let limited_ltr = limited_ltr_slope(&x, &y, MAX_SLOPE, 0, &peak_inds, &limit_free_mask);
    let limited_rtl =
        limited_rtl_slope(&x, &y, MAX_SLOPE, rtl_start, &peak_inds, &limit_free_mask);

    // Element-wise min
    let mut combined: Vec<f64> = limited_ltr
        .iter()
        .zip(limited_rtl.iter())
        .map(|(&l, &r)| l.min(r))
        .collect();

    // Apply treble_gain_k BEFORE clipping [Gotcha E]
    // gain_k = 1 + (treble_gain_k - 1) * sigmoid; treble_gain_k=1.0 → identity for our goldens
    for (v, &freq) in combined.iter_mut().zip(x.iter()) {
        let k = log_f_sigmoid(freq, TREBLE_F_LOWER, TREBLE_F_UPPER);
        let gain_k = 1.0 + (1.0_f64 - 1.0) * k; // treble_gain_k=1.0
        *v *= gain_k;
    }

    // Clip positive gain to MAX_GAIN (no negative cap)
    for v in combined.iter_mut() {
        if *v > MAX_GAIN {
            *v = MAX_GAIN;
        }
    }

    // Re-smooth with 1/5 oct both zones
    let combined_fr: Vec<FreqPoint> = x
        .iter()
        .zip(combined.iter())
        .map(|(&freq, &db)| FreqPoint { freq, db })
        .collect();

    two_zone_smooth(&combined_fr, 1.0 / 5.0, 1.0 / 5.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    // Verified against scipy.signal.find_peaks 2026-04-25
    #[test]
    fn test_find_peaks_basic() {
        let y = [0.0, 1.0, 0.5, 2.0, 0.0, 1.5, 0.0];
        assert_eq!(find_peaks(&y, 0.0), vec![1, 3, 5]);
    }

    #[test]
    fn test_find_peaks_prominence_filter() {
        // Peak at 1 has prominence 0.5 → filtered with min=1.0
        let y = [0.0, 1.0, 0.5, 2.0, 0.0, 1.5, 0.0];
        assert_eq!(find_peaks(&y, 1.0), vec![3, 5]);
    }

    #[test]
    fn test_find_peaks_equal_height_traversal() {
        // Equal-height peaks scan through each other; both have prominence 1.0
        let y = [0.0, 1.0, 0.5, 1.0, 0.0];
        assert_eq!(find_peaks(&y, 0.0), vec![1, 3]);
        assert_eq!(find_peaks(&y, 1.0), vec![1, 3]);
    }

    #[test]
    fn test_find_peaks_plateau_left_edge() {
        let y = [0.0, 1.0, 1.0, 0.0];
        assert_eq!(find_peaks(&y, 0.0), vec![1]);
    }

    #[test]
    fn test_protection_mask_fewer_than_3_dips() {
        // One peak, one dip → after synthesis still < 3 → zero mask [Gotcha H]
        let y = [0.0f64, 2.0, 0.5, 1.0, 0.0];
        let peaks = find_peaks(&y, 1.0);
        let y_neg: Vec<f64> = y.iter().map(|&v| -v).collect();
        let dips = find_peaks(&y_neg, 1.0);
        let mask = protection_mask(&y, &peaks, &dips);
        assert!(mask.iter().all(|&v| !v));
    }

    #[test]
    fn test_find_rtl_start_no_dips() {
        // No dips: threshold = max(y[0], y[-1]) [Gotcha G]
        let y = [0.0f64, 3.0, 1.0, 2.0, 0.5];
        let rtl = find_rtl_start(&y, &[1], &[]);
        // threshold = max(0.0, 0.5) = 0.5; y[1..] = [3,1,2,0.5]; first <= 0.5 at rel idx 3 → abs 4
        assert_eq!(rtl, 4);
    }

    #[test]
    fn test_limited_ltr_slope_clipped_prior() {
        // Gotcha I: with a spike at i=1 (y=100), i=2 (y=99) must also be clipped because
        // slope is computed against limited[1] (~0.25), not raw y[1]=100.
        let x: Vec<f64> = (0..4).map(|i| 20.0 * 1.01f64.powi(i)).collect();
        let y = [0.0, 100.0, 99.5, 0.0];
        let result = limited_ltr_slope(&x, &y, 18.0, 0, &[1], &[false; 4]);
        assert!(result[1] < 1.0, "index 1 should be slope-limited");
        assert!(result[2] < 2.0, "index 2 should also be slope-limited (cascade)");
    }

    #[test]
    fn test_equalize_flat_error() {
        // Must use enough points so savgol window fits (695 for 1.01 grid)
        let error: Vec<FreqPoint> = (0..695)
            .map(|i| FreqPoint { freq: 20.0 * 1.01f64.powi(i), db: 0.0 })
            .collect();
        let eq = equalize(&error);
        for p in &eq {
            assert_abs_diff_eq!(p.db, 0.0, epsilon = 1e-10);
        }
    }
}
