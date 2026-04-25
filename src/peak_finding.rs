/// Properties returned by find_peaks_full.
#[derive(Debug, Clone)]
pub struct PeakProps {
    pub peak_heights: Vec<f64>,
    pub prominences: Vec<f64>,
    pub left_bases: Vec<usize>,
    pub right_bases: Vec<usize>,
    pub widths: Vec<f64>,
    pub width_heights: Vec<f64>,
    pub left_ips: Vec<f64>,
    pub right_ips: Vec<f64>,
}

/// scipy _local_maxima_1d semantics: plateaus return midpoint index.
pub(crate) fn local_maxima(y: &[f64]) -> Vec<usize> {
    let n = y.len();
    if n < 3 {
        return vec![];
    }
    let mut peaks = vec![];
    let mut i = 1usize;
    while i < n - 1 {
        if y[i] > y[i - 1] {
            let mut j = i + 1;
            while j < n && y[j] == y[i] {
                j += 1;
            }
            if j == n || y[j] < y[i] {
                peaks.push((i + j - 1) / 2);
                i = j;
                continue;
            }
        }
        i += 1;
    }
    peaks
}

/// scipy peak_prominences: (prominences, left_bases, right_bases).
/// Scans left/right from each peak until a strictly higher sample is found.
/// Equal-height neighbors do NOT terminate the scan.
pub(crate) fn peak_prominences(
    y: &[f64],
    peaks: &[usize],
) -> (Vec<f64>, Vec<usize>, Vec<usize>) {
    let n = y.len();
    let mut prominences = Vec::with_capacity(peaks.len());
    let mut left_bases = Vec::with_capacity(peaks.len());
    let mut right_bases = Vec::with_capacity(peaks.len());

    for &p in peaks {
        let h = y[p];

        // Left: scan until y[i] > h (strictly). Interval is [left_start, p).
        let left_start = if p > 0 {
            let mut i = p - 1;
            loop {
                if y[i] > h {
                    break i + 1;
                }
                if i == 0 {
                    break 0;
                }
                i -= 1;
            }
        } else {
            p // empty range → left_base = p → left contribution = h → prominence 0
        };

        let left_base = (left_start..p)
            .min_by(|&a, &b| y[a].partial_cmp(&y[b]).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(p);

        // Right: scan until y[j] > h. Interval is (p, right_end).
        let right_end = (p + 1..n).find(|&j| y[j] > h).unwrap_or(n);

        let right_base = (p + 1..right_end)
            .min_by(|&a, &b| y[a].partial_cmp(&y[b]).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(p);

        let base_level = y[left_base].max(y[right_base]);
        prominences.push(h - base_level);
        left_bases.push(left_base);
        right_bases.push(right_base);
    }

    (prominences, left_bases, right_bases)
}

/// scipy peak_widths at given rel_height (typically 0.5).
/// Returns (widths, width_heights, left_ips, right_ips) with fractional sample indices.
pub(crate) fn peak_widths_at(
    y: &[f64],
    peaks: &[usize],
    rel_height: f64,
    prominences: &[f64],
    left_bases: &[usize],
    right_bases: &[usize],
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut widths = Vec::with_capacity(peaks.len());
    let mut width_heights = Vec::with_capacity(peaks.len());
    let mut left_ips = Vec::with_capacity(peaks.len());
    let mut right_ips = Vec::with_capacity(peaks.len());

    for (idx, &p) in peaks.iter().enumerate() {
        let h_eval = y[p] - prominences[idx] * rel_height;
        let lb = left_bases[idx];
        let rb = right_bases[idx];

        // Left side: scan from p-1 down to lb, find first i where y[i] < h_eval.
        let left_ip = if p > lb {
            let mut i = p - 1;
            loop {
                if y[i] < h_eval {
                    // Crossing between i (below) and i+1 (above). y[i+1] >= h_eval guaranteed.
                    let denom = y[i + 1] - y[i]; // > 0
                    let frac = if denom.abs() < 1e-300 {
                        0.0
                    } else {
                        ((h_eval - y[i]) / denom).clamp(0.0, 1.0)
                    };
                    break i as f64 + frac;
                }
                if i == lb {
                    break lb as f64; // reached base without crossing
                }
                i -= 1;
            }
        } else {
            lb as f64
        };

        // Right side: scan from p+1 up to rb, find first j where y[j] < h_eval.
        let right_ip = if p < rb {
            let mut j = p + 1;
            loop {
                if y[j] < h_eval {
                    // Crossing between j-1 (above) and j (below). y[j-1] >= h_eval guaranteed.
                    let denom = y[j] - y[j - 1]; // < 0
                    let frac = if denom.abs() < 1e-300 {
                        0.0
                    } else {
                        ((h_eval - y[j - 1]) / denom).clamp(0.0, 1.0)
                    };
                    break (j - 1) as f64 + frac;
                }
                if j == rb {
                    break rb as f64; // reached base without crossing
                }
                j += 1;
            }
        } else {
            rb as f64
        };

        widths.push(right_ip - left_ip);
        width_heights.push(h_eval);
        left_ips.push(left_ip);
        right_ips.push(right_ip);
    }

    (widths, width_heights, left_ips, right_ips)
}

/// Simple find_peaks: returns indices of local maxima with prominence >= min_prominence.
/// Used by the equalize pipeline (prominence=1.0) and anywhere full PeakProps are not needed.
pub fn find_peaks(y: &[f64], min_prominence: f64) -> Vec<usize> {
    let candidates = local_maxima(y);
    if candidates.is_empty() {
        return vec![];
    }
    if min_prominence <= 0.0 {
        return candidates;
    }
    let (proms, _, _) = peak_prominences(y, &candidates);
    candidates
        .into_iter()
        .zip(proms)
        .filter(|(_, p)| *p >= min_prominence)
        .map(|(i, _)| i)
        .collect()
}

/// Full find_peaks with prominence/height/width filters, returning PeakProps.
/// Passing 0.0 for all thresholds returns all local maxima.
/// Used by init_peaking (peq.py Peaking.init) which needs height×width scoring.
pub fn find_peaks_full(
    y: &[f64],
    min_prominence: f64,
    min_height: f64,
    min_width: f64,
) -> (Vec<usize>, PeakProps) {
    let candidates = local_maxima(y);
    if candidates.is_empty() {
        return (
            vec![],
            PeakProps {
                peak_heights: vec![],
                prominences: vec![],
                left_bases: vec![],
                right_bases: vec![],
                widths: vec![],
                width_heights: vec![],
                left_ips: vec![],
                right_ips: vec![],
            },
        );
    }

    // Filter by height
    let candidates: Vec<usize> =
        candidates.into_iter().filter(|&i| y[i] >= min_height).collect();

    if candidates.is_empty() {
        return empty_props();
    }

    // Compute prominences
    let (proms, lbs, rbs) = peak_prominences(y, &candidates);

    // Filter by prominence — collect index, prom, lb, rb tuples in one pass
    let mut filtered: Vec<usize> = Vec::new();
    let mut filtered_proms: Vec<f64> = Vec::new();
    let mut filtered_lbs: Vec<usize> = Vec::new();
    let mut filtered_rbs: Vec<usize> = Vec::new();
    for (idx, &ix) in candidates.iter().enumerate() {
        if proms[idx] >= min_prominence {
            filtered.push(ix);
            filtered_proms.push(proms[idx]);
            filtered_lbs.push(lbs[idx]);
            filtered_rbs.push(rbs[idx]);
        }
    }

    if filtered.is_empty() {
        return empty_props();
    }

    // Compute widths
    let (ws, whs, lips, rips) =
        peak_widths_at(y, &filtered, 0.5, &filtered_proms, &filtered_lbs, &filtered_rbs);

    // Filter by width
    let mut final_inds: Vec<usize> = Vec::new();
    let mut peak_heights: Vec<f64> = Vec::new();
    let mut final_proms: Vec<f64> = Vec::new();
    let mut final_lbs: Vec<usize> = Vec::new();
    let mut final_rbs: Vec<usize> = Vec::new();
    let mut final_ws: Vec<f64> = Vec::new();
    let mut final_whs: Vec<f64> = Vec::new();
    let mut final_lips: Vec<f64> = Vec::new();
    let mut final_rips: Vec<f64> = Vec::new();

    for (idx, &ix) in filtered.iter().enumerate() {
        if ws[idx] >= min_width {
            final_inds.push(ix);
            peak_heights.push(y[ix]);
            final_proms.push(filtered_proms[idx]);
            final_lbs.push(filtered_lbs[idx]);
            final_rbs.push(filtered_rbs[idx]);
            final_ws.push(ws[idx]);
            final_whs.push(whs[idx]);
            final_lips.push(lips[idx]);
            final_rips.push(rips[idx]);
        }
    }

    (
        final_inds,
        PeakProps {
            peak_heights,
            prominences: final_proms,
            left_bases: final_lbs,
            right_bases: final_rbs,
            widths: final_ws,
            width_heights: final_whs,
            left_ips: final_lips,
            right_ips: final_rips,
        },
    )
}

fn empty_props() -> (Vec<usize>, PeakProps) {
    (
        vec![],
        PeakProps {
            peak_heights: vec![],
            prominences: vec![],
            left_bases: vec![],
            right_bases: vec![],
            widths: vec![],
            width_heights: vec![],
            left_ips: vec![],
            right_ips: vec![],
        },
    )
}


#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_local_maxima_basic() {
        let y = [0.0, 1.0, 0.5, 2.0, 0.0, 1.5, 0.0];
        assert_eq!(local_maxima(&y), vec![1, 3, 5]);
    }

    #[test]
    fn test_local_maxima_plateau() {
        let y = [0.0, 1.0, 1.0, 0.0];
        assert_eq!(local_maxima(&y), vec![1]); // midpoint of [1,2] = 1
    }

    #[test]
    fn test_find_peaks_prominence_filter() {
        let y = [0.0, 1.0, 0.5, 2.0, 0.0, 1.5, 0.0];
        assert_eq!(find_peaks(&y, 0.0), vec![1, 3, 5]);
        assert_eq!(find_peaks(&y, 1.0), vec![3, 5]);
    }

    #[test]
    fn test_peak_prominences_basic() {
        // Peak at 3 (height 2), surrounded by valleys at 0 and 0. Prominence = 2.
        // Peak at 1 (height 1), right base = idx 2 (height 0.5). Left base = idx 0 (0.0). Prominence = 1 - 0.5 = 0.5.
        let y = [0.0, 1.0, 0.5, 2.0, 0.0, 1.5, 0.0];
        let peaks = vec![1, 3, 5];
        let (proms, lbs, rbs) = peak_prominences(&y, &peaks);
        assert_abs_diff_eq!(proms[0], 0.5, epsilon = 1e-12); // peak@1: right_base y[2]=0.5, left_base y[0]=0 → 1-0.5=0.5
        assert_abs_diff_eq!(proms[1], 2.0, epsilon = 1e-12); // peak@3: left_start=0, argmin y[0..3]=y[0]=0 → lb=0, rb=4 → 2-0=2
        assert_abs_diff_eq!(proms[2], 1.5, epsilon = 1e-12); // peak@5: lb=4(y=0), rb=6(y=0) → 1.5-0=1.5
        assert_eq!(lbs[1], 0); // left scan from peak@3: no y>2 found, left_start=0, argmin y[0..3] at idx 0
        assert_eq!(rbs[1], 4); // right scan from peak@3: y[4]=0<2, first strictly higher never found, argmin y[4..7] at idx 4
    }

    #[test]
    fn test_peak_widths_at_basic() {
        // Simple symmetric peak: y = [0, 0, 1, 0, 0] at index 2 with prominence 1.0
        let y = [0.0, 0.0, 1.0, 0.0, 0.0];
        let peaks = vec![2];
        let (proms, lbs, rbs) = peak_prominences(&y, &peaks);
        let (ws, _, lips, rips) = peak_widths_at(&y, &peaks, 0.5, &proms, &lbs, &rbs);
        // h_eval = 1.0 - 0.5 * 1.0 = 0.5. Left crossing between idx 1 (y=0) and idx 2 (y=1) → left_ip = 1 + 0.5 = 1.5
        // Right crossing between idx 2 (y=1) and idx 3 (y=0) → right_ip = 2 + 0.5 = 2.5
        // width = 2.5 - 1.5 = 1.0
        assert_abs_diff_eq!(ws[0], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(lips[0], 1.5, epsilon = 1e-12);
        assert_abs_diff_eq!(rips[0], 2.5, epsilon = 1e-12);
    }

    #[test]
    fn test_find_peaks_full_zero_thresholds() {
        // All local maxima returned with zero thresholds
        let y = [0.0, 2.0, 1.0, 3.0, 0.0, 1.5, 0.0];
        let (inds, props) = find_peaks_full(&y, 0.0, 0.0, 0.0);
        assert_eq!(inds, vec![1, 3, 5]);
        assert_eq!(props.peak_heights, vec![2.0, 3.0, 1.5]);
        assert_eq!(props.widths.len(), 3);
    }
}
