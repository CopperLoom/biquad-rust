# Implementation Plan — biquad-fit (Rust)

Rust crate reimplementing AutoEQ's parametric EQ optimizer with SLSQP and per-parameter
locking. Architecture and algorithm spec: `docs/ARCHITECTURE.md`.

---

## Phased Implementation

Each phase produces a testable increment. Tests are written alongside (not after)
implementation. A phase is not done until its tests pass.

---

### Phase 0: Prerequisites + Golden File Regeneration

Resolve ambiguities, pin references, and regenerate goldens **from AutoEQ directly** before
any Rust code is written. Eliminates biquad-fit from the correctness chain entirely.

**Tasks:**

**0a. Pin AutoEQ version.**
Record the exact commit hash of `reference-implementations/autoEQ/` in ARCHITECTURE.md's
Reference Implementations table. This pins the oracle version for all goldens.

**0b. Write `tests/generate_golden.py`** (Python, in this repo — replaces biquad-fit's version).
~50 lines. Imports AutoEQ directly. For each of 90 combinations:
1. Load IEM FR from `tests/fixtures/fr/{iem}.json`
2. Load target from `tests/fixtures/targets/{target}.json`
3. Run full AutoEQ pipeline (interpolate → center → compensate → equalize → optimize)
4. Compute pregain as `pregain = -peq.max_gain` if `peq.max_gain > 0` else `0.0`
   (pin exact formula from `peq.py:max_gain` + `PREAMP_HEADROOM` — resolve ambiguity
   by reading `frequency_response.py:write_eqapo_parametric_eq` and `_optimize_peq_filters`)
5. Write JSON with **Rust-native field names**: `filter_type`, `fc`, `gain`, `q`
   (not AutoEQ's `type`, `freq`). No serde renames needed.

**0c. Run the generator, commit regenerated goldens.**
All 90 `tests/fixtures/golden/*.json` replaced. Verify count = 90.

**0d. Resolve remaining ARCHITECTURE.md ambiguities.**
- **10 kHz slice boundary**: `peq.py:449,593` — `_10k_ix = argmin(|f - 10000|)`,
  slice `[_10k_ix:]` is inclusive of that index. Update ARCH §7.
- **Pregain formula**: pinned by step 0b above. Update ARCH §8.
- **Shelf fc search clamp**: `peq.py:334–335, 386–387` — init searches
  `[max(40, min_fc), min(10000, max_fc)]` regardless of user config. Update ARCH §6.

---

### Phase 1: Project Scaffolding + Types ✓ COMPLETE

Define all shared types. (Cargo init, fixtures, and bench stub are already done.)

**Numeric convention:** f64 throughout — matches AutoEQ's numpy float64. No f32, no generics.

**Tasks:**
- Clean `Cargo.toml`: remove `ndarray`, `realfft`, `num-traits` (not needed; AutoEQ uses none of these).
  Final deps: `slsqp`, `serde` + `serde_json`. Dev: `approx`, `criterion`.
- Create `src/types.rs`:
  - `FilterType` enum: `PK`, `LSQ`, `HSQ` (with serde rename for JSON compat)
  - `FreqPoint { freq: f64, db: f64 }`
  - `Filter { filter_type: FilterType, fc: f64, gain: f64, q: f64 }`
    Golden files use the same field names (`filter_type`, `fc`) — no serde renames needed.
    Generator (Phase 0) writes Rust-native names.
  - `FilterSpec` — mirrors AutoEQ's per-filter fields exactly:
    ```rust
    pub struct FilterSpec {
        pub filter_type:   Option<FilterType>,  // defaults to PK
        pub fc:            Option<f64>,    // initial value; None = auto-init from correction curve
        pub q:             Option<f64>,    // initial value; None = auto-init
        pub gain:          Option<f64>,    // initial value; None = auto-init
        pub optimize_fc:   Option<bool>,   // None = true; false = lock fc at provided value
        pub optimize_q:    Option<bool>,   // None = true; false = lock q
        pub optimize_gain: Option<bool>,   // None = true; false = lock gain
        pub fc_range:   Option<(f64, f64)>,  // defaults by type
        pub q_range:    Option<(f64, f64)>,  // defaults by type
        pub gain_range: (f64, f64),           // required
    }
    ```
    Three states per param:
    - `fc=None, optimize_fc=None/true` → auto-init from correction curve, free to optimize
    - `fc=Some(x), optimize_fc=None/true` → seeded at x, free to optimize (matches AutoEQ)
    - `fc=Some(x), optimize_fc=Some(false)` → locked at x (validation error if fc=None + optimize_fc=false)
  - `MinStd` enum:
    ```rust
    pub enum MinStd {
        Default,       // 0.002 — matches DEFAULT_PEQ_OPTIMIZER_MIN_STD
        Disabled,      // run to 150 iterations (AutoEQ peq.yaml `min_std: null`)
        Custom(f64),   // caller-specified threshold
    }
    ```
  - `Constraints`:
    ```rust
    pub struct Constraints {
        pub filter_specs: Vec<FilterSpec>,
        pub freq_range: Option<(f64, f64)>,  // default [20, 10000]
        pub fs: Option<f64>,                 // default 44100
        pub min_std: Option<MinStd>,         // None = MinStd::Default
    }
    ```
  - `OptimizeResult { pregain: f64, filters: Vec<Filter> }`
  - `InterpolateOptions { step: Option<f64>, f_min: Option<f64>, f_max: Option<f64> }`
  - `BiquadError` enum:
    ```rust
    pub enum BiquadError {
        InvalidFilterSpec(String),      // e.g. fc=None + optimize_fc=false
        InvalidFrequencyResponse(String), // NaN, < 2 points, etc.
        OptimizerFailed(String),
    }
    ```
  - `pub fn optimize(...) -> Result<OptimizeResult, BiquadError>` — no panics on public API
    surface; internal invariants may still use `.expect()`
- Replace `src/lib.rs` stub with module declarations
- Create `tests/helpers/mod.rs` for golden file loading and RMSE computation

**Tests:** Types compile, golden files parse correctly, RMSE helper produces known values.

---

### Phase 2: Biquad Response + Interpolation ✓ COMPLETE

Two modules with no internal dependencies.

**Tasks:**
- `src/biquad_response.rs`:
  - `biquad_coeffs(filter_type, fc, gain, q, fs) -> BiquadCoeffs`
  - `eval_magnitude(coeffs, freq, fs) -> f64` — phi identity formula
  - `biquad_response(filter_type, fc, gain, q, frequencies, fs) -> Vec<f64>`
- `src/interpolate.rs`:
  - `build_grid(f_min, f_max, step) -> Vec<f64>`
  - `interpolate(fr, options) -> Vec<FreqPoint>` — log-linear with binary search

**Tests:**
- Biquad: PK/LSQ/HSQ at known params match expected dB. Cross-check against AutoEQ
  `PEQFilter.fr` for identical parameters.
- Interpolate: grid point count matches AutoEQ (695 for 1.01, 349 for 1.02).
  Values at original measurement points preserved within tolerance.

---

### Phase 3: Smooth + Compensate ✓ COMPLETE

**Tasks:**
- `src/smooth.rs`:
  - `savgol_coeffs(window_size, poly_order) -> Vec<f64>` — Vandermonde + Gauss-Jordan
  - `savgol_filter(data, window_size) -> Vec<f64>` — convolution + edge polynomial fit
    **Edge mode `'interp'`:** fit a polynomial to the last `window_length` samples at each
    boundary; do not use mirror/reflect/nearest. Verify chosen crate supports this or
    implement manually. Affects 20 Hz and 20 kHz extremes. [Gotcha J]
  - `smoothing_window_size(freqs, octaves) -> usize`
  - `log_f_sigmoid(f, f_lower, f_upper) -> f64`
  - `two_zone_smooth(fr, normal_octaves, treble_octaves) -> Vec<FreqPoint>`
    **Two full passes:** run savgol over the entire array with normal window → array A;
    run savgol over the entire array with treble window → array B; sigmoid-blend A and B
    in the 6–8 kHz zone. Not a single variable-window pass. [Gotcha D]
  - `smooth(fr, window_octaves) -> Vec<FreqPoint>` — public single-zone API
- `src/compensate.rs`:
  - `compensate(measured, target) -> Vec<FreqPoint>` — **four-step pipeline** [Gotcha A]:
    1. Interpolate target to measurement grid (k=1 linear spline, extrapolates beyond
       input range — does not clamp) [Gotcha C]
    2. Center target at 1 kHz via log-linear interpolation (1 kHz is off the 1.01 grid;
       nearest points ~995/1005 Hz — interpolate, do not index nearest) [Gotcha B]
    3. Add `create_target()` contributions (tilt, bass/treble shelves; default = zero)
    4. Subtract: `error[i] = measured[i] - target[i]`

**Tests:**
- Savitzky-Golay: polynomial data (degree <= 2) passes through unchanged.
  Window size matches AutoEQ's `smoothing_window_size()` for 1.01 and 1.02 grids.
  Edge values match scipy `mode='interp'` output on known fixture.
- Two-zone smooth: verify that the blend at 7 kHz is between the normal-window and
  treble-window outputs (not some other interpolant).
- Sigmoid: verify weights at 5 kHz (~0), 7 kHz (~0.5), 10 kHz (~1).
- Compensate: verify four-step pipeline against AutoEQ Python output on a fixture pair.
  Specifically: off-grid 1 kHz centering and extrapolating interpolation.

---

### Phase 4: Equalize ✓ COMPLETE

The most complex single module. Full slope-limiting pipeline.

**Tasks:**
- `src/equalize.rs`:
  - `find_peaks(arr, prominence, width, height) -> (Vec<usize>, PeakProps)` —
    **Two distinct call sites with different params; implement a single flexible function:**
    - Equalize pipeline (ARCH §5): `prominence >= 1.0`, on raw `y` and `-y`
    - Peaking filter init (Phase 5): `prominence=0, width=0, height=0`, on
      `clip(target, 0, None)` and `clip(-target, 0, None)`
    Port scipy's algorithm from `scipy/signal/_peak_finding.py` directly. Prominence
    is non-trivial (recursive parent-peak search). `PeakProps` includes `widths`,
    `peak_heights` (needed by peaking init's height×width scoring).
  - `peak_widths(arr, peak_ixs) -> Vec<f64>` — interpolated widths at `rel_height=0.5`
    (scipy default). Port from `scipy/signal/_peak_finding.py`. Needed by peaking init.
    **Do not implement `band_penalty`** — it exists in AutoEQ's class hierarchy but is
    never summed into the loss function (`_optimizer_loss` only adds `sharpness_penalty`).
  - `protection_mask(y, peak_inds, dip_inds) -> Vec<bool>`
    **Early return:** after inserting synthetic endpoint dips, if total dip count < 3,
    return zero mask immediately (no protected zones). [Gotcha H]
  - `find_rtl_start(y, peak_inds, dip_inds) -> usize`
    **No-dips fallback:** when no dips exist, threshold = `max(y[0], y[-1])`; RTL start =
    first index where `y[i] > threshold`. Not "start at 0" or "start at end". [Gotcha G]
  - `limited_ltr_slope(freqs, y, max_slope, start_index, peak_inds, limit_free_mask) -> Vec<f64>`
    — with region validation.
    **Region bounds:** `[region_start, i)` exclusive upper end; protection mask checked at
    index `i`, not `i-1`. [Gotcha F]
    **Slope baseline:** use `limited[-1]` (last already-clipped output) as the prior value
    for slope calculation, not `y[i-1]` (raw input). Clipping cascades through the array.
    [Gotcha I, `frequency_response.py:738`]
  - `limited_rtl_slope(...)` — flip, run LTR, flip back
  - `equalize(error) -> Vec<FreqPoint>` — full pipeline.
    **`treble_gain_k` ordering:** multiply treble region by `treble_gain_k` **before**
    clipping to `max_gain` (not after). For our goldens `treble_gain_k=1.0`, but order
    must be correct for API completeness. [Gotcha E, `frequency_response.py:616-621`]

**Tests:**
- Flat error -> equalization is the negation.
- Known step error -> slope capped at 18 dB/oct.
- Peak finding: compare against AutoEQ Python (`scipy.signal.find_peaks`) output on fixture data.
- protection_mask: <3 dips after synthetic insertion → zero mask. [Gotcha H]
- find_rtl_start: no-dips input → threshold = max(y[0], y[-1]). [Gotcha G]
- limited_ltr_slope: verify slope uses clipped prior, not raw prior. [Gotcha I]
- Full equalize: compare output against AutoEQ Python for each test fixture.

#### Research Notes

Line numbers below verified against
`reference-implementations/autoEQ/autoeq/frequency_response.py` on 2026-04-24.
See `docs/ARCHITECTURE.md` §"Phase 4 Source Reference" for verbatim source.

**Key algorithmic decisions extracted from source:**

- `find_peaks` is called twice only, both inside `equalize()` at lines 577-578
  with the exact signature `find_peaks(y, prominence=1)` / `find_peaks(-y, prominence=1)`.
  No `width`, `height`, `distance`, `threshold`, `wlen`, `plateau_size`, or
  `rel_height` are ever passed from the equalize pipeline — all defaults.
  `peak_props` is discarded; only the index arrays flow forward.
- `peak_widths` is **not** called in the equalize pipeline. It is only used
  by peaking-filter initialization (Phase 5, in `peq.py`). Do not waste Phase
  4 implementation budget on `peak_widths` unless also tackling Phase 5 inits.
- `prominence=1` means minimum=1.0 dB, no upper bound (scipy's `_unpack_condition_args`
  interprets a scalar as `(imin, None)`).
- The smoothing applied inside `equalize()` (lines 568-570) uses `window_size=1/12`,
  `treble_window_size=2` by default — the same two-zone shape as Phase 3, but
  applied to the **error** (`self.error`) inside a throwaway `FrequencyResponse`.
- After min-combining LTR/RTL (line 613), the treble weighting `gain_k`
  (log-f sigmoid between `treble_f_lower` and `treble_f_upper`) multiplies
  the combined curve (line 617) **before** clipping to `max_gain` (line 621),
  **before** a final 1/5-oct re-smooth (line 623).
- Final re-smooth at line 623 uses `window_size=1/5, treble_window_size=1/5`
  — same window for both zones. This flattens hard kinks from the clipper;
  do not skip it.
- `log_log_gradient` on line 738 uses `limited[-1]`, not `y[i-1]` — clipping
  cascades. This is Gotcha I.
- Clipped-region accounting uses at least **three distinct indices** that
  must not be conflated:
  - `region_start` — the index where clipping began (line 750).
  - `i` — the first unclipped sample after the region (line 763 stores
    `i + 1` as the "end" marker).
  - `region_end = i + 1` as stored in `regions` — one past the first
    unclipped sample.
  - The peak-intersection validation on line 766 uses `peak_inds >= region_start`
    **AND** `peak_inds < i` (not `< i + 1`, not `< region_end`).
  - The revert span on lines 768-769 is `[region_start : i]` (exclusive at `i`).
  - The trailing-open-region close on line 774 appends `len(x) - 1`, not `len(x)`.
- `limited_rtl_slope` (lines 691-701) flips `y`, `peak_inds`, `limit_free_mask`
  and transforms `start_index` → `len(x) - start_index - 1`, but **does not
  flip `x`**. The inner LTR call receives a monotonically increasing frequency
  axis paired with a reversed amplitude array. Octave ratios come out
  sign-swapped but are only used as magnitudes, so it works. Port this
  asymmetry faithfully.
- `protection_mask` endpoint synthesis (lines 648-656) is asymmetric:
  - Last-peak-after-last-dip: append `argmin(y[last_peak:]) + last_peak`,
    then read its level from `y`.
  - Otherwise: append literal index `-1`, then **overwrite** that level with
    `np.min(y)` — the sentinel index is never dereferenced for its value.
    A naive port that appends `y[-1]` as the level diverges when `y[-1]` is
    not the minimum.
- `protection_mask` `< 3` early return is at lines 659-660, **after** the
  synthetic insertion — count the synthesized array, not the input. Gotcha H.
- `protection_mask` mask-write (line 668) is `mask[left_ind : right_ind + 1]`,
  right-inclusive. The `right_ind` computation on line 667 subtracts 1
  (`+ dip_ind - 1`), so the net stored span is `[left_ind, right_ind + 1)` in
  Python-slice terms — verify this exactly when porting; an off-by-one here
  leaks a protected sample in or out.
- `find_rtl_start` uses `<=` for the threshold crossing, not `<` or `>`
  (lines 795 and 798). The no-dips fallback threshold is `max(y[0], y[-1])`
  (line 798) — Gotcha G. Fallback when nothing crosses: `len(y) - 1`
  (line 802), not 0 and not `len(y)`.

**Gotcha line-number references (verified against source):**

- Gotcha E (`treble_gain_k` before clip): lines 616-621 — matches existing
  architecture note.
- Gotcha F (region bounds `[region_start, i)`, mask check at `i`): lines
  746, 750, 763, 766, 768.
- Gotcha G (RTL no-dips fallback): line 798.
- Gotcha H (`protection_mask` `< 3` early return): lines 659-660.
- Gotcha I (slope baseline is `limited[-1]`): line 738 — matches existing
  architecture note.

**Edge cases / surprises not already documented:**

- `np.argwhere(...)[-1, 0] + 1` (line 666) and `np.argwhere(...)[0, 0] + dip_ind - 1`
  (line 667) both call `argwhere` unguarded. If either returns an empty array
  (monotone signal where no sample is `>= target`), Python raises `IndexError`.
  AutoEQ relies on the structural invariant that adjacent dip levels exist on
  both sides of an interior dip. The Rust port should either replicate
  (panic on malformed input) or guard and fall back — decide explicitly.
- `protection_mask`'s sentinel `-1` index is *never* used to look up a `y`
  value (line 656 overwrites that slot before use). A Rust port using `usize`
  cannot store `-1`; carry a separate sentinel flag or model dip_levels as
  `Vec<f64>` directly without round-tripping through indices.
- `regions_ltr` / `regions_rtl` are returned but never consumed by `equalize`.
  Only `limited_*` and `clipped_*` (and, implicitly, the in-function region
  validation) matter. The Rust port can skip materializing regions beyond
  what the LTR loop needs internally.
- scipy's `_peak_prominences` and `_peak_widths` live in
  `scipy/signal/_peak_finding_utils` — shipped as a compiled `.so` in this
  install (no `.pyx` source on disk). The normative spec is the wrapper
  docstrings in `_peak_finding.py` at lines 323-466 and 467-590. Validate
  the Rust port against scipy runtime output on fixture data, not against
  a reference Python source.
- scipy's prominence scan ignores **equal-height** neighbouring peaks (the
  horizontal line must hit a *higher* peak to terminate). A naive `>=`
  termination deviates from scipy on plateaus and multi-peak ridges.
- scipy's width output `left_ips` / `right_ips` are *fractional* sample
  indices from linear interpolation on the straddling slope. Returning
  integer indices drops ~0.5 sample of width precision per side and will
  break any downstream Q-from-width scoring.
- The concha-interference path (lines 593-595, 740) and `max_slope_decay`
  path (line 744) are live code but unused by our golden fixtures. Omitting
  them from the Rust port is consistent with the existing plan; just document
  that skipping them makes the equalize surface a strict subset of AutoEQ's.

---

### Phase 5: Optimizer Core (SLSQP) ✓ COMPLETE

The critical phase. Wire up `relf/slsqp` with loss function, initialization, param
encoding, and per-parameter locking.

**Tasks:**
- `src/optimize.rs`:
  - `resolve_specs(filter_specs, default_freq_range) -> Vec<ResolvedFilterSpec>`
    - Derives `optimize_fc`, `optimize_q`, `optimize_gain` from `FilterSpec.fc/q/gain`
  - `encode_params(filters, resolved_specs) -> Vec<f64>` — skip fixed params
  - `decode_params(x, resolved_specs, fixed_values) -> Vec<Filter>` — splice fixed back in
  - `build_bounds(resolved_specs) -> Vec<(f64, f64)>` — skip fixed params
  - `total_response(filters, freqs, fs) -> Vec<f64>`
  - `sharpness_penalty(filter_type, fc, gain, q, freqs, fs) -> f64`
  - `joint_loss(filters, freqs, correction_db, fs) -> f64`
  - `init_peaking(freqs, correction_db, spec, fs) -> Filter`
  - `init_low_shelf(freqs, correction_db, spec, fs) -> Filter`
    — **signed** weighting: `dot(target, shelf_fr) / sum(shelf_fr)`
  - `init_high_shelf(freqs, correction_db, spec, fs) -> Filter`
    — same signed weighting
  - `compute_pregain(filters, freqs, fs, gain_range) -> f64`
  - `joint_optimize(initial_filters, resolved_specs, freqs, correction_db, fs) -> Vec<Filter>`
    — SLSQP with STD-based convergence callback
  - `optimize(measured, target, constraints) -> OptimizeResult`
    — full pipeline entry point

**SLSQP API spike (do first in this phase):** Read `relf/slsqp` crate docs/source to confirm
whether it provides finite-difference gradients internally or requires an explicit gradient
function. If explicit: use forward differences with h = sqrt(f64::EPSILON). Also confirm the
callback/termination mechanism — AutoEQ exits via Python exception (`OptimizationFinished`);
Rust needs a different signal (return value, mutable flag, or crate-native API). Document
both findings before proceeding.

**Init priority sort — explicit task:** Implement the 12-entry priority table from
ARCHITECTURE.md §6 (`_init_optimizer_params` ordering). Filters must be sorted by priority
before initialization. This is easy to miss and breaks correctness if skipped — read
ARCHITECTURE.md §6 fully before writing `resolve_specs`.

**Parameter vector ordering:** Free params enter the SLSQP `x` vector per-filter grouped:
`[f0.fc, f0.q, f0.gain, f1.fc, f1.q, f1.gain, ...]`, skipping locked params within each
filter. `encode_params` and `decode_params` must use this same order. Matches AutoEQ's
`_parse_optimizer_params` (peq.py:567–583).

**`init_peaking` — call site details (peq.py:179–182):**
- Calls `find_peaks` with `prominence=0, width=0, height=0` on `clip(target, 0, None)`
  (positive peaks) and `clip(-target, 0, None)` (negative peaks/dips) separately.
- Uses `peak_props['widths']` (from `peak_widths` at rel_height=0.5) and `peak_heights`
  for height × width scoring. Fc range filtered to `[min_fc_ix, max_fc_ix]`.
- No-peaks fallback: fc = midpoint of fc range, Q = sqrt(2), gain = 0.0.

**Shelf init details (peq.py:317–401):**
- fc search clamped to `[max(40, min_fc), min(10000, max_fc)]` — hardcoded outer bounds.
- Q init: `clip(0.7, min_q, max_q)` — clamps into user-specified q_range.
- Gain: `dot(target, shelf_fr) / sum(shelf_fr)` with `self.gain = 1` seed before computing.

**Convergence callback — population std:** `np.std(history[-n:])` is population std (ddof=0).
Implement as `sqrt(mean((x_i - mean(x))^2))` — 2-line manual impl, no stdlib equivalent.
Best-params restoration on exit: accumulate `(params, loss)` every callback iteration;
restore `params[argmin(loss)]` after early termination (peq.py:719).

**Sharpness penalty — numerical stability (peq.py:262–266):**
Sigmoid `1 / (1 + exp(-100 * x))` overflows for moderate x. Clamp exponent to [-500, 500]
before `exp`, or use stable form: `if x >= 0 { 1/(1+exp(-100x)) } else { exp(100x)/(1+exp(100x)) }`.
**Shelf filters return `sharpness_penalty = 0.0` always** — only Peaking has a non-zero penalty.
**`band_penalty` is dead code — do not implement** (defined on filters but never added to loss).

**Max iterations:** 150.

**Per-parameter locking implementation:**
1. `FilterSpec { fc: Some(8000.0), optimize_fc: Some(false), .. }` -> fc locked at 8000
2. `FilterSpec { fc: Some(2500.0), optimize_fc: None, .. }` -> seeded at 2500, optimizer is free
3. `encode_params` skips locked params -> shorter optimization vector
4. `build_bounds` skips locked params -> matching shorter bounds vector
5. `joint_loss` includes all filters in cascade (locked and free)
6. `decode_params` splices locked values back into full filter list
7. Init sequence: locked-param filters use provided values (not re-initialized),
   and their response is subtracted from remaining target before initializing free filters

**Real-world pattern (from AutoEQ's `peq.yaml`):** Shelf overlap is prevented by
locking shelf fc values or constraining their ranges, not by any optimizer logic:
- LSQ: `fc: Some(105.0), optimize_fc: Some(false), q: Some(0.7), optimize_q: Some(false)` — gain-only
- HSQ: `fc_range: Some((5000.0, 12000.0))` — fc seeded + free but bounded away from LSQ

**Configurable convergence (`MinStd` enum):**
- `None` / `Some(MinStd::Default)` -> 0.002 (matches `DEFAULT_PEQ_OPTIMIZER_MIN_STD`)
- `Some(MinStd::Disabled)` -> run to 150 iterations (AutoEQ's `min_std: null`)
- `Some(MinStd::Custom(v))` -> caller-specified threshold

**Tests:**
- Loss: known filter set + target -> verify MSE matches hand calculation.
- Init: compare params against AutoEQ Python output for fixtures.
- Encoding round-trip: encode -> decode is identity.
- Locking: locked params unchanged after optimization.

---

### Phase 6: Apply Filters + Public API ✓ COMPLETE

**Tasks:**
- `src/apply_filters.rs`:
  - `apply_filters(fr, filters, pregain, fs) -> Vec<FreqPoint>`
- Finalize `src/lib.rs`:
  - Re-export: `optimize`, `apply_filters`, `interpolate`, `compensate`, `smooth`,
    `equalize`, `biquad_response`
  - Re-export all public types

**Tests:**
- Apply known filters to known FR, verify output matches expected corrected curve.

---

### Phase 7: Golden File Integration Tests

The critical validation phase. Every combination must pass.

**Tasks:**
- `tests/integration/golden.rs` (or `tests/golden.rs`):
  - Load each of 90 golden files (format: `{ iem, target, constraint, fs, pregain, filters }`,
    where filters use Rust-native field names: `filter_type`, `fc`, `gain`, `q`)
  - Run `optimize()` with same inputs → our `OptimizeResult`
  - **RMSE comparison method:**
    The measured FR cancels out algebraically (corrected = measured + filters + pregain;
    difference = (filters_g + pre_g) − (filters_o + pre_o)). No need to load the IEM FR.
    1. Compute golden filter cascade response + pregain on the optimizer grid (1.02 step, 20–20000 Hz)
    2. Compute our filter cascade response + pregain on the same grid
    3. RMSE of the difference, assert ≤ 0.5 dB
  - This compares equivalent corrected response curves, not raw params (params may differ while producing equivalent output)

**Test matrix:** 5 IEMs x 6 targets x 3 constraint sets = 90

| | blessing3 | hexa | andromeda | zero2 | origin_s |
|---|---|---|---|---|---|
| harman_ie_2019 | x | x | x | x | x |
| diffuse_field | x | x | x | x | x |
| flat | x | x | x | x | x |
| v_shaped | x | x | x | x | x |
| bass_heavy | x | x | x | x | x |
| bright | x | x | x | x | x |

**Constraint sets:**
- **Standard:** 5 filters (LSQ + PK x3 + HSQ), gain +/-12, Q 0.5-10
- **Restricted:** 3 filters (PK only), gain +/-6, Q 1.0-5.0
- **Qudelix-10:** 10 filters (LSQ + PK x8 + HSQ), gain +/-12, Q 0.5-10

**Per-parameter locking tests (new, no golden files):**
- Lock one PK band fully (fc+gain+Q fixed), optimize remaining
  - Assert: locked band params unchanged in output
  - Assert: RMSE <= unlocked baseline + 0.5 dB
- Lock only fc on a shelf filter, let gain and Q optimize
  - Assert: fc unchanged, gain and Q differ from init
- Lock only gain on a PK filter
  - Assert: gain unchanged, fc and Q optimized
- All bands fully locked -> output equals input (no optimization occurs)
- **peq.yaml pattern:** LSQ with `fc=Some(105), optimize_fc=false, q=Some(0.7), optimize_q=false`
  (gain-only), HSQ with `fc_range=(5000, 12000)` — verify shelves don't overlap and result is reasonable
- `min_std: Some(MinStd::Disabled)` — verify optimizer runs to 150 iterations, not early-stopped

---

### Phase 8: Performance + Benchmarks ✓ COMPLETE

**Tasks:**
- `benches/optimize_benchmark.rs`:
  - 5-band standard config: measure full `optimize()` latency
  - 10-band Qudelix config: measure full `optimize()` latency
  - Target: sub-100ms for 10-band on typical hardware
- Profile hot paths if needed (biquad eval in loss function is called O(iterations * params))

**Results (Apple M-series, release profile):**

| Config | Time |
|--------|------|
| 5-band standard (blessing3 + harman_ie_2019) | 17.8 ms |
| 10-band Qudelix (blessing3 + harman_ie_2019) | 82.8 ms |

Both under target. **Optimization applied:** fused `total_response` + `sharpness_penalty`
into a single per-filter pass in `joint_loss`, eliminating redundant `biquad_response`
calls for PK filters (~42% improvement on 10-band).

---

### Phase 9: Documentation + CI

**Tasks:**
- Rustdoc on all public types and functions
- `README.md` with:
  - Installation (Cargo dependency)
  - Quick start example
  - API reference summary
  - Algorithm & accuracy section (SLSQP parity with AutoEQ)
  - Per-parameter locking examples
- `CLAUDE.md` for Claude Code sessions
- `.github/workflows/ci.yml`: `cargo test`, `cargo clippy`, `cargo fmt --check`

---

### Phase 10: Revisit SLSQP-Parity Gap (deferred from Phase 7)

Phase 7 closed at 87/90 golden cases. The remaining 3 failures
(`hexa__diffuse_field__restricted` 2.29 dB, `origin_s__bass_heavy__qudelix_10`
0.66 dB, `zero2__bass_heavy__restricted` 0.51 dB) are SLSQP-implementation
divergence between `relf/slsqp` (Rust) and scipy's `fmin_slsqp` (Fortran via
Python). Init and correction curves are bit-identical to AutoEQ; the gap is
purely in the solver's local-minimum selection. See ARCHITECTURE.md "Known
Divergences" for details.

**Tasks (in order of effort, lowest first):**

1. **Solver tuning.** Inspect `relf/slsqp` for exposed knobs: finite-difference
   step size, line-search parameters, active-set tolerances. Try matching
   scipy's defaults exactly. Re-run the 3 failing cases. Cheap to try; may
   close 1–2 marginal cases (the 0.51 / 0.57 ones) but unlikely to help
   `hexa__diffuse_field__restricted`.

2. **Alternate Rust SLSQP.** Survey `crates.io` for other SLSQP / sequential
   QP implementations (e.g. `argmin`, `cobyla`, custom ports). Benchmark each
   against the 3 failing cases and against the existing 87 passing cases (no
   regressions). Decision criteria: pass rate, runtime, dependency cost.

3. **Reference oracle via PyO3.** Wrap scipy's `fmin_slsqp` through PyO3 and
   gate it behind a `cfg(test)` feature. Use it only as a per-case "what would
   AutoEQ do" oracle for diffing solver trajectories — not as a runtime
   dependency. Helps diagnose *where* the trajectories diverge (iteration
   count, intermediate gradients, active constraint flips).

4. **Cost-surface analysis.** For each of the 3 failing cases, plot the loss
   landscape near both solver's final solutions. If both are valid local
   minima of equivalent quality, the goldens may be over-specifying the test
   — consider relaxing the assertion to "RMSE ≤ 0.5 dB OR loss ≤
   golden_loss + ε" so equivalent-quality solutions pass.

5. **Decision point.** If after (1)–(4) we still have unresolved cases,
   document them as permanent known divergences and ship 87/90. The 0.5 dB
   threshold was the regression gate; in practice these are inaudible
   filter-cascade differences for an end user.

---

## Test Strategy Summary

| Level | What | Count (est.) |
|-------|------|-------------|
| Unit | Per-module correctness | ~80-100 |
| Integration | Golden file RMSE (90 combos) | 90 |
| Integration | Per-parameter locking | ~10-15 |
| Benchmark | Performance targets | 2-3 |

Total: ~185-210 tests.

---

## Key Decisions

| Decision | Rationale |
|----------|-----------|
| SLSQP via `relf/slsqp` | Same algorithm as AutoEQ. Eliminates L-BFGS divergence and pathological shelf inversions. |
| `FilterSpec` mirrors AutoEQ fields exactly | `fc/q/gain` = initial value, `optimize_fc/q/gain` = lock flag. Supports auto-init, seeded-free, and locked — all three states AutoEQ exposes. |
| f64 throughout | Matches AutoEQ's numpy float64. No f32, no generics. Precision matters in biquad phi-identity and SLSQP finite differences. |
| `MinStd` enum (not `Option<Option<f64>>`) | Three states (Default / Disabled / Custom) made self-documenting. Eliminates nested-Option footgun. |
| Signed shelf gain weighting | Matches AutoEQ (`dot(target, fr) / sum(fr)`). TypeScript used `abs(fr)` — a minor bug. |
| Configurable `min_std` | AutoEQ's own configs override this (e.g. `peq.yaml` sets `min_std: null`). Must be exposed, not hardcoded. |
| No `ndarray` / `realfft` / `num-traits` | AutoEQ uses none of these. Plain `Vec<f64>` is sufficient and keeps the dependency surface minimal. |
| Goldens regenerated from AutoEQ directly | Eliminates biquad-fit from correctness chain. Our `tests/generate_golden.py` imports AutoEQ, controls field names, and pins the oracle version. |
| Golden field names are Rust-native | Generator emits `filter_type`/`fc` matching our `Filter` struct — no serde renames on the hot path. |
| `compensate()` is a four-step pipeline | Interpolate target → center at 1 kHz (log-linear, off-grid) → add `create_target()` → subtract. Not a raw `measured - target`. [Gotcha A] |
| savgol edge mode = `'interp'` | scipy default fits a polynomial to the last `window_length` samples at each boundary. Most Rust savgol crates default to mirror/nearest/reflect — must verify or implement manually. Affects 20 Hz / 20 kHz extremes. [Gotcha J] |
| Separate repo | Clean start. No mixed-language build. TypeScript version remains for browser/npm. |
| serde for I/O | Standard Rust JSON. Also needed for eventual Tauri `compute_eq` command. |

---

## Future: eq-coach Integration

After this crate is complete and passing all tests, it will be wired into eq-coach as a
Tauri command:
1. Add `biquad-fit-rs` as a path dependency in eq-coach's `Cargo.toml`
2. Create Tauri command `compute_eq` that calls `optimize()`
3. Remove JS `biquad-fit` dependency from eq-coach frontend
4. Verify end-to-end: search IEM -> fetch FR -> compute EQ (Rust) -> write to device
