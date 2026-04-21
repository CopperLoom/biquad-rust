# Implementation Plan — biquad-fit (Rust)

Rust crate reimplementing AutoEQ's parametric EQ optimizer with SLSQP and per-parameter
locking. Architecture and algorithm spec: `docs/ARCHITECTURE.md`.

---

## Phased Implementation

Each phase produces a testable increment. Tests are written alongside (not after)
implementation. A phase is not done until its tests pass.

---

### Phase 1: Project Scaffolding + Types

Set up the Rust project structure and define all shared types.

**Tasks:**
- `cargo init --lib` in project root
- Add dependencies to `Cargo.toml`:
  - `slsqp` — SLSQP optimizer
  - `serde` + `serde_json` — serialization
  - Dev: `approx` (float comparison), `criterion` (benchmarks)
- Create `src/types.rs`:
  - `FilterType` enum: `PK`, `LSQ`, `HSQ` (with serde rename for JSON compat)
  - `FreqPoint { freq: f64, db: f64 }`
  - `Filter { filter_type: FilterType, fc: f64, gain: f64, q: f64 }`
  - `FilterSpec`:
    ```rust
    pub struct FilterSpec {
        pub filter_type: Option<FilterType>,  // defaults to PK
        pub gain_range: (f64, f64),           // required
        pub q_range: Option<(f64, f64)>,      // defaults by type
        pub fc_range: Option<(f64, f64)>,     // defaults by type
        pub fc: Option<f64>,                  // Some = fixed, None = optimize
        pub q: Option<f64>,                   // Some = fixed, None = optimize
        pub gain: Option<f64>,                // Some = fixed, None = optimize
    }
    ```
  - `Constraints { filter_specs: Vec<FilterSpec>, freq_range: Option<(f64, f64)>, fs: Option<f64> }`
  - `OptimizeResult { pregain: f64, filters: Vec<Filter> }`
  - `InterpolateOptions { step: Option<f64>, f_min: Option<f64>, f_max: Option<f64> }`
- Create `src/lib.rs` with module declarations
- Copy test fixtures into `tests/fixtures/{fr,targets,golden}/` (see setup.txt)
- Create `tests/helpers/mod.rs` for golden file loading and RMSE computation

**Tests:** Types compile, golden files parse correctly, RMSE helper produces known values.

---

### Phase 2: Biquad Response + Interpolation

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
- Interpolate: grid point count matches TypeScript (461 for 1.01, ~350 for 1.02).
  Values at original measurement points preserved within tolerance.

---

### Phase 3: Smooth + Compensate

**Tasks:**
- `src/smooth.rs`:
  - `savgol_coeffs(window_size, poly_order) -> Vec<f64>` — Vandermonde + Gauss-Jordan
  - `savgol_filter(data, window_size) -> Vec<f64>` — convolution + edge polynomial fit
  - `smoothing_window_size(freqs, octaves) -> usize`
  - `log_f_sigmoid(f, f_lower, f_upper) -> f64`
  - `two_zone_smooth(fr, normal_octaves, treble_octaves) -> Vec<FreqPoint>`
  - `smooth(fr, window_octaves) -> Vec<FreqPoint>` — public single-zone API
- `src/compensate.rs`:
  - `compensate(measured, target) -> Vec<FreqPoint>` — element-wise subtraction

**Tests:**
- Savitzky-Golay: polynomial data (degree <= 2) passes through unchanged.
  Window size matches AutoEQ's `smoothing_window_size()` for 1.01 and 1.02 grids.
- Sigmoid: verify weights at 5 kHz (~0), 7 kHz (~0.5), 10 kHz (~1).
- Compensate: simple subtraction on matched grids.

---

### Phase 4: Equalize

The most complex single module. Full slope-limiting pipeline.

**Tasks:**
- `src/equalize.rs`:
  - `find_peaks(arr, min_prominence) -> Vec<usize>` — local maxima with prominence
  - `protection_mask(y, peak_inds, dip_inds) -> Vec<bool>`
  - `find_rtl_start(y, peak_inds, dip_inds) -> usize`
  - `limited_ltr_slope(freqs, y, max_slope, start_index, peak_inds, limit_free_mask) -> Vec<f64>`
    — with region validation
  - `limited_rtl_slope(...)` — flip, run LTR, flip back
  - `equalize(error) -> Vec<FreqPoint>` — full pipeline

**Tests:**
- Flat error -> equalization is the negation.
- Known step error -> slope capped at 18 dB/oct.
- Peak finding: compare against scipy `find_peaks` output on fixture data.
- Full equalize: compare output against TypeScript for each test fixture (should match
  exactly — same algorithm).

---

### Phase 5: Optimizer Core (SLSQP)

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

**SLSQP integration notes:**
- Check `relf/slsqp` crate API — may provide own finite-difference gradient, or may
  require explicit gradient function. If explicit: forward differences with h = sqrt(f64::EPSILON).
- Convergence via callback matching AutoEQ's `_callback` logic.
- Max iterations: 150.

**Per-parameter locking implementation:**
1. `FilterSpec { fc: Some(8000.0), .. }` -> `optimize_fc = false`, fc fixed at 8000
2. `encode_params` skips fixed params -> shorter optimization vector
3. `build_bounds` skips fixed params -> matching shorter bounds vector
4. `joint_loss` includes all filters in cascade (fixed and free)
5. `decode_params` splices fixed values back into full filter list
6. Init sequence: fixed-param filters still initialized (or use provided values)
   and subtracted from remaining target before initializing free-param filters

**Tests:**
- Loss: known filter set + target -> verify MSE matches hand calculation.
- Init: compare params against TypeScript output for fixtures.
- Encoding round-trip: encode -> decode is identity.
- Locking: locked params unchanged after optimization.

---

### Phase 6: Apply Filters + Public API

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
  - Load each of 90 golden files
  - Run `optimize()` with same inputs
  - Compare output via RMSE
  - Assert RMSE <= 0.5 dB per combination

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

---

### Phase 8: Performance + Benchmarks

**Tasks:**
- `benches/optimize_benchmark.rs`:
  - 5-band standard config: measure full `optimize()` latency
  - 10-band Qudelix config: measure full `optimize()` latency
  - Target: sub-100ms for 10-band on typical hardware
- Profile hot paths if needed (biquad eval in loss function is called O(iterations * params))

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
| Per-parameter locking | Matches AutoEQ's `optimize_fc`/`optimize_q`/`optimize_gain`. Required for fixed-band EQ and eq-coach locked-band feature. |
| Signed shelf gain weighting | Matches AutoEQ (`dot(target, fr) / sum(fr)`). TypeScript used `abs(fr)` — a minor bug. |
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
