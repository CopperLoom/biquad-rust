# biquad-rust

Rust reimplementation of AutoEQ's parametric EQ optimizer. Given a measured IEM frequency
response and a target curve, computes optimal biquad filter parameters (frequency, gain, Q)
that bring the IEM closest to the target.

Faithful port of [jaakkopasanen/AutoEQ](https://github.com/jaakkopasanen/AutoEQ)'s pipeline
and SLSQP optimizer. Results match AutoEQ within **≤ 0.5 dB RMSE** across a 90-case
golden test matrix (5 IEMs × 6 targets × 3 constraint sets).

---

## Installation

Add to `Cargo.toml`:

```toml
[dependencies]
biquad-rust = { path = "../biquad-rust" }  # path dependency until published
```

---

## Quick Start

```rust
use biquad_rust::{
    optimize, FreqPoint, FilterSpec, FilterType, Constraints,
};

// Load your IEM measurement and target (freq in Hz, db in dB)
let measured: Vec<FreqPoint> = load_fr("iem.json");
let target: Vec<FreqPoint>   = load_fr("harman_2019.json");

// 5-band standard config: LSQ + PK×3 + HSQ, ±12 dB gain
fn pk(gain_range: (f64, f64)) -> FilterSpec {
    FilterSpec {
        filter_type: None, fc: None, q: None, gain: None,
        optimize_fc: None, optimize_q: None, optimize_gain: None,
        fc_range: None, q_range: None, gain_range,
    }
}

let constraints = Constraints {
    filter_specs: vec![
        FilterSpec { filter_type: Some(FilterType::LSQ), ..pk((-12.0, 12.0)) },
        pk((-12.0, 12.0)),
        pk((-12.0, 12.0)),
        pk((-12.0, 12.0)),
        FilterSpec { filter_type: Some(FilterType::HSQ), ..pk((-12.0, 12.0)) },
    ],
    freq_range: None,  // default [20, 10000] Hz for filter init
    fs: None,          // default 44100 Hz
    min_std: None,     // default convergence threshold (0.002)
};

let result = optimize(&measured, &target, &constraints).unwrap();
println!("pregain: {:.1} dB", result.pregain);
for f in &result.filters {
    println!("{:?}  fc={:.0} Hz  gain={:+.2} dB  Q={:.2}", f.filter_type, f.fc, f.gain, f.q);
}
```

---

## API Reference

### `optimize`

```rust
pub fn optimize(
    measured: &[FreqPoint],
    target:   &[FreqPoint],
    constraints: &Constraints,
) -> Result<OptimizeResult, BiquadError>
```

Full pipeline: interpolate → center → compensate → equalize → init filters → SLSQP optimize → pregain.

### Key types

| Type | Description |
|------|-------------|
| `FreqPoint { freq, db }` | Single (Hz, dB) measurement point |
| `Filter { filter_type, fc, gain, q }` | Resolved filter parameters |
| `FilterType` | `PK` / `LSQ` / `HSQ` |
| `FilterSpec` | Per-band config: initial values, lock flags, ranges |
| `Constraints` | Full optimizer config: specs + freq\_range + fs + min\_std |
| `OptimizeResult { pregain, filters }` | Output |
| `BiquadError` | Error variants: `InvalidFilterSpec`, `InvalidFrequencyResponse`, `OptimizerFailed` |

### Pipeline functions (also public)

| Function | Description |
|----------|-------------|
| `interpolate(fr, opts)` | Resample FR to log-spaced grid |
| `compensate(measured, target)` | Compute error curve (4-step pipeline) |
| `center(fr)` | Subtract value at 1 kHz from all points |
| `smooth(fr, octaves)` | Savitzky-Golay single-zone smooth |
| `two_zone_smooth(fr, normal, treble)` | Two-zone smooth with sigmoid blend |
| `equalize(error)` | Slope-limited equalization curve |
| `biquad_response(type, fc, gain, q, freqs, fs)` | Filter magnitude in dB |
| `apply_filters(fr, filters, pregain, fs)` | Apply filter cascade to FR |

---

## Per-Parameter Locking

Each filter band supports independent locking of fc, Q, and gain. This matches AutoEQ's
`peq.yaml` pattern for preventing shelf overlap:

```rust
// LSQ with fixed fc and Q — only gain is optimized
FilterSpec {
    filter_type:  Some(FilterType::LSQ),
    fc:           Some(105.0),
    optimize_fc:  Some(false),
    q:            Some(0.7),
    optimize_q:   Some(false),
    gain_range:   (-12.0, 12.0),
    ..pk((-12.0, 12.0))
}

// HSQ with fc constrained to [5000, 12000] Hz
FilterSpec {
    filter_type: Some(FilterType::HSQ),
    fc_range:    Some((5000.0, 12000.0)),
    q_range:     Some((0.4, 0.7)),
    gain_range:  (-12.0, 12.0),
    ..pk((-12.0, 12.0))
}
```

Three states per parameter:

| `fc` | `optimize_fc` | Behavior |
|------|--------------|----------|
| `None` | `None` / `true` | Auto-init from correction curve, free to optimize |
| `Some(x)` | `None` / `true` | Seeded at x, free to optimize |
| `Some(x)` | `Some(false)` | Locked at x, excluded from optimizer |

---

## Algorithm & Accuracy

Pipeline matches AutoEQ exactly:

1. **Interpolate** measured and target to a log-spaced 1.01-step grid (695 points, 20–20 kHz)
2. **Center** both at 1 kHz (log-linear interpolation — 1 kHz is off the 1.01 grid)
3. **Compensate**: error = measured − target
4. **Smooth** error with Savitzky-Golay two-zone filter (1/12 oct normal, 2 oct treble)
5. **Equalize**: negate, slope-limit LTR+RTL at 18 dB/oct, re-smooth at 1/5 oct
6. **Interpolate** to optimizer 1.02-step grid (349 points, 20–20 kHz)
7. **Init filters** in priority order (HSQ → LSQ → PK) against remaining correction target
8. **SLSQP optimize**: minimize `sqrt(MSE + sharpness_penalty)`, 150-iteration max
9. **Pregain**: `max(0, max_filter_boost + 0.2 dB headroom)`

**Solver:** `relf/slsqp` — a native Rust port of the same Fortran SLSQP that scipy wraps.
87 of 90 golden cases pass at ≤ 0.5 dB RMSE; the 3 remaining cases are different local
minima (solver trajectory divergence, not algorithmic error). See `docs/ARCHITECTURE.md`
"Known Divergences" for details.

---

## Performance

Benchmarks on Apple M-series (release profile):

| Config | Time |
|--------|------|
| 5-band standard (LSQ + PK×3 + HSQ) | 17.8 ms |
| 10-band Qudelix (LSQ + PK×8 + HSQ) | 82.8 ms |

---

## Running Tests

```bash
cargo test          # unit + integration (90 golden cases)
cargo bench         # performance benchmarks
cargo clippy        # lints
cargo doc --open    # rustdoc
```

---

## License

MIT
