//! Rust reimplementation of AutoEQ's parametric EQ optimizer.
//!
//! Computes optimal biquad filter parameters (frequency, gain, Q) to match a measured
//! IEM frequency response to a target curve. Faithful port of AutoEQ's pipeline and
//! SLSQP optimizer — results match within 0.5 dB RMSE across the 90-case golden matrix.
//!
//! # Quick start
//!
//! ```no_run
//! use biquad_rust::{optimize, FreqPoint, FilterSpec, FilterType, Constraints};
//!
//! let measured: Vec<FreqPoint> = vec![/* IEM measurements */];
//! let target: Vec<FreqPoint>   = vec![/* target curve    */];
//!
//! fn pk(gain_range: (f64, f64)) -> FilterSpec {
//!     FilterSpec {
//!         filter_type: None, fc: None, q: None, gain: None,
//!         optimize_fc: None, optimize_q: None, optimize_gain: None,
//!         fc_range: None, q_range: None, gain_range,
//!     }
//! }
//!
//! let constraints = Constraints {
//!     filter_specs: vec![
//!         FilterSpec { filter_type: Some(FilterType::LSQ), ..pk((-12.0, 12.0)) },
//!         pk((-12.0, 12.0)),
//!         pk((-12.0, 12.0)),
//!         pk((-12.0, 12.0)),
//!         FilterSpec { filter_type: Some(FilterType::HSQ), ..pk((-12.0, 12.0)) },
//!     ],
//!     freq_range: None,
//!     fs: None,
//!     min_std: None,
//! };
//!
//! let result = optimize(&measured, &target, &constraints).unwrap();
//! println!("pregain: {} dB", result.pregain);
//! for f in &result.filters {
//!     println!("{:?} fc={:.0} Hz gain={:.2} dB Q={:.2}", f.filter_type, f.fc, f.gain, f.q);
//! }
//! ```

pub mod apply_filters;
pub mod biquad_response;
pub mod compensate;
pub mod equalize;
pub mod interpolate;
pub mod optimize;
pub mod peak_finding;
pub mod smooth;
pub mod types;

pub use apply_filters::apply_filters;
pub use biquad_response::biquad_response;
pub use compensate::{center, compensate};
pub use equalize::equalize;
pub use interpolate::interpolate;
pub use optimize::optimize;
pub use smooth::{smooth, two_zone_smooth};
pub use types::{
    BiquadError, Constraints, Filter, FilterSpec, FilterType, FreqPoint, InterpolateOptions,
    MinStd, OptimizeResult,
};
