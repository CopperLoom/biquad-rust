use serde::{Deserialize, Serialize};

/// Biquad filter topology. Matches AutoEQ's `PEQFilter` subclasses.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum FilterType {
    /// Peaking EQ (bell) filter.
    #[serde(rename = "PK")]
    PK,
    /// Low-shelf filter.
    #[serde(rename = "LSQ")]
    LSQ,
    /// High-shelf filter.
    #[serde(rename = "HSQ")]
    HSQ,
}

/// A single (frequency, dB) measurement point.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FreqPoint {
    /// Frequency in Hz.
    pub freq: f64,
    /// Amplitude in dB.
    pub db: f64,
}

/// A fully-resolved biquad filter with concrete parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Filter {
    pub filter_type: FilterType,
    /// Center / shelf frequency in Hz.
    pub fc: f64,
    /// Gain in dB.
    pub gain: f64,
    /// Quality factor.
    pub q: f64,
}

/// Per-filter optimization spec. Mirrors AutoEQ's per-filter config fields.
///
/// Three states per parameter:
/// - `fc=None, optimize_fc=None/true`  → auto-init from correction curve, free to optimize
/// - `fc=Some(x), optimize_fc=None/true` → seeded at x, free to optimize
/// - `fc=Some(x), optimize_fc=Some(false)` → locked at x (error if fc=None + optimize_fc=false)
#[derive(Debug, Clone)]
pub struct FilterSpec {
    pub filter_type: Option<FilterType>,
    pub fc: Option<f64>,
    pub q: Option<f64>,
    pub gain: Option<f64>,
    pub optimize_fc: Option<bool>,
    pub optimize_q: Option<bool>,
    pub optimize_gain: Option<bool>,
    pub fc_range: Option<(f64, f64)>,
    pub q_range: Option<(f64, f64)>,
    pub gain_range: (f64, f64),
}

/// Convergence threshold for the SLSQP optimizer.
#[derive(Debug, Clone)]
pub enum MinStd {
    /// 0.002 — matches DEFAULT_PEQ_OPTIMIZER_MIN_STD
    Default,
    /// Run to 150 iterations (AutoEQ peq.yaml `min_std: null`)
    Disabled,
    /// Caller-specified threshold
    Custom(f64),
}

/// Full optimizer configuration: which filters to fit and how.
#[derive(Debug, Clone)]
pub struct Constraints {
    /// One entry per filter band, in any order.
    pub filter_specs: Vec<FilterSpec>,
    /// Frequency range for filter initialization (default [20, 10000] Hz).
    /// The loss function always evaluates [20, 20000] Hz regardless of this setting.
    pub freq_range: Option<(f64, f64)>,
    /// Sample rate in Hz (default 44100).
    pub fs: Option<f64>,
    /// Convergence criterion (default [`MinStd::Default`]).
    pub min_std: Option<MinStd>,
}

/// Output of [`optimize`](crate::optimize): pregain + filter parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizeResult {
    /// Pregain in dB. ≤ 0 (volume reduction to prevent clipping after boost filters).
    pub pregain: f64,
    /// Optimized filter parameters, in the same order as the input `filter_specs`.
    pub filters: Vec<Filter>,
}

/// Options for [`interpolate`](crate::interpolate).
#[derive(Debug, Clone)]
pub struct InterpolateOptions {
    /// Multiplicative step per grid point (default 1.01).
    pub step: Option<f64>,
    /// Grid start in Hz (default 20).
    pub f_min: Option<f64>,
    /// Grid end in Hz (default 20000).
    pub f_max: Option<f64>,
}

/// Errors returned by the public API.
#[derive(Debug, Clone)]
pub enum BiquadError {
    /// A [`FilterSpec`] is internally inconsistent (e.g. `fc=None` with `optimize_fc=false`).
    InvalidFilterSpec(String),
    /// The frequency response has too few points or contains NaN.
    InvalidFrequencyResponse(String),
    /// The SLSQP optimizer failed to converge.
    OptimizerFailed(String),
}

impl std::fmt::Display for BiquadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BiquadError::InvalidFilterSpec(s) => write!(f, "InvalidFilterSpec: {s}"),
            BiquadError::InvalidFrequencyResponse(s) => write!(f, "InvalidFrequencyResponse: {s}"),
            BiquadError::OptimizerFailed(s) => write!(f, "OptimizerFailed: {s}"),
        }
    }
}

impl std::error::Error for BiquadError {}
