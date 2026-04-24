use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum FilterType {
    #[serde(rename = "PK")]
    PK,
    #[serde(rename = "LSQ")]
    LSQ,
    #[serde(rename = "HSQ")]
    HSQ,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct FreqPoint {
    pub freq: f64,
    pub db: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Filter {
    pub filter_type: FilterType,
    pub fc: f64,
    pub gain: f64,
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

#[derive(Debug, Clone)]
pub struct Constraints {
    pub filter_specs: Vec<FilterSpec>,
    /// Frequency range for filter init (default [20, 10000]). Note: loss always uses [20, 20000].
    pub freq_range: Option<(f64, f64)>,
    /// Sample rate (default 44100)
    pub fs: Option<f64>,
    pub min_std: Option<MinStd>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizeResult {
    pub pregain: f64,
    pub filters: Vec<Filter>,
}

#[derive(Debug, Clone)]
pub struct InterpolateOptions {
    pub step: Option<f64>,
    pub f_min: Option<f64>,
    pub f_max: Option<f64>,
}

#[derive(Debug, Clone)]
pub enum BiquadError {
    InvalidFilterSpec(String),
    InvalidFrequencyResponse(String),
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
