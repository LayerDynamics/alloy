//! Gemma 3 text equations and resident mixed sliding/full attention.
//! Native core validation is separate from profile/agent qualification.
pub mod config;
pub mod model;
pub mod planning;
pub mod weights;
pub use config::Gemma3Config;
