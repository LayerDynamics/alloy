//! Native Llama entry point over the shared dense adapter.
pub use super::qwen::{
    Qwen3Artifact as Llama3Artifact, Qwen3BackendLimits as Llama3BackendLimits,
    Qwen3Execution as Llama3Execution,
};
/// Construct the Llama backend with explicit native placement and resource limits.
pub fn backend(
    artifacts: Vec<Llama3Artifact>,
    limits: Llama3BackendLimits,
    execution: Llama3Execution,
) -> Result<super::qwen::Qwen3Backend, crate::error::EngineError> {
    super::qwen::Qwen3Backend::new_llama3(artifacts, limits, execution)
}
