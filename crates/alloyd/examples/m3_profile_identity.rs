//! Print the native Metal tool profile identity used by the qualification runner.
use alloyd::{BootRequest, boot::inspect_qualification_identity};
use engine::{
    memory_guard::MemoryGuardConfig,
    tiered_kv::{HotCacheLimit, TieredKvConfig},
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let argv: Vec<_> = std::env::args().skip(1).collect();
    let [profile, artifact, context, candidate, dependencies] = argv.as_slice() else {
        return Err(
            "usage: m3_profile_identity PROFILE ARTIFACT CONTEXT CANDIDATE DEPENDENCIES_SHA256"
                .into(),
        );
    };
    let request = BootRequest {
        profile_id: profile.clone(),
        artifact_dir: artifact.into(),
        corpus_models_json: None,
        memory_guard: MemoryGuardConfig::balanced(),
        admission_budget_override_bytes: None,
        prefer_metal: true,
        tools_enabled: true,
        tiered_kv: TieredKvConfig::new(None, HotCacheLimit::Bytes(0)),
    };
    let (identity, profile) =
        inspect_qualification_identity(&request, Some(context.parse()?), candidate, dependencies)
            .map_err(|error| error.message)?;
    let result = serde_json::json!({
        "profile": profile.id().as_str(), "artifact": profile.artifact_id().as_str(),
        "context": profile.context_limit_tokens(), "output_reserve": profile.max_output_tokens(),
        "safety_margin": profile.safety_margin_tokens(), "identity": identity,
    });
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}
