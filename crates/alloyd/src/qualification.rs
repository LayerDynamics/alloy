//! Distribution-bound native qualification. Only the packaging build supplies the trust root.
use crate::rpc::RpcError;
use engine::conformance::{QualificationReport, QualificationSpec};
use serde::{Deserialize, de::DeserializeOwned};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs::File, io::Read, path::Path, process::Command};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Catalog {
    version: u32,
    candidate: String,
    host: Host,
    metal_library_sha256: String,
    profiles: BTreeMap<String, Entry>,
}

#[derive(Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Host {
    model: String,
    os_build: String,
    memory_bytes: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Entry {
    context: usize,
    dependencies_sha256: String,
    spec_sha256: String,
    report_sha256: String,
}

pub(crate) struct PackageProfile {
    pub context: usize,
    pub spec: QualificationSpec,
    pub report: QualificationReport,
}

fn invalid(message: impl std::fmt::Display) -> RpcError {
    RpcError::invalid_params(format!(
        "native profile is not qualified for production: {message}"
    ))
}

fn read_pinned<T: DeserializeOwned>(path: &Path, digest: &str, limit: u64) -> Result<T, RpcError> {
    let file = File::open(path).map_err(invalid)?;
    let meta = file.metadata().map_err(invalid)?;
    if !meta.is_file() || meta.len() > limit {
        return Err(invalid("invalid qualification file size or type"));
    }
    let mut data = Vec::new();
    file.take(limit + 1)
        .read_to_end(&mut data)
        .map_err(invalid)?;
    if data.len() as u64 > limit || format!("{:x}", Sha256::digest(&data)) != digest {
        return Err(invalid(format!(
            "qualification digest mismatch for {}",
            path.display()
        )));
    }
    serde_json::from_slice(&data).map_err(invalid)
}

fn verify_kernel(path: &Path, expected: &str) -> Result<(), RpcError> {
    let file = File::open(path).map_err(invalid)?;
    let meta = file.metadata().map_err(invalid)?;
    if !meta.is_file() || meta.len() > 512 << 20 {
        return Err(invalid("invalid packaged Metal library"));
    }
    let mut digest = Sha256::new();
    let bytes = std::io::copy(&mut file.take((512 << 20) + 1), &mut digest).map_err(invalid)?;
    if bytes != meta.len() || format!("{:x}", digest.finalize()) != expected {
        return Err(invalid("packaged Metal library changed"));
    }
    Ok(())
}

fn sysctl(name: &str) -> Result<String, RpcError> {
    let value = Command::new("/usr/sbin/sysctl")
        .args(["-n", name])
        .output()
        .map_err(invalid)?;
    if !value.status.success() {
        return Err(invalid(format!("cannot determine {name}")));
    }
    let value = String::from_utf8(value.stdout).map_err(invalid)?;
    Ok(value.trim().to_owned())
}

impl Catalog {
    fn profile(
        &self,
        candidate: &str,
        host: &Host,
        profile: &str,
        requested: Option<usize>,
    ) -> Result<&Entry, RpcError> {
        if self.version != 1 || self.candidate != candidate || self.host != *host {
            return Err(invalid("qualification source or host changed"));
        }
        let entry = self
            .profiles
            .get(profile)
            .ok_or_else(|| invalid(format!("no packaged qualification for {profile}")))?;
        if requested.is_some_and(|context| context != entry.context) {
            return Err(invalid("requested context differs from measured context"));
        }
        Ok(entry)
    }
}

pub(crate) fn packaged(
    profile: &str,
    requested: Option<usize>,
) -> Result<Option<PackageProfile>, RpcError> {
    let Some(digest) = option_env!("ALLOY_M3_RELEASE_CATALOG_SHA256") else {
        return Ok(None);
    };
    let candidate = option_env!("ALLOY_M3_RELEASE_REVISION")
        .ok_or_else(|| invalid("missing build revision"))?;
    let exe = std::env::current_exe().map_err(invalid)?;
    let root = exe
        .parent()
        .ok_or_else(|| invalid("daemon location unavailable"))?
        .join("qualification");
    let catalog: Catalog = read_pinned(&root.join("catalog.json"), digest, 1 << 20)?;
    let host = Host {
        model: sysctl("hw.model")?,
        os_build: sysctl("kern.osversion")?,
        memory_bytes: sysctl("hw.memsize")?.parse().map_err(invalid)?,
    };
    verify_kernel(
        &exe.with_file_name("mlx.metallib"),
        &catalog.metal_library_sha256,
    )?;
    let entry = catalog.profile(candidate, &host, profile, requested)?;
    // BootRequest has already validated the profile ID against the registered IDs.
    let directory = root.join(profile);
    let spec: QualificationSpec =
        read_pinned(&directory.join("spec.json"), &entry.spec_sha256, 128 << 20)?;
    let report = read_pinned(
        &directory.join("report.json"),
        &entry.report_sha256,
        128 << 20,
    )?;
    if spec.profile_id != profile
        || spec.artifact_id != profile
        || spec.identity.candidate_revision != candidate
        || spec.identity.dependencies_sha256 != entry.dependencies_sha256
    {
        return Err(invalid(
            "packaged suite does not match the build and selected profile",
        ));
    }
    Ok(Some(PackageProfile {
        context: entry.context,
        spec,
        report,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;
    #[test]
    fn packaged_files_require_build_digest() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("report.json");
        let good = br#"{"value":1}"#;
        std::fs::write(&path, good).unwrap();
        let digest = format!("{:x}", Sha256::digest(good));
        assert_eq!(
            read_pinned::<Value>(&path, &digest, 16).unwrap()["value"],
            1
        );
        assert!(read_pinned::<Value>(&path, &digest, 10).is_err());
        std::fs::write(&path, br#"{"value":2}"#).unwrap();
        assert!(read_pinned::<Value>(&path, &digest, 16).is_err());
    }

    #[test]
    fn packaged_profile_rejects_changed_context_and_host() {
        let catalog:Catalog=serde_json::from_value(serde_json::json!({
            "version":1,"candidate":"build","metal_library_sha256":"kernels","host":{"model":"Mac","os_build":"A","memory_bytes":36u64<<30},
            "profiles":{"model":{"context":40960,"spec_sha256":"spec","report_sha256":"report","dependencies_sha256":"deps"}}
        })).unwrap();
        assert_eq!(
            catalog
                .profile("build", &catalog.host, "model", None)
                .unwrap()
                .context,
            40960
        );
        assert!(
            catalog
                .profile("build", &catalog.host, "model", Some(2048))
                .is_err()
        );
        assert!(
            catalog
                .profile("new", &catalog.host, "model", None)
                .is_err()
        );
        assert!(
            catalog
                .profile("build", &catalog.host, "other", None)
                .is_err()
        );
        let different = Host {
            model: "Mac".into(),
            os_build: "B".into(),
            memory_bytes: 36 << 30,
        };
        assert!(catalog.profile("build", &different, "model", None).is_err());
    }
}
