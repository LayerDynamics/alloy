//! Identity-bound M3 profile qualification.
//!
//! A trusted native conformance runner supplies observations; this module does
//! not turn caller-authored pass flags into proof of execution. Reference vectors
//! and the required fixture set come from the trusted suite, never the report.
//! Failed reports remain serializable and can replace a prior passing result.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

/// Version of the report format and numerical acceptance rule.
pub const REPORT_VERSION: u32 = 1;

/// All inputs whose changes invalidate a qualification result.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QualificationIdentity {
    /// Reviewed source revision (40- or 64-character hexadecimal commit ID).
    pub candidate_revision: String,
    /// Digest covering the lockfile, MLX/mlx-c pins and execution environment.
    pub dependencies_sha256: String,
    /// Digest of the trusted, verified artifact file manifest.
    pub artifact_manifest_sha256: String,
    /// Actual registered architecture implementation.
    pub family: String,
    /// Family implementation version, independent of the package version.
    pub family_version: String,
    /// Public engine contract version.
    pub engine_api_version: u32,
    /// Exact consumed tokenizer bytes.
    pub tokenizer_sha256: String,
    /// Exact compiled text template.
    pub template_sha256: String,
    /// Exact compiled tool template and parser contract.
    pub tool_protocol_sha256: String,
    /// Profile settings, including validated context, reserves and decoding.
    pub profile_sha256: String,
}

/// Mandatory independent behaviors in the M3 qualification suite.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FixtureKind {
    /// Tool syntax, argument schema and reasoning separation.
    ToolSchema,
    /// End-of-turn and configured stop behavior.
    Stops,
    /// Counted and generated input identity.
    ExactCounting,
    /// The advertised validated context length.
    LongContext,
    /// Real overflow, valid checkpoint, continuation and event preservation.
    Compaction,
    /// Cached and fresh numerical/token equivalence, including invalidation.
    CachedUncached,
    /// Incremental output through the daemon/harness.
    Streaming,
    /// Cancellation with subsequent successful execution.
    Cancellation,
    /// Durable restart, including the advertised cache/state modes.
    Restart,
    /// Measured weights/cache/temporary memory and fit or refusal at 8 GiB.
    Memory8GiB,
    /// Measured weights/cache/temporary memory on the real 36 GiB host.
    Memory36GiB,
    /// Real native policy-gated local tool task with verified effect.
    PolicyGatedTask,
}

impl FixtureKind {
    /// Required M3 coverage; later capabilities add fixtures separately.
    pub const REQUIRED: [Self; 12] = [
        Self::ToolSchema,
        Self::Stops,
        Self::ExactCounting,
        Self::LongContext,
        Self::Compaction,
        Self::CachedUncached,
        Self::Streaming,
        Self::Cancellation,
        Self::Restart,
        Self::Memory8GiB,
        Self::Memory36GiB,
        Self::PolicyGatedTask,
    ];
}

/// Trusted definition of a fixed prompt and its eager reference logits.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NumericalReference {
    /// Digest of exact prompt/token input and reference-capture provenance.
    pub input_sha256: String,
    /// Fixed eager reference values, in vocabulary/position order.
    pub logits: Vec<f64>,
}

/// Trusted suite specification. A report cannot reduce this fixture set.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QualificationSpec {
    /// Profile identifier in the registry.
    pub profile_id: String,
    /// Artifact identifier in the registry.
    pub artifact_id: String,
    /// Expected identity from the actual candidate and verified inputs.
    pub identity: QualificationIdentity,
    /// Named must-pass fixtures and their coverage categories.
    pub fixtures: BTreeMap<String, FixtureKind>,
    /// Named fixed numerical prompts; at least one is required.
    pub references: BTreeMap<String, NumericalReference>,
}

/// One actual native numerical observation. No supplied metrics/pass flag.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NumericalObservation {
    /// Name in the trusted reference suite.
    pub fixture_id: String,
    /// Exact input identity must match the reference.
    pub input_sha256: String,
    /// Actual native logits in the reference ordering.
    pub logits: Vec<f64>,
}

/// Observation produced by a trusted fixture executor.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FixtureObservation {
    /// Name in the trusted suite.
    pub fixture_id: String,
    /// Terminal fixture assertion result (skips are not passes).
    pub passed: bool,
    /// Digest of retained native execution evidence.
    pub evidence_sha256: String,
    /// Concrete observed outcome or failure reason.
    pub detail: String,
}

/// Machine-readable native qualification observations, including failures.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QualificationReport {
    /// Must equal [`REPORT_VERSION`].
    pub version: u32,
    /// Actual identity of the candidate and inputs that were executed.
    pub identity: QualificationIdentity,
    /// Digest of the entire trusted suite, including fixed reference vectors.
    pub suite_sha256: String,
    /// Actual numerical observations; no missing, duplicate or extra entries.
    pub numerical: Vec<NumericalObservation>,
    /// Actual fixture observations; no missing, duplicate or extra entries.
    pub fixtures: Vec<FixtureObservation>,
}

/// Recomputed FR-8 numerical measurements.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NumericalMetrics {
    /// Largest absolute error across every observed logit.
    pub max_absolute: f64,
    /// L2 error divided by reference L2 norm (zero only for exact zero matches).
    pub relative_l2: f64,
    /// Largest pointwise relative error; absent for a nonzero error at zero.
    pub max_pointwise_relative: Option<f64>,
}

impl NumericalMetrics {
    /// Compute from paired vectors. Invalid shapes/nonfinite values fail closed.
    /// Both norms use hypot to avoid intermediate squared-value overflow.
    pub fn compare(reference: &[f64], actual: &[f64]) -> Result<Self, String> {
        if reference.is_empty() || reference.len() != actual.len() {
            return Err("numerical vectors must be nonempty and have identical lengths".into());
        }
        let mut max_absolute = 0.0_f64;
        let mut error_norm = 0.0_f64;
        let mut reference_norm = 0.0_f64;
        let mut pointwise = Some(0.0_f64);
        for (&expected, &observed) in reference.iter().zip(actual) {
            let error = (observed - expected).abs();
            if !expected.is_finite() || !observed.is_finite() || !error.is_finite() {
                return Err("nonfinite reference, observation or error".into());
            }
            max_absolute = max_absolute.max(error);
            error_norm = error_norm.hypot(error);
            reference_norm = reference_norm.hypot(expected);
            let ratio = if expected == 0.0 {
                if error == 0.0 { Some(0.0) } else { None }
            } else {
                let ratio = error / expected.abs();
                ratio.is_finite().then_some(ratio)
            };
            pointwise = pointwise.zip(ratio).map(|(a, b)| a.max(b));
        }
        let relative_l2 = if reference_norm == 0.0 && error_norm == 0.0 {
            0.0
        } else {
            error_norm / reference_norm
        };
        if !reference_norm.is_finite() || !error_norm.is_finite() || !relative_l2.is_finite() {
            return Err("undefined or unrepresentable relative numerical error".into());
        }
        Ok(Self {
            max_absolute,
            relative_l2,
            max_pointwise_relative: pointwise,
        })
    }

    /// Strict independent FR-8 limits; never an additive tolerance.
    pub fn passes(&self) -> bool {
        self.max_absolute.is_finite()
            && self.relative_l2.is_finite()
            && self.max_absolute >= 0.0
            && self.relative_l2 >= 0.0
            && self.max_absolute < 0.02
            && self
                .max_pointwise_relative
                .is_some_and(|relative| relative.is_finite() && (0.0..0.01).contains(&relative))
    }
}

/// Evaluation output suitable for retaining beside a failed or passing report.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QualificationVerdict {
    /// True only when identity, all numerical checks and all fixtures pass.
    pub usable: bool,
    /// Every observed rejection; preserved even when other fixtures pass.
    pub failures: Vec<String>,
    /// Recomputed measurements for valid numerical observations.
    pub numerical: BTreeMap<String, NumericalMetrics>,
}

impl QualificationSpec {
    /// Validate the trusted suite and hash its deterministic JSON representation.
    pub fn sha256(&self) -> Result<String, String> {
        if self.profile_id.is_empty()
            || self.artifact_id.is_empty()
            || self.identity.family.is_empty()
            || self.identity.family_version.is_empty()
            || self.identity.engine_api_version != crate::api::ENGINE_API_VERSION
        {
            return Err(
                "missing profile/artifact/family identity or incompatible engine API".into(),
            );
        }
        let revision = &self.identity.candidate_revision;
        if !matches!(revision.len(), 40 | 64) || !is_hex(revision) {
            return Err("candidate must be an immutable hexadecimal source revision".into());
        }
        for digest in [
            &self.identity.dependencies_sha256,
            &self.identity.artifact_manifest_sha256,
            &self.identity.tokenizer_sha256,
            &self.identity.template_sha256,
            &self.identity.tool_protocol_sha256,
            &self.identity.profile_sha256,
        ] {
            if !is_digest(digest) {
                return Err("identity requires SHA-256 digests".into());
            }
        }
        let coverage: BTreeSet<_> = self.fixtures.values().copied().collect();
        if FixtureKind::REQUIRED
            .iter()
            .any(|kind| !coverage.contains(kind))
        {
            return Err("suite omits mandatory M3 fixture coverage".into());
        }
        if self.fixtures.keys().any(|id| id.is_empty()) || self.references.is_empty() {
            return Err("suite requires named fixtures and fixed numerical references".into());
        }
        for (id, reference) in &self.references {
            if id.is_empty()
                || !is_digest(&reference.input_sha256)
                || reference.logits.is_empty()
                || reference.logits.iter().any(|v| !v.is_finite())
            {
                return Err("invalid fixed numerical reference".into());
            }
        }
        let bytes = serde_json::to_vec(self).map_err(|e| e.to_string())?;
        Ok(format!("{:x}", Sha256::digest(bytes)))
    }

    /// Re-evaluate a report against current trusted expectations. Never reuse a
    /// saved `usable` field; a failure must revoke any previous qualification.
    pub fn evaluate(&self, report: &QualificationReport) -> QualificationVerdict {
        let mut failures = Vec::new();
        match self.sha256() {
            Ok(digest) if digest == report.suite_sha256 => {}
            Ok(_) => failures.push("qualification suite identity changed".into()),
            Err(error) => failures.push(error),
        }
        if report.version != REPORT_VERSION {
            failures.push("unsupported report version".into());
        }
        if report.identity != self.identity {
            failures.push("qualification input identity changed".into());
        }
        let mut seen = BTreeSet::new();
        for observation in &report.fixtures {
            let id = &observation.fixture_id;
            if !seen.insert(id.clone()) || !self.fixtures.contains_key(id) {
                failures.push(format!("duplicate or unexpected fixture: {id}"));
            }
            if !observation.passed
                || !is_digest(&observation.evidence_sha256)
                || observation.detail.trim().is_empty()
            {
                failures.push(format!(
                    "failed or unevidenced fixture: {id}: {}",
                    observation.detail
                ));
            }
        }
        for id in self.fixtures.keys().filter(|id| !seen.contains(*id)) {
            failures.push(format!("missing fixture: {id}"));
        }
        let mut numerical = BTreeMap::new();
        seen.clear();
        for observation in &report.numerical {
            let id = &observation.fixture_id;
            if !seen.insert(id.clone()) {
                failures.push(format!("duplicate numerical fixture: {id}"));
            }
            let Some(reference) = self.references.get(id) else {
                failures.push(format!("unexpected numerical fixture: {id}"));
                continue;
            };
            if reference.input_sha256 != observation.input_sha256 {
                failures.push(format!("numerical input identity changed: {id}"));
            }
            match NumericalMetrics::compare(&reference.logits, &observation.logits) {
                Ok(metrics) => {
                    if !metrics.passes() {
                        failures.push(format!("FR-8 numerical limits exceeded: {id}"));
                    }
                    numerical.insert(id.clone(), metrics);
                }
                Err(error) => failures.push(format!("{id}: {error}")),
            }
        }
        for id in self.references.keys().filter(|id| !seen.contains(*id)) {
            failures.push(format!("missing numerical fixture: {id}"));
        }
        QualificationVerdict {
            usable: failures.is_empty(),
            failures,
            numerical,
        }
    }
}

fn is_hex(value: &str) -> bool {
    value.bytes().all(|c| c.is_ascii_hexdigit())
}
fn is_digest(value: &str) -> bool {
    value.len() == 64 && is_hex(value)
}
