//! W3 canonical encoding, lifecycle labels, and immutable invocation bindings.
//!
//! Bindings and graph labels are not execution capabilities. Policy preparation
//! validates registry requests but does not persist/consume approvals, check a
//! live deadline, run hooks, or execute tools. Those require the durable coordinator.

use serde_json::Value;
use std::{
    error::Error,
    fmt,
    io::{self, Write},
};

/// Frozen project-local encoding version, not an RFC 8785 implementation.
pub const CANONICALIZATION_VERSION: &str = "alloy-json-c14n/1";
/// Frozen W3 invocation/storage contract identifier.
pub const CONTRACT_VERSION: &str = "alloy.w3/1";
/// Upper bound on a complete canonical binding, not just its arguments.
pub const MAX_CANONICAL_BINDING_BYTES: usize = 1_048_576;

/// Failure to encode a value under the frozen byte ceiling.
#[derive(Debug)]
pub enum CanonicalizationError {
    /// Encoded output cannot fit within the frozen binding ceiling.
    LimitExceeded,
    /// The pinned JSON serializer rejected a token.
    Serialization(serde_json::Error),
}

impl fmt::Display for CanonicalizationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LimitExceeded => f.write_str("canonical binding exceeds 1 MiB"),
            Self::Serialization(error) => write!(f, "canonical JSON serialization: {error}"),
        }
    }
}
impl Error for CanonicalizationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Serialization(error) => Some(error),
            _ => None,
        }
    }
}

struct BoundedOutput {
    bytes: Vec<u8>,
    exceeded: bool,
}
impl BoundedOutput {
    fn append(&mut self, bytes: &[u8]) -> Result<(), CanonicalizationError> {
        if bytes.len() > MAX_CANONICAL_BINDING_BYTES - self.bytes.len() {
            self.exceeded = true;
            return Err(CanonicalizationError::LimitExceeded);
        }
        self.bytes.extend_from_slice(bytes);
        Ok(())
    }
    fn string(&mut self, value: &str) -> Result<(), CanonicalizationError> {
        if let Err(error) = serde_json::to_writer(&mut *self, value) {
            return Err(if self.exceeded {
                CanonicalizationError::LimitExceeded
            } else {
                CanonicalizationError::Serialization(error)
            });
        }
        Ok(())
    }
}
impl Write for BoundedOutput {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.append(bytes)
            .map_err(|_| io::Error::new(io::ErrorKind::WriteZero, "canonical byte ceiling"))?;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

enum Frame<'a> {
    Value(&'a Value),
    Array(std::slice::Iter<'a, Value>, bool),
    Object(std::vec::IntoIter<(&'a String, &'a Value)>, bool),
}

/// Encode a previously validated value using `alloy-json-c14n/1`.
///
/// Objects sort keys by UTF-8 bytes, arrays retain order, and strings/numbers
/// use pinned serde_json encoding. No normalization/coercion/defaulting occurs.
/// Traversal is iterative; output cannot exceed the complete binding ceiling.
/// This does not validate schemas, reject duplicates lost by an earlier parser,
/// or authorize anything. An argument-only success does not establish that a
/// later complete invocation binding also fits the ceiling.
pub fn canonical_json(value: &Value) -> Result<Vec<u8>, CanonicalizationError> {
    let mut output = BoundedOutput {
        bytes: Vec::new(),
        exceeded: false,
    };
    let mut stack = vec![Frame::Value(value)];
    while let Some(frame) = stack.pop() {
        match frame {
            Frame::Value(value) => match value {
                Value::Null => output.append(b"null")?,
                Value::Bool(true) => output.append(b"true")?,
                Value::Bool(false) => output.append(b"false")?,
                Value::Number(number) => output.append(number.to_string().as_bytes())?,
                Value::String(string) => output.string(string)?,
                Value::Array(values) => {
                    output.append(b"[")?;
                    stack.push(Frame::Array(values.iter(), true));
                }
                Value::Object(values) => {
                    // Even four bytes per member cannot fit beyond this count.
                    // Check before allocating the temporary sorted reference list.
                    if values.len() > MAX_CANONICAL_BINDING_BYTES / 4 {
                        return Err(CanonicalizationError::LimitExceeded);
                    }
                    let mut entries: Vec<_> = values.iter().collect();
                    entries.sort_unstable_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
                    output.append(b"{")?;
                    stack.push(Frame::Object(entries.into_iter(), true));
                }
            },
            Frame::Array(mut values, first) => {
                if let Some(value) = values.next() {
                    if !first {
                        output.append(b",")?;
                    }
                    stack.push(Frame::Array(values, false));
                    stack.push(Frame::Value(value));
                } else {
                    output.append(b"]")?;
                }
            }
            Frame::Object(mut entries, first) => {
                if let Some((key, value)) = entries.next() {
                    if !first {
                        output.append(b",")?;
                    }
                    output.string(key)?;
                    output.append(b":")?;
                    stack.push(Frame::Object(entries, false));
                    stack.push(Frame::Value(value));
                } else {
                    output.append(b"}")?;
                }
            }
        }
    }
    Ok(output.bytes)
}

/// Pure lifecycle label. Freely constructing a label grants no authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ApprovalState {
    /// Waiting for an explicit decision.
    AwaitingApproval,
    /// Eligible for a later atomic claim, subject to live checks.
    Ready,
    /// Single-use claim held while pre-start work is performed.
    Claimed,
    /// A durable start may precede an attempted external effect.
    Started,
    /// Explicit human denial, before execution.
    Denied,
    /// Authorization lifetime elapsed before start.
    Expired,
    /// Pre-start authority withdrawn, including restart invalidation.
    Withdrawn,
    /// Policy/hook/final-check rejection before start.
    Rejected,
    /// Actual outcome durably recorded.
    Completed,
    /// A started operation has no safely inferable completion after recovery.
    OutcomeUnknown,
}
impl ApprovalState {
    /// Exhaustive frozen labels, including terminal states.
    pub const ALL: [Self; 10] = [
        Self::AwaitingApproval,
        Self::Ready,
        Self::Claimed,
        Self::Started,
        Self::Denied,
        Self::Expired,
        Self::Withdrawn,
        Self::Rejected,
        Self::Completed,
        Self::OutcomeUnknown,
    ];
    /// Stable internal contract spelling, not a new RPC enum.
    pub const fn name(self) -> &'static str {
        match self {
            Self::AwaitingApproval => "AwaitingApproval",
            Self::Ready => "Ready",
            Self::Claimed => "Claimed",
            Self::Started => "Started",
            Self::Denied => "Denied",
            Self::Expired => "Expired",
            Self::Withdrawn => "Withdrawn",
            Self::Rejected => "Rejected",
            Self::Completed => "Completed",
            Self::OutcomeUnknown => "OutcomeUnknown",
        }
    }
    /// Whether this label can never transition back to runnable authority.
    pub const fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Denied
                | Self::Expired
                | Self::Withdrawn
                | Self::Rejected
                | Self::Completed
                | Self::OutcomeUnknown
        )
    }
    /// Check only graph legality; no preconditions or external effects are evaluated.
    ///
    /// The durable coordinator validates exact identity, expiry, epoch,
    /// policy/registration revisions, cancellation, ownership and hook outcome,
    /// and atomically persists state and event. This return value is not a grant.
    pub fn transition(self, event: Transition) -> Result<Self, InvalidTransition> {
        use ApprovalState::*;
        use Transition::*;
        let next = match (self, event) {
            (AwaitingApproval, Approve) => Ready,
            (AwaitingApproval, Deny) => Denied,
            (Ready, Claim) => Claimed,
            (Claimed, Start) => Started,
            (Started, Complete) => Completed,
            (AwaitingApproval | Ready | Claimed, Expire) => Expired,
            (AwaitingApproval | Ready | Claimed, Withdraw) => Withdrawn,
            (Claimed, Reject) => Rejected,
            (AwaitingApproval | Ready | Claimed, Restart) => Withdrawn,
            (Started, Restart) => OutcomeUnknown,
            (state, Restart) if state.is_terminal() => state,
            _ => return Err(InvalidTransition { state: self, event }),
        };
        Ok(next)
    }
}

/// A proposed graph transition, not proof that its preconditions hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Transition {
    /// First explicit matching allow decision.
    Approve,
    /// First explicit matching deny decision.
    Deny,
    /// Atomically acquire a single-use pre-start claim.
    Claim,
    /// Record start only after final checks and successful durable commit.
    Start,
    /// Persist the actual execution outcome.
    Complete,
    /// Deadline reached before start, including while claimed.
    Expire,
    /// Withdraw pre-start authority.
    Withdraw,
    /// Reject claimed invocation before effects.
    Reject,
    /// Recover labels after exclusive ownership and a new daemon epoch.
    Restart,
}
impl Transition {
    /// Exhaustive internal transition vocabulary.
    pub const ALL: [Self; 9] = [
        Self::Approve,
        Self::Deny,
        Self::Claim,
        Self::Start,
        Self::Complete,
        Self::Expire,
        Self::Withdraw,
        Self::Reject,
        Self::Restart,
    ];
    /// Stable fixture spelling; this is not a public wire error/notification.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Approve => "Approve",
            Self::Deny => "Deny",
            Self::Claim => "Claim",
            Self::Start => "Start",
            Self::Complete => "Complete",
            Self::Expire => "Expire",
            Self::Withdraw => "Withdraw",
            Self::Reject => "Reject",
            Self::Restart => "Restart",
        }
    }
}

/// A graph edge not permitted by the frozen contract.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct InvalidTransition {
    /// Original state, unchanged by rejection.
    pub state: ApprovalState,
    /// Proposed transition that was rejected.
    pub event: Transition,
}
impl fmt::Display for InvalidTransition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "invalid W3 transition: {} + {}",
            self.state.name(),
            self.event.name()
        )
    }
}
impl Error for InvalidTransition {}

// W3 policy/binding slice: snapshots only, never execution grants.

/// Fixed authorization lifetime, checked with overflow-safe arithmetic.
pub const AUTHORIZATION_TTL_MS: i64 = 300_000;

/// Failure to prepare an immutable invocation identity, not a wire error enum.
#[derive(Debug)]
pub enum BindingError {
    /// A canonical token stream exceeds its ceiling or cannot be encoded.
    Canonicalization(CanonicalizationError),
    /// Trusted contract metadata could not be represented as JSON.
    Serialization(serde_json::Error),
    /// Daemon-supplied epoch or timestamp is invalid.
    InvalidContext(&'static str),
    /// The expiration timestamp cannot be represented by the store's i64 clock.
    TimestampOverflow,
}
impl fmt::Display for BindingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Canonicalization(error) => write!(f, "binding canonicalization: {error}"),
            Self::Serialization(error) => write!(f, "binding serialization: {error}"),
            Self::InvalidContext(field) => write!(f, "invalid binding context: {field}"),
            Self::TimestampOverflow => f.write_str("binding expiration timestamp overflow"),
        }
    }
}
impl Error for BindingError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Canonicalization(error) => Some(error),
            Self::Serialization(error) => Some(error),
            _ => None,
        }
    }
}
impl From<CanonicalizationError> for BindingError {
    fn from(error: CanonicalizationError) -> Self {
        Self::Canonicalization(error)
    }
}
impl From<serde_json::Error> for BindingError {
    fn from(error: serde_json::Error) -> Self {
        Self::Serialization(error)
    }
}

/// Immutable, bounded request identity. NOT approval or permission to execute.
///
/// Construction is crate-private and requires a registry-resolved invocation.
/// No executable implementation is retained. No Clone, Deserialize, or unchecked
/// public constructor is provided. The durable coordinator enforces
/// uniqueness, monotonic deadlines, revision/epoch checks and single-use claims.
#[derive(Debug)]
pub struct InvocationBinding {
    invocation: tool_registry::call::ToolInvocation,
    definition: tool_registry::definition::ToolDefinition,
    snapshot: Value,
    bytes: Vec<u8>,
    digest: String,
}

/// Already evaluated daemon inputs; not accepted from model/client data.
pub(crate) struct BindingInputs<'a> {
    /// Current daemon instance identity.
    pub(crate) daemon_epoch: &'a str,
    /// Complete effective policy digest.
    pub(crate) policy_revision: &'a str,
    /// Resolved prompt requirement after daemon policy.
    pub(crate) effective_approval: tool_registry::definition::ApprovalRequirement,
    /// Tightened duration ceiling.
    pub(crate) timeout_ms: u64,
    /// Tightened collection ceiling.
    pub(crate) max_output_bytes: u64,
    /// Daemon clock observation, not a client-provided deadline.
    pub(crate) created_at_unix_ms: i64,
}

/// SHA-256 identity checksum; not a signature or authorization credential.
pub(crate) fn identity_digest(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(bytes))
}

impl InvocationBinding {
    pub(crate) fn from_resolved(
        resolved: &tool_registry::registry::ResolvedInvocation<'_>,
        inputs: BindingInputs<'_>,
    ) -> Result<Self, BindingError> {
        use sha2::{Digest, Sha256};
        if inputs.daemon_epoch.is_empty() || inputs.daemon_epoch.trim() != inputs.daemon_epoch {
            return Err(BindingError::InvalidContext("daemon_epoch"));
        }
        if inputs.created_at_unix_ms < 0 {
            return Err(BindingError::InvalidContext("created_at_unix_ms"));
        }
        let expires = inputs
            .created_at_unix_ms
            .checked_add(AUTHORIZATION_TTL_MS)
            .ok_or(BindingError::TimestampOverflow)?;
        let invocation = resolved.invocation();
        let definition = resolved.definition();
        let argument_bytes = canonical_json(&Value::Object(invocation.call.arguments.clone()))?;
        // canonical_json emits valid UTF-8 JSON; errors still remain fail-closed.
        let arguments = String::from_utf8(argument_bytes.clone())
            .map_err(|_| BindingError::InvalidContext("canonical_arguments_utf8"))?;
        let definition_bytes = canonical_json(&serde_json::to_value(definition)?)?;
        let snapshot = serde_json::json!({
            "contract_version": CONTRACT_VERSION,
            "daemon_epoch": inputs.daemon_epoch,
            "session_id": invocation.context.session_id,
            "turn_id": invocation.context.turn_id,
            "generation_id": invocation.context.generation_id,
            "call_id": invocation.call.call_id,
            "tool_id": definition.id,
            "registered_tool_version": definition.version,
            "registered_definition_digest": identity_digest(&definition_bytes),
            "canonical_arguments": arguments,
            "arguments_digest": identity_digest(&argument_bytes),
            "policy_revision": inputs.policy_revision,
            "effective_effect": definition.policy.effect,
            "effective_approval_requirement": inputs.effective_approval,
            "effective_timeout_ms": inputs.timeout_ms,
            "effective_max_output_bytes": inputs.max_output_bytes,
            "created_at_unix_ms": inputs.created_at_unix_ms,
            "expires_at_unix_ms": expires,
        });
        // The complete envelope must fit, even when arguments alone already fit.
        let bytes = canonical_json(&snapshot)?;
        let mut hash = Sha256::new();
        hash.update(b"alloy.w3.invocation.v1\0");
        hash.update(&bytes);
        Ok(Self {
            invocation: invocation.clone(),
            definition: definition.clone(),
            snapshot,
            bytes,
            digest: format!("{:x}", hash.finalize()),
        })
    }
    /// Read the original validated request; no mutation or implementation handle.
    pub fn invocation(&self) -> &tool_registry::call::ToolInvocation {
        &self.invocation
    }
    /// Read the registered definition captured during validation.
    pub fn definition(&self) -> &tool_registry::definition::ToolDefinition {
        &self.definition
    }
    /// Read all frozen binding fields; callers cannot mutate the retained snapshot.
    pub fn snapshot(&self) -> &Value {
        &self.snapshot
    }
    /// Exact project-local canonical bytes of the complete binding.
    pub fn canonical_bytes(&self) -> &[u8] {
        &self.bytes
    }
    /// Domain-separated SHA-256 identity, not a bearer token or approval.
    pub fn digest(&self) -> &str {
        &self.digest
    }
}
