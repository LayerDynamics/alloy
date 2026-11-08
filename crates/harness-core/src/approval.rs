//! Pure W3 contract primitives, not durable authorization.
//!
//! This module implements canonical byte encoding and the permitted state graph.
//! A state value is NOT an execution capability. These functions perform no
//! registry validation, policy evaluation, clock checks, database transaction,
//! hook invocation, or tool execution. The later durable coordinator must prove
//! those preconditions before making a transition usable for execution.

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
    /// A later durable coordinator must validate exact identity, expiry, epoch,
    /// policy/registration revisions, cancellation, ownership and hook outcome,
    /// then atomically persist state and event. This return value is not a grant.
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
