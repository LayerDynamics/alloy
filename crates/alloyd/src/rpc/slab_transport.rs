//! Borrowed, bounded serialization for slab proposal notifications.
//! The success path creates neither a Value tree nor an encoded line buffer.
//! Serializer control state is stack-local; socket/kernel buffers and control-
//! plane error diagnostics are outside this transport-buffer contract.
use super::{ErrorKind, RpcError};
use engine::parser_allocation::FrozenParserAllocation;
use serde::{
    Serialize, Serializer,
    ser::{Error as _, SerializeSeq},
};
use std::{
    io::{self, Write},
    num::NonZeroUsize,
    time::Duration,
};

/// Explicit proposal-notification policy. Does not enable tools or approvals.
#[derive(Clone, Copy, Debug)]
pub struct RpcTransportLimits {
    max_notification_bytes: NonZeroUsize,
    write_timeout: Duration,
}
impl RpcTransportLimits {
    /// Limit includes the complete escaped JSON envelope and trailing newline.
    /// Socket timeout bounds each blocking write; it is not a whole-turn deadline.
    pub fn new(max_notification_bytes: usize, write_timeout: Duration) -> Result<Self, RpcError> {
        let max_notification_bytes = NonZeroUsize::new(max_notification_bytes)
            .ok_or_else(|| RpcError::invalid_params("notification byte limit must be positive"))?;
        if write_timeout.is_zero() {
            return Err(RpcError::invalid_params(
                "socket write timeout must be positive",
            ));
        }
        Ok(Self {
            max_notification_bytes,
            write_timeout,
        })
    }
    /// Full escaped notification-byte ceiling, including its newline.
    pub fn max_notification_bytes(self) -> usize {
        self.max_notification_bytes.get()
    }
    /// Timeout installed on connected Unix sockets, per blocking write operation.
    pub fn write_timeout(self) -> Duration {
        self.write_timeout
    }
}

#[derive(Serialize)]
struct Notification<'a> {
    jsonrpc: &'static str,
    method: &'static str,
    params: Params<'a>,
}
#[derive(Serialize)]
struct Params<'a> {
    session_id: &'a str,
    event: &'static str,
    body: Body<'a>,
}
#[derive(Serialize)]
struct Body<'a> {
    encoding: &'static str,
    calls: Calls<'a>,
}
#[derive(Serialize)]
struct Call<'a> {
    name: &'a str,
    arguments_json: &'a str,
}
struct Calls<'a>(&'a FrozenParserAllocation);
impl Serialize for Calls<'_> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut sequence = serializer.serialize_seq(Some(self.0.proposal_count()))?;
        for index in 0..self.0.proposal_count() {
            let proposal = self
                .0
                .proposal(index)
                .ok_or_else(|| S::Error::custom("invalid slab proposal descriptor"))?;
            sequence.serialize_element(&Call {
                name: proposal.name,
                arguments_json: proposal.arguments_json,
            })?;
        }
        sequence.end()
    }
}
fn notification<'a>(
    session_id: &'a str,
    proposals: &'a FrozenParserAllocation,
) -> Notification<'a> {
    Notification {
        jsonrpc: "2.0",
        method: "session.event",
        params: Params {
            session_id,
            event: "tool_call_batch_proposed",
            body: Body {
                encoding: "validated_json_text",
                calls: Calls(proposals),
            },
        },
    }
}

struct Counter {
    count: usize,
    limit: usize,
}
impl Write for Counter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let next = self
            .count
            .checked_add(bytes.len())
            .ok_or_else(|| io::Error::other("notification byte-count overflow"))?;
        if next > self.limit {
            return Err(io::Error::other(
                "notification exceeds transport byte limit",
            ));
        }
        self.count = next;
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
struct ExactWriter<'a, W> {
    inner: &'a mut W,
    remaining: usize,
}
impl<W: Write> Write for ExactWriter<'_, W> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > self.remaining {
            return Err(io::Error::other(
                "notification serialization exceeded its measured size",
            ));
        }
        let written = self.inner.write(bytes)?;
        if written > bytes.len() {
            return Err(io::Error::other("writer returned an invalid byte count"));
        }
        self.remaining -= written;
        Ok(written)
    }
    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

/// Measure the exact escaped envelope without retaining serialized bytes.
/// A refused frame produces no writes to the destination socket.
pub fn measure_slab_session_event(
    session_id: &str,
    proposals: &FrozenParserAllocation,
    limits: RpcTransportLimits,
) -> Result<usize, RpcError> {
    let mut counter = Counter {
        count: 0,
        limit: limits.max_notification_bytes(),
    };
    serde_json::to_writer(&mut counter, &notification(session_id, proposals)).map_err(|error| {
        RpcError::application(
            ErrorKind::EngineFailed,
            format!("measure slab notification: {error}"),
        )
    })?;
    counter.write_all(b"\n").map_err(|error| {
        RpcError::application(
            ErrorKind::EngineFailed,
            format!("measure slab notification newline: {error}"),
        )
    })?;
    Ok(counter.count)
}

/// Synchronously serialize borrowed slab views, with exact preflight sizing.
///
/// The caller retains the slab throughout measuring, writing, and flushing.
/// There is no queued owned response, Vec of calls, parsed argument tree,
/// or encoded String. Wire schema is unchanged: arguments remain JSON text.
/// Errors after output begins require discarding the connection; server.rs
/// enforces this with a writer that refuses further output after I/O failure.
pub fn write_slab_session_event<W: Write>(
    writer: &mut W,
    session_id: &str,
    proposals: &FrozenParserAllocation,
    limits: RpcTransportLimits,
) -> Result<usize, RpcError> {
    let expected = measure_slab_session_event(session_id, proposals, limits)?;
    let mut output = ExactWriter {
        inner: writer,
        remaining: expected,
    };
    serde_json::to_writer(&mut output, &notification(session_id, proposals))
        .map_err(|error| RpcError::internal(format!("write slab notification: {error}")))?;
    output
        .write_all(b"\n")
        .map_err(|error| RpcError::internal(format!("write slab notification newline: {error}")))?;
    if output.remaining != 0 {
        return Err(RpcError::internal(
            "notification serialization differed from its measured size",
        ));
    }
    output
        .flush()
        .map_err(|error| RpcError::internal(format!("flush slab notification: {error}")))?;
    Ok(expected)
}

#[cfg(test)]
mod tests {
    use super::*;
    use engine::{
        admission::AdmissionController,
        parser_allocation::{ParserAllocation, ParserAllocationLimits},
    };
    use serde_json::Value;
    use std::io::BufRead;

    fn fixture() -> (AdmissionController, FrozenParserAllocation) {
        let limits = ParserAllocationLimits {
            carry_bytes: 32,
            call_bytes: 512,
            decoded_bytes: 512,
            depth: 8,
            keys: 16,
            nodes: 32,
            proposal_bytes: 1024,
            calls: 4,
        };
        let bytes = ParserAllocation::required_bytes(limits).expect("layout");
        let admission = AdmissionController::new(bytes).expect("controller");
        let mut grant = admission.reserve_grant("parser", &[bytes]).expect("grant");
        let mut storage = ParserAllocation::from_grant(&mut grant, limits).expect("storage");
        // Validated fixture JSON; this suite tests transport, not parsing.
        let args = r#"{"value":"quote \" slash \\ \u0000 雪","approved":true}"#.as_bytes();
        {
            let w = storage.workspaces();
            w.decoded[..7].copy_from_slice(b"fs.read");
            w.call[..args.len()].copy_from_slice(args);
        }
        storage
            .commit_validated_call(7, 0..7, args.len(), 0..args.len())
            .expect("commit");
        (admission, storage.freeze())
    }
    fn limits(maximum: usize) -> RpcTransportLimits {
        RpcTransportLimits::new(maximum, Duration::from_secs(2)).expect("limits")
    }
    #[test]
    fn streamed_slab_codec_preserves_json_and_escape_semantics() {
        let (_, batch) = fixture();
        let session = "sess-\"\\\n雪";
        let mut bytes = Vec::new();
        let measured = measure_slab_session_event(session, &batch, limits(65536)).expect("measure");
        assert_eq!(
            write_slab_session_event(&mut bytes, session, &batch, limits(measured)).expect("write"),
            measured
        );
        assert_eq!(bytes.len(), measured);
        assert_eq!(bytes.iter().filter(|&&b| b == b'\n').count(), 1);
        let decoded: Value = serde_json::from_slice(&bytes).expect("JSON");
        assert_eq!(decoded["jsonrpc"], "2.0");
        assert_eq!(decoded["method"], "session.event");
        assert_eq!(decoded["params"]["session_id"], session);
        assert_eq!(decoded["params"]["event"], "tool_call_batch_proposed");
        assert_eq!(decoded["params"]["body"]["encoding"], "validated_json_text");
        assert_eq!(
            decoded["params"]["body"]["calls"][0]["arguments_json"],
            batch.proposal(0).expect("call").arguments_json
        );
        assert!(
            decoded["params"]["body"]["calls"][0]
                .get("approved")
                .is_none()
        );
    }
    #[test]
    fn one_byte_short_is_rejected_before_any_destination_write() {
        struct NeverWrite;
        impl Write for NeverWrite {
            fn write(&mut self, _: &[u8]) -> io::Result<usize> {
                panic!("destination was touched before preflight passed")
            }
            fn flush(&mut self) -> io::Result<()> {
                panic!("destination was flushed")
            }
        }
        let (_, batch) = fixture();
        let exact = measure_slab_session_event("s", &batch, limits(65536)).expect("size");
        assert!(write_slab_session_event(&mut NeverWrite, "s", &batch, limits(exact - 1)).is_err());
    }
    #[test]
    fn source_reservation_remains_live_through_write_and_flush() {
        struct Observe {
            admission: AdmissionController,
            expected: u64,
            count: usize,
            flushed: bool,
        }
        impl Write for Observe {
            fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
                assert_eq!(
                    self.admission.snapshot().expect("ledger").reserved_bytes(),
                    self.expected
                );
                self.count += bytes.len();
                Ok(bytes.len())
            }
            fn flush(&mut self) -> io::Result<()> {
                assert_eq!(
                    self.admission.snapshot().expect("ledger").reserved_bytes(),
                    self.expected
                );
                self.flushed = true;
                Ok(())
            }
        }
        let (admission, batch) = fixture();
        let expected = batch.reserved_bytes();
        let mut sink = Observe {
            admission: admission.clone(),
            expected,
            count: 0,
            flushed: false,
        };
        let written =
            write_slab_session_event(&mut sink, "s", &batch, limits(65536)).expect("write");
        assert_eq!(sink.count, written);
        assert!(sink.flushed);
        assert_eq!(
            admission.snapshot().expect("owner live").reserved_bytes(),
            expected
        );
        drop(batch);
        assert_eq!(admission.snapshot().expect("released").reserved_bytes(), 0);
    }
    #[test]
    fn counter_overflow_is_rejected_without_allocating_a_frame() {
        let mut counter = Counter {
            count: usize::MAX,
            limit: usize::MAX,
        };
        assert!(counter.write(b"x").is_err());
        assert_eq!(counter.count, usize::MAX);
    }
    #[test]
    fn explicit_limits_reject_zero_values() {
        assert!(RpcTransportLimits::new(0, Duration::from_secs(1)).is_err());
        assert!(RpcTransportLimits::new(1, Duration::ZERO).is_err());
    }
    #[test]
    fn unix_socket_roundtrip_uses_the_borrowed_notification_writer() {
        use std::os::unix::net::UnixStream;
        let (writer, reader) = UnixStream::pair().expect("socket pair");
        writer
            .set_write_timeout(Some(Duration::from_secs(2)))
            .expect("write timeout");
        reader
            .set_read_timeout(Some(Duration::from_secs(2)))
            .expect("read timeout");
        let (admission, batch) = fixture();
        let held = batch.reserved_bytes();
        let mut writer = writer;
        let count = write_slab_session_event(&mut writer, "socket-session", &batch, limits(65536))
            .expect("write");
        let mut line = String::new();
        std::io::BufReader::new(reader)
            .read_line(&mut line)
            .expect("read line");
        assert_eq!(line.len(), count);
        let value: Value = serde_json::from_str(&line).expect("JSON");
        assert_eq!(value["params"]["session_id"], "socket-session");
        assert_eq!(
            admission.snapshot().expect("retained").reserved_bytes(),
            held
        );
        drop(batch);
        assert_eq!(admission.snapshot().expect("released").reserved_bytes(), 0);
    }
}

// W2_BORROWED_SLAB_TRANSPORT_V1
