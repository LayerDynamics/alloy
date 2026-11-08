use std::{net::IpAddr, time::Duration};
use tool_registry::{
    builtins::http::{HttpDestination, HttpError, HttpRequest, HttpScope},
    definition::{ApprovalRequirement, ToolEffect, ToolPolicy},
    execution::{BoundedOutput, CancellationToken, ExecutionContext, ExecutionError, OutputSink},
};
mod fixture {
    include!("fixtures/w4_http_server.rs");
}
use fixture::{Server, response};
fn scope(origin: &str, redirects: u32, body: u64) -> HttpScope {
    HttpScope::new(
        vec![HttpDestination::new(origin, "127.0.0.1".parse::<IpAddr>().unwrap()).unwrap()],
        redirects,
        body,
        8192,
        Duration::from_millis(300),
    )
    .unwrap()
}
fn context(ms: u64, maximum: u64, token: CancellationToken) -> ExecutionContext {
    ExecutionContext::new(
        &ToolPolicy {
            effect: ToolEffect::Read,
            approval: ApprovalRequirement::Policy,
            timeout_ms: ms,
            max_output_bytes: maximum,
        },
        ms,
        maximum,
        token,
    )
    .unwrap()
}
fn body(bytes: &[u8]) -> Vec<u8> {
    let mut result = Vec::new();
    for line in bytes.split(|b| *b == b'\n').filter(|line| !line.is_empty()) {
        let value: serde_json::Value = serde_json::from_slice(line).unwrap();
        if value["type"] == "body" {
            result.extend(serde_json::from_value::<Vec<u8>>(value["data"].clone()).unwrap());
        }
    }
    result
}
#[test]
fn get_preserves_binary_body_and_exact_output_count() {
    let _http_timing_guard = fixture::http_timing_test_guard();
    let server = Server::new(|_, stream, _| response(stream, 200, "", &[0, 255, 128, 10]));
    let ctx = context(5000, 65536, CancellationToken::default());
    let mut out = BoundedOutput::new(&ctx, 65536).unwrap();
    let report = scope(&server.origin, 2, 4096)
        .fetch(
            &HttpRequest::new(&format!("{}/ok", server.origin), "GET").unwrap(),
            &ctx,
            &mut out,
        )
        .unwrap();
    let bytes = out.finish().unwrap();
    assert_eq!(body(&bytes), vec![0, 255, 128, 10]);
    assert_eq!(report.body_bytes, 4);
    assert_eq!(report.output_bytes, bytes.len() as u64);
    assert_eq!(server.calls.lock().unwrap().len(), 1);
}
#[test]
fn head_sends_head_and_collects_no_body() {
    let _http_timing_guard = fixture::http_timing_test_guard();
    let server = Server::new(|_, stream, _| response(stream, 200, "", b""));
    let ctx = context(5000, 65536, CancellationToken::default());
    let mut out = BoundedOutput::new(&ctx, 65536).unwrap();
    let report = scope(&server.origin, 0, 4096)
        .fetch(
            &HttpRequest::new(&server.origin, "HEAD").unwrap(),
            &ctx,
            &mut out,
        )
        .unwrap();
    assert_eq!(report.body_bytes, 0);
    assert!(body(&out.finish().unwrap()).is_empty());
    assert!(server.calls.lock().unwrap()[0].starts_with("HEAD "));
}
#[test]
fn relative_redirect_is_checked_and_followed() {
    let _http_timing_guard = fixture::http_timing_test_guard();
    let server = Server::new(|line, stream, _| {
        if line.contains(" /redirect ") {
            response(stream, 302, "Location: /ok\r\n", b"")
        } else {
            response(stream, 200, "", b"ok")
        }
    });
    let ctx = context(5000, 65536, CancellationToken::default());
    let mut out = BoundedOutput::new(&ctx, 65536).unwrap();
    let report = scope(&server.origin, 1, 4096)
        .fetch(
            &HttpRequest::new(&format!("{}/redirect", server.origin), "GET").unwrap(),
            &ctx,
            &mut out,
        )
        .unwrap();
    assert_eq!(report.requests_attempted, 2);
    assert_eq!(body(&out.finish().unwrap()), b"ok");
    assert_eq!(server.calls.lock().unwrap().len(), 2);
}
#[test]
fn redirect_scope_escape_never_contacts_target() {
    let _http_timing_guard = fixture::http_timing_test_guard();
    let denied = Server::new(|_, stream, _| response(stream, 200, "", b"secret"));
    let location = denied.origin.clone();
    let allowed = Server::new(move |_, stream, _| {
        response(
            stream,
            302,
            &format!("Location: {location}/secret\r\n"),
            b"",
        )
    });
    let ctx = context(5000, 65536, CancellationToken::default());
    let mut out = BoundedOutput::new(&ctx, 65536).unwrap();
    let error = scope(&allowed.origin, 2, 4096)
        .fetch(
            &HttpRequest::new(&allowed.origin, "GET").unwrap(),
            &ctx,
            &mut out,
        )
        .unwrap_err();
    assert!(matches!(error.cause, HttpError::Policy(_)));
    assert_eq!(error.requests_attempted, 1);
    assert!(denied.calls.lock().unwrap().is_empty());
    assert!(out.finish().is_err());
}
#[test]
fn initial_denial_connects_to_no_server() {
    let _http_timing_guard = fixture::http_timing_test_guard();
    let denied = Server::new(|_, stream, _| response(stream, 200, "", b"secret"));
    let allowed = Server::new(|_, stream, _| response(stream, 200, "", b"ok"));
    let ctx = context(5000, 65536, CancellationToken::default());
    let mut out = BoundedOutput::new(&ctx, 65536).unwrap();
    let error = scope(&allowed.origin, 2, 4096)
        .fetch(
            &HttpRequest::new(&denied.origin, "GET").unwrap(),
            &ctx,
            &mut out,
        )
        .unwrap_err();
    assert_eq!(error.requests_attempted, 0);
    assert!(denied.calls.lock().unwrap().is_empty());
    assert!(out.finish().is_err());
}
#[test]
fn credentials_fragments_schemes_and_methods_reject() {
    let _http_timing_guard = fixture::http_timing_test_guard();
    for (url, method) in [
        ("file:///etc/passwd", "GET"),
        ("http://user:password@example.test/", "GET"),
        ("http://example.test/#fragment", "GET"),
        ("http://example.test/", "POST"),
        ("http://example.test/\nheader", "GET"),
    ] {
        assert!(HttpRequest::new(url, method).is_err());
    }
}
#[test]
fn redirect_limit_and_missing_location_fail_closed() {
    let _http_timing_guard = fixture::http_timing_test_guard();
    for missing in [false, true] {
        let server = Server::new(move |_, stream, _| {
            response(
                stream,
                302,
                if missing { "" } else { "Location: /loop\r\n" },
                b"",
            )
        });
        let ctx = context(5000, 65536, CancellationToken::default());
        let mut out = BoundedOutput::new(&ctx, 65536).unwrap();
        assert!(
            scope(&server.origin, 1, 4096)
                .fetch(
                    &HttpRequest::new(&server.origin, "GET").unwrap(),
                    &ctx,
                    &mut out
                )
                .is_err()
        );
        assert!(out.finish().is_err());
        assert_eq!(
            server.calls.lock().unwrap().len(),
            if missing { 1 } else { 2 }
        );
    }
}
#[test]
fn body_and_serialized_output_limits_abort_whole_result() {
    let _http_timing_guard = fixture::http_timing_test_guard();
    for (body_limit, output_limit) in [(3, 65536), (4096, 8)] {
        let server = Server::new(|_, stream, _| response(stream, 200, "", b"1234"));
        let ctx = context(5000, output_limit, CancellationToken::default());
        let mut out = BoundedOutput::new(&ctx, output_limit as usize).unwrap();
        assert!(
            scope(&server.origin, 0, body_limit)
                .fetch(
                    &HttpRequest::new(&server.origin, "GET").unwrap(),
                    &ctx,
                    &mut out
                )
                .is_err()
        );
        assert!(out.finish().is_err());
    }
}
#[test]
fn precancellation_has_zero_network_attempts() {
    let _http_timing_guard = fixture::http_timing_test_guard();
    let server = Server::new(|_, stream, _| response(stream, 200, "", b"ok"));
    let token = CancellationToken::default();
    token.cancel();
    let ctx = context(5000, 65536, token);
    let mut out = BoundedOutput::new(&ctx, 65536).unwrap();
    let error = scope(&server.origin, 0, 4096)
        .fetch(
            &HttpRequest::new(&server.origin, "GET").unwrap(),
            &ctx,
            &mut out,
        )
        .unwrap_err();
    assert_eq!(error.requests_attempted, 0);
    assert!(server.calls.lock().unwrap().is_empty());
}
#[test]
fn expired_deadline_is_not_renewed() {
    let _http_timing_guard = fixture::http_timing_test_guard();
    let server = Server::new(|_, stream, _| response(stream, 200, "", b"ok"));
    let ctx = context(20, 65536, CancellationToken::default());
    let mut out = BoundedOutput::new(&ctx, 65536).unwrap();
    std::thread::sleep(Duration::from_millis(40));
    assert!(
        scope(&server.origin, 0, 4096)
            .fetch(
                &HttpRequest::new(&server.origin, "GET").unwrap(),
                &ctx,
                &mut out
            )
            .is_err()
    );
    assert!(server.calls.lock().unwrap().is_empty());
}
struct Reject;
impl OutputSink for Reject {
    fn write_chunk(&mut self, _: &[u8]) -> Result<(), ExecutionError> {
        Err(ExecutionError::OutputLimit)
    }
    fn abort(&mut self) {}
}
#[test]
fn sink_failure_aborts_without_retrying_request() {
    let _http_timing_guard = fixture::http_timing_test_guard();
    let server = Server::new(|_, stream, _| response(stream, 200, "", b"ok"));
    let ctx = context(5000, 65536, CancellationToken::default());
    assert!(
        scope(&server.origin, 0, 4096)
            .fetch(
                &HttpRequest::new(&server.origin, "GET").unwrap(),
                &ctx,
                &mut Reject
            )
            .is_err()
    );
    assert_eq!(server.calls.lock().unwrap().len(), 1);
}
#[test]
fn named_origin_uses_only_configured_address() {
    let _http_timing_guard = fixture::http_timing_test_guard();
    let server = Server::new(|_, stream, _| response(stream, 200, "", b"pinned"));
    let port = server.origin.rsplit(':').next().unwrap();
    let named = format!("http://not-a-real-host.invalid:{port}");
    let ctx = context(5000, 65536, CancellationToken::default());
    let mut out = BoundedOutput::new(&ctx, 65536).unwrap();
    scope(&named, 0, 4096)
        .fetch(&HttpRequest::new(&named, "GET").unwrap(), &ctx, &mut out)
        .unwrap();
    assert_eq!(body(&out.finish().unwrap()), b"pinned");
}
#[test]
fn cancellation_during_stalled_headers_releases_owned_socket() {
    let _http_timing_guard = fixture::http_timing_test_guard();
    use std::{
        io::Read,
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        },
        time::Instant,
    };
    let closed = Arc::new(AtomicBool::new(false));
    let observed = closed.clone();
    let server = Server::new(move |_, stream, stop| {
        let start = Instant::now();
        let mut byte = [0];
        while start.elapsed() < Duration::from_secs(2) && !stop.load(Ordering::Acquire) {
            match stream.read(&mut byte) {
                Ok(0) => {
                    observed.store(true, Ordering::Release);
                    break;
                }
                Ok(_) => (),
                Err(_) => (),
            }
        }
    });
    let token = CancellationToken::default();
    let cancel = token.clone();
    let calls = server.calls.clone();
    let worker = std::thread::spawn(move || {
        let start = Instant::now();
        while calls.lock().unwrap().is_empty() {
            assert!(start.elapsed() < Duration::from_secs(2));
            std::thread::sleep(Duration::from_millis(2));
        }
        cancel.cancel();
    });
    let ctx = context(5000, 65536, token);
    let mut out = BoundedOutput::new(&ctx, 65536).unwrap();
    let error = scope(&server.origin, 0, 4096)
        .fetch(
            &HttpRequest::new(&server.origin, "GET").unwrap(),
            &ctx,
            &mut out,
        )
        .unwrap_err();
    worker.join().unwrap();
    assert!(matches!(
        error.cause,
        HttpError::Execution(ExecutionError::Cancelled)
    ));
    let start = Instant::now();
    while !closed.load(Ordering::Acquire) {
        assert!(start.elapsed() < Duration::from_secs(1));
        std::thread::sleep(Duration::from_millis(2));
    }
    assert!(out.finish().is_err());
}

#[test]
fn exact_body_limit_succeeds_but_chunked_overflow_fails() {
    let _http_timing_guard = fixture::http_timing_test_guard();
    let exact = Server::new(|_, stream, _| response(stream, 200, "", b"1234"));
    let ctx = context(5000, 65536, CancellationToken::default());
    let mut out = BoundedOutput::new(&ctx, 65536).unwrap();
    scope(&exact.origin, 0, 4)
        .fetch(
            &HttpRequest::new(&exact.origin, "GET").unwrap(),
            &ctx,
            &mut out,
        )
        .unwrap();
    assert_eq!(body(&out.finish().unwrap()), b"1234");
    let chunked = Server::new(|_, stream, _| {
        use std::io::Write;
        let _=stream.write_all(b"HTTP/1.1 200 Fixture\r\nConnection: close\r\nTransfer-Encoding: chunked\r\n\r\n4\r\n1234\r\n0\r\n\r\n");
    });
    let mut out = BoundedOutput::new(&ctx, 65536).unwrap();
    assert!(
        scope(&chunked.origin, 0, 3)
            .fetch(
                &HttpRequest::new(&chunked.origin, "GET").unwrap(),
                &ctx,
                &mut out
            )
            .is_err()
    );
    assert!(out.finish().is_err());
}
#[test]
fn truncated_body_never_becomes_complete_output() {
    let _http_timing_guard = fixture::http_timing_test_guard();
    let server = Server::new(|_, stream, _| {
        use std::io::Write;
        let _ = stream.write_all(
            b"HTTP/1.1 200 Fixture\r\nConnection: close\r\nContent-Length: 10\r\n\r\nabc",
        );
    });
    let ctx = context(5000, 65536, CancellationToken::default());
    let mut out = BoundedOutput::new(&ctx, 65536).unwrap();
    assert!(
        scope(&server.origin, 0, 4096)
            .fetch(
                &HttpRequest::new(&server.origin, "GET").unwrap(),
                &ctx,
                &mut out
            )
            .is_err()
    );
    assert!(out.finish().is_err());
}
#[test]
fn encoded_response_is_rejected_instead_of_unbounded_decompression() {
    let _http_timing_guard = fixture::http_timing_test_guard();
    let server = Server::new(|_, stream, _| {
        response(stream, 200, "Content-Encoding: gzip\r\n", b"not gzip")
    });
    let ctx = context(5000, 65536, CancellationToken::default());
    let mut out = BoundedOutput::new(&ctx, 65536).unwrap();
    assert!(
        scope(&server.origin, 0, 4096)
            .fetch(
                &HttpRequest::new(&server.origin, "GET").unwrap(),
                &ctx,
                &mut out
            )
            .is_err()
    );
    assert!(out.finish().is_err());
}
#[test]
fn head_is_preserved_across_redirects() {
    let _http_timing_guard = fixture::http_timing_test_guard();
    let server = Server::new(|line, stream, _| {
        if line.contains(" /redirect ") {
            response(stream, 303, "Location: /ok\r\n", b"")
        } else {
            response(stream, 200, "", b"")
        }
    });
    let ctx = context(5000, 65536, CancellationToken::default());
    let mut out = BoundedOutput::new(&ctx, 65536).unwrap();
    scope(&server.origin, 1, 4096)
        .fetch(
            &HttpRequest::new(&format!("{}/redirect", server.origin), "HEAD").unwrap(),
            &ctx,
            &mut out,
        )
        .unwrap();
    assert!(body(&out.finish().unwrap()).is_empty());
    let calls = server.calls.lock().unwrap();
    assert_eq!(calls.len(), 2);
    assert!(calls.iter().all(|line| line.starts_with("HEAD ")));
}

#[test]
fn stalled_body_timeout_releases_socket_without_background_worker() {
    let _http_timing_guard = fixture::http_timing_test_guard();
    use std::{
        io::{Read, Write},
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        },
        time::Instant,
    };
    let closed = Arc::new(AtomicBool::new(false));
    let observed = closed.clone();
    let server = Server::new(move |_, stream, stop| {
        let _ = stream.write_all(
            b"HTTP/1.1 200 Fixture\r\nConnection: close\r\nContent-Length: 100\r\n\r\nx",
        );
        let start = Instant::now();
        let mut byte = [0];
        while start.elapsed() < Duration::from_secs(2) && !stop.load(Ordering::Acquire) {
            match stream.read(&mut byte) {
                Ok(0) => {
                    observed.store(true, Ordering::Release);
                    break;
                }
                Ok(_) => (),
                Err(_) => (),
            }
        }
    });
    let ctx = context(5000, 65536, CancellationToken::default());
    let mut out = BoundedOutput::new(&ctx, 65536).unwrap();
    let start = Instant::now();
    assert!(
        scope(&server.origin, 0, 4096)
            .fetch(
                &HttpRequest::new(&server.origin, "GET").unwrap(),
                &ctx,
                &mut out
            )
            .is_err()
    );
    assert!(start.elapsed() < Duration::from_secs(3));
    assert!(out.finish().is_err());
    let start = Instant::now();
    while !closed.load(Ordering::Acquire) {
        assert!(start.elapsed() < Duration::from_secs(1));
        std::thread::sleep(Duration::from_millis(2));
    }
}

#[test]
fn fixture_waits_across_header_read_poll_timeouts() {
    let _http_timing_guard = fixture::http_timing_test_guard();
    use std::io::{Read, Write};

    let server = Server::new(|_, stream, _| response(stream, 200, "", b"ok"));
    let address = server.origin.strip_prefix("http://").unwrap();
    let mut client = std::net::TcpStream::connect(address).unwrap();
    client
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();

    // Deliberately exceed the fixture's 100 ms read-poll timeout.
    std::thread::sleep(Duration::from_millis(250));

    client
        .write_all(b"GET /ok HTTP/1.1\r\nHost: fixture\r\nConnection: close\r\n\r\n")
        .unwrap();

    let mut received = Vec::new();
    client.read_to_end(&mut received).unwrap();

    assert!(received.starts_with(b"HTTP/1.1 200 "));
    assert!(received.ends_with(b"ok"));
    assert_eq!(server.calls.lock().unwrap().len(), 1);
}

#[test]
fn four_http_requests_overlap_without_cross_operation_output() {
    let _http_timing_guard = fixture::http_timing_test_guard();
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use std::time::Instant;

    let arrived = Arc::new(AtomicUsize::new(0));
    let mut servers = Vec::new();
    let mut workers = Vec::new();

    for index in 0u8..4 {
        let seen = arrived.clone();
        let server = Server::new(move |_, stream, stopped| {
            seen.fetch_add(1, Ordering::AcqRel);
            let started = Instant::now();
            // No successful response is sent until all four requests arrive.
            // A serialized production request path therefore cannot pass.
            while seen.load(Ordering::Acquire) < 4 {
                if stopped.load(Ordering::Acquire) || started.elapsed() >= Duration::from_secs(4) {
                    response(stream, 503, "", b"overlap requirement not met");
                    return;
                }
                std::thread::sleep(Duration::from_millis(2));
            }
            response(stream, 200, "", &[index]);
        });
        let origin = server.origin.clone();
        servers.push(server);
        workers.push(std::thread::spawn(move || {
            // This test checks actual overlap and byte isolation, not a 300 ms
            // latency target. The existing short-budget tests remain unchanged.
            let scope = HttpScope::new(
                vec![HttpDestination::new(&origin, "127.0.0.1".parse().unwrap()).unwrap()],
                0,
                4096,
                8192,
                Duration::from_secs(5),
            )
            .unwrap();
            let ctx = context(10000, 65536, CancellationToken::default());
            let mut output = BoundedOutput::new(&ctx, 65536).unwrap();
            let report = scope
                .fetch(
                    &HttpRequest::new(&origin, "GET").unwrap(),
                    &ctx,
                    &mut output,
                )
                .unwrap();
            assert_eq!(report.status, 200, "all four requests must overlap");
            assert_eq!(report.requests_attempted, 1);
            assert_eq!(body(&output.finish().unwrap()), vec![index]);
        }));
    }

    // Join every worker before asserting, including after any worker failure.
    let completed: Vec<_> = workers.into_iter().map(|worker| worker.join()).collect();
    assert!(
        completed.iter().all(Result::is_ok),
        "concurrent HTTP worker failed"
    );
    assert_eq!(arrived.load(Ordering::Acquire), 4);
    for server in servers {
        assert_eq!(server.calls.lock().unwrap().len(), 1);
    }
}
