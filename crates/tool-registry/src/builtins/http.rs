//! Scoped GET/HEAD with explicit origin/address policy and owned blocking I/O.
//! No ambient DNS, proxies, automatic redirects, cookies or credentials are used.
//! The daemon supplies pinned addresses, body/URL limits and a maximum exchange
//! duration. Cancellation is cooperative; in-flight blocking I/O is bounded by
//! min(remaining execution deadline, configured exchange duration). A slow valid
//! exchange can time out at this stricter bound; it is never automatically retried.
//! Normal HTTPS certificate/hostname verification remains enabled. Configuration
//! and execution context grant no W3 approval authority by themselves.
use crate::execution::{ExecutionContext, ExecutionError, OutputSink};
use std::{
    fmt,
    io::{self, Read},
    net::{IpAddr, SocketAddr},
    time::Duration,
};
use url::Url;

/// Internal failures mapped to the existing tool-result error categories.
#[derive(Debug)]
pub enum HttpError {
    /// URL, destination, redirect, encoding or body size violates scoped policy.
    Policy(&'static str),
    /// Cancellation, absolute execution deadline, or accepted-output failure.
    Execution(ExecutionError),
    /// The configured per-exchange budget expired before the overall deadline.
    ExchangeTimeout,
    /// HTTP transport or response framing failed.
    Transport,
}
impl fmt::Display for HttpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HTTP fetch: {self:?}")
    }
}
impl std::error::Error for HttpError {}
impl From<ExecutionError> for HttpError {
    fn from(error: ExecutionError) -> Self {
        Self::Execution(error)
    }
}
/// Failure metadata is conservative: attempting an exchange does not prove the
/// server did or did not receive a GET/HEAD request. Never automatically replay.
#[derive(Debug)]
pub struct HttpFailure {
    /// Original failure classification.
    pub cause: HttpError,
    /// Number of exchanges whose transport call was attempted.
    pub requests_attempted: u32,
}
impl fmt::Display for HttpFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} after {} attempted exchanges",
            self.cause, self.requests_attempted
        )
    }
}
impl std::error::Error for HttpFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.cause)
    }
}
/// Validated GET or HEAD and URL; not an execution capability.
pub struct HttpRequest {
    url: Url,
    method: String,
}
fn parse_url(value: &str) -> Result<Url, HttpError> {
    if value.bytes().any(|byte| byte.is_ascii_control()) {
        return Err(HttpError::Policy("URL control characters rejected"));
    }
    let url = Url::parse(value).map_err(|_| HttpError::Policy("invalid URL"))?;
    if !matches!(url.scheme(), "http" | "https")
        || url.host_str().is_none()
        || !url.username().is_empty()
        || url.password().is_some()
        || url.fragment().is_some()
    {
        return Err(HttpError::Policy(
            "HTTP(S) without credentials or fragment required",
        ));
    }
    Ok(url)
}
impl HttpRequest {
    /// Preserve GET/HEAD semantics; no caller headers, body or credentials exist.
    pub fn new(value: &str, method: &str) -> Result<Self, HttpError> {
        if !matches!(method, "GET" | "HEAD") {
            return Err(HttpError::Policy("GET or HEAD required"));
        }
        Ok(Self {
            url: parse_url(value)?,
            method: method.into(),
        })
    }
}
/// A trusted exact scheme/host/port origin pinned to one explicit IP address.
/// The IP may be local/private only when the daemon deliberately allows it.
pub struct HttpDestination {
    origin: String,
    address: IpAddr,
}
impl HttpDestination {
    /// Configure an origin only, not a path/query prefix. Hostname DNS resolution
    /// must happen outside execution under daemon policy; no fallback DNS exists.
    pub fn new(origin: &str, address: IpAddr) -> Result<Self, HttpError> {
        let url = parse_url(origin)?;
        if url.path() != "/"
            || url.query().is_some()
            || address.is_unspecified()
            || address.is_multicast()
        {
            return Err(HttpError::Policy(
                "origin and usable pinned address required",
            ));
        }
        // For a literal host, do not silently connect to a different configured IP.
        if let Some(host) = url.host() {
            let literal = match host {
                url::Host::Ipv4(ip) => Some(IpAddr::V4(ip)),
                url::Host::Ipv6(ip) => Some(IpAddr::V6(ip)),
                _ => None,
            };
            if literal.is_some_and(|ip| ip != address) {
                return Err(HttpError::Policy("literal host/address mismatch"));
            }
        }
        Ok(Self {
            origin: url.origin().ascii_serialization(),
            address,
        })
    }
}
/// Complete final response metadata; output is self-describing NDJSON containing
/// one response frame followed by binary-safe body frames. Redirect bodies are
/// not read or accepted as final output. HEAD reports zero body bytes.
#[derive(Debug)]
pub struct HttpReport {
    /// Final HTTP status; 4xx/5xx are complete responses but tool error outcomes.
    pub status: u16,
    /// Complete final URL after all policy-checked redirects.
    pub final_url: String,
    /// Number of attempted network exchanges.
    pub requests_attempted: u32,
    /// Accepted decoded-transfer body bytes (no content decompression is allowed).
    pub body_bytes: u64,
    /// Exact serialized output byte count including frame newlines.
    pub output_bytes: u64,
}
/// Immutable daemon policy and additional resource ceilings. No implicit network
/// roots or default capacities are invented; each limit must be supplied.
pub struct HttpScope {
    destinations: Vec<HttpDestination>,
    max_redirects: u32,
    max_body_bytes: u64,
    max_url_bytes: usize,
    max_exchange: Duration,
}
impl HttpScope {
    /// Configure origin/address allowlists and nonzero body, URL and exchange
    /// limits. Zero redirects permits only the initial response without following.
    pub fn new(
        destinations: Vec<HttpDestination>,
        max_redirects: u32,
        max_body_bytes: u64,
        max_url_bytes: usize,
        max_exchange: Duration,
    ) -> Result<Self, HttpError> {
        if destinations.is_empty()
            || max_body_bytes == 0
            || max_url_bytes == 0
            || max_exchange.is_zero()
        {
            return Err(HttpError::Policy(
                "nonempty scope and positive limits required",
            ));
        }
        let mut origins = std::collections::BTreeSet::new();
        if destinations
            .iter()
            .any(|destination| !origins.insert(destination.origin.clone()))
        {
            return Err(HttpError::Policy("duplicate origins rejected"));
        }
        Ok(Self {
            destinations,
            max_redirects,
            max_body_bytes,
            max_url_bytes,
            max_exchange,
        })
    }
    fn destination(&self, url: &Url) -> Result<SocketAddr, HttpError> {
        if url.as_str().len() > self.max_url_bytes {
            return Err(HttpError::Policy("URL limit exceeded"));
        }
        let origin = url.origin().ascii_serialization();
        let destination = self
            .destinations
            .iter()
            .find(|entry| entry.origin == origin)
            .ok_or(HttpError::Policy("destination origin denied"))?;
        Ok(SocketAddr::new(
            destination.address,
            url.port_or_known_default()
                .ok_or(HttpError::Policy("missing port"))?,
        ))
    }
    /// Perform bounded GET/HEAD after independent W3 approval/start. All owned
    /// response readers and agents drop on every return path; pooling is disabled.
    /// No worker thread is detached on cancellation, timeout or sink failure.
    pub fn fetch(
        &self,
        request: &HttpRequest,
        context: &ExecutionContext,
        output: &mut dyn OutputSink,
    ) -> Result<HttpReport, HttpFailure> {
        let mut attempts = 0;
        match self.fetch_inner(request, context, output, &mut attempts) {
            Ok(report) => Ok(report),
            Err(cause) => {
                output.abort();
                Err(HttpFailure {
                    cause,
                    requests_attempted: attempts,
                })
            }
        }
    }
    fn fetch_inner(
        &self,
        request: &HttpRequest,
        context: &ExecutionContext,
        output: &mut dyn OutputSink,
        attempts: &mut u32,
    ) -> Result<HttpReport, HttpError> {
        let mut current = request.url.clone();
        let mut redirects = 0u32;
        loop {
            context.check()?;
            // Validate before resolver/transport construction for every hop.
            let pinned = self.destination(&current)?;
            let budget = context.remaining()?.min(self.max_exchange);
            let agent = ureq::AgentBuilder::new()
                .redirects(0)
                .try_proxy_from_env(false)
                .max_idle_connections(0)
                .resolver(move |_host: &str| -> io::Result<Vec<SocketAddr>> { Ok(vec![pinned]) })
                .timeout_connect(budget)
                .timeout_read(budget)
                .timeout_write(budget)
                .timeout(budget)
                .build();
            context.check()?;
            *attempts = attempts
                .checked_add(1)
                .ok_or(HttpError::Policy("exchange count overflow"))?;
            let response = match agent
                .request(&request.method, current.as_str())
                .set("Accept-Encoding", "identity")
                .call()
            {
                Ok(response) => response,
                Err(ureq::Error::Status(_, response)) => response,
                Err(error) => {
                    context.check()?;
                    return Err(if error_chain_timeout(&error) {
                        HttpError::ExchangeTimeout
                    } else {
                        HttpError::Transport
                    });
                }
            };
            context.check()?;
            let status = response.status();
            if matches!(status,301..=303|307..=308) {
                if redirects >= self.max_redirects {
                    return Err(HttpError::Policy("redirect limit exceeded"));
                }
                let location = response
                    .header("Location")
                    .ok_or(HttpError::Policy("redirect missing Location"))?;
                if location.len() > self.max_url_bytes
                    || location.bytes().any(|byte| byte.is_ascii_control())
                {
                    return Err(HttpError::Policy("redirect URL rejected"));
                }
                let joined = current
                    .join(location)
                    .map_err(|_| HttpError::Policy("invalid redirect URL"))?;
                let next = parse_url(joined.as_str())?;
                if current.scheme() == "https" && next.scheme() != "https" {
                    return Err(HttpError::Policy("TLS downgrade rejected"));
                }
                self.destination(&next)?;
                context.check()?;
                drop(response);
                drop(agent);
                current = next;
                redirects += 1;
                continue;
            }
            if response
                .header("Content-Encoding")
                .is_some_and(|encoding| !encoding.eq_ignore_ascii_case("identity"))
            {
                return Err(HttpError::Policy("encoded response bodies rejected"));
            }
            if request.method != "HEAD"
                && response
                    .header("Content-Length")
                    .and_then(|length| length.parse::<u64>().ok())
                    .is_some_and(|length| length > self.max_body_bytes)
            {
                return Err(HttpError::Policy("response body limit exceeded"));
            }
            let mut output_bytes = 0;
            emit(
                &serde_json::json!({"type":"response","url":current.as_str(),"status":status,"method":request.method}),
                context,
                output,
                &mut output_bytes,
            )?;
            let mut body_bytes = 0u64;
            if request.method != "HEAD" {
                let mut reader = response.into_reader();
                let mut buffer = [0u8; 4096];
                loop {
                    context.check()?;
                    let n = match reader.read(&mut buffer) {
                        Ok(n) => n,
                        Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                        Err(error) => {
                            context.check()?;
                            return Err(if error_chain_timeout(&error) {
                                HttpError::ExchangeTimeout
                            } else {
                                HttpError::Transport
                            });
                        }
                    };
                    context.check()?;
                    if n == 0 {
                        break;
                    }
                    body_bytes = body_bytes
                        .checked_add(n as u64)
                        .ok_or(HttpError::Policy("body length overflow"))?;
                    if body_bytes > self.max_body_bytes {
                        return Err(HttpError::Policy("response body limit exceeded"));
                    }
                    emit(
                        &serde_json::json!({"type":"body","data":&buffer[..n]}),
                        context,
                        output,
                        &mut output_bytes,
                    )?;
                }
            } else {
                drop(response);
            }
            drop(agent);
            context.check()?;
            return Ok(HttpReport {
                status,
                final_url: current.to_string(),
                requests_attempted: *attempts,
                body_bytes,
                output_bytes,
            });
        }
    }
}
fn emit(
    value: &serde_json::Value,
    context: &ExecutionContext,
    output: &mut dyn OutputSink,
    used: &mut u64,
) -> Result<(), HttpError> {
    let mut bytes = serde_json::to_vec(value).map_err(|_| HttpError::Transport)?;
    bytes.push(b'\n');
    if bytes.len() as u64 > context.max_output_bytes().saturating_sub(*used) {
        return Err(ExecutionError::OutputLimit.into());
    }
    context.check()?;
    output.write_chunk(&bytes)?;
    *used += bytes.len() as u64;
    context.check()?;
    Ok(())
}
fn error_chain_timeout(error: &(dyn std::error::Error + 'static)) -> bool {
    let mut current = Some(error);
    for _ in 0..8 {
        let Some(error) = current else {
            break;
        };
        if error.downcast_ref::<io::Error>().is_some_and(|io| {
            matches!(
                io.kind(),
                io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock
            )
        }) {
            return true;
        }
        current = error.source();
    }
    false
}
/// Real HTTP backend bound to daemon network policy. Only the receipt-owned
/// harness path executes; generic/legacy entrypoints fail closed.
pub struct HttpFetchTool {
    scope: HttpScope,
}
impl HttpFetchTool {
    /// Bind explicit network/resource configuration, not execution approval.
    pub fn new(scope: HttpScope) -> Self {
        Self { scope }
    }
}
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Arguments {
    url: String,
    method: String,
}
impl crate::registry::ToolImplementation for HttpFetchTool {
    fn execute(&self, invocation: &crate::call::ToolInvocation) -> crate::result::ToolResult {
        crate::result::ToolResult::Inline(crate::result::InlineToolResult {
            call_id: invocation.call.call_id.clone(),
            is_error: true,
            content: serde_json::json!({"kind":"policy","message":"Receipt-owned HTTP execution required","details":{"reason":"http_receipt_required","requests_attempted":0}}),
        })
    }
    fn execute_with_context(
        &self,
        invocation: &crate::call::ToolInvocation,
        _context: &ExecutionContext,
        output: &mut dyn OutputSink,
    ) -> crate::result::ToolResult {
        output.abort();
        self.execute(invocation)
    }
    fn execute_http_fetch(
        &self,
        invocation: &crate::call::ToolInvocation,
        context: &ExecutionContext,
        output: &mut dyn OutputSink,
    ) -> Result<HttpReport, HttpFailure> {
        let request = (|| {
            context.check()?;
            if invocation.call.tool_id != "http.fetch" {
                return Err(HttpError::Policy("http.fetch identity required"));
            }
            let arguments: Arguments = serde_json::from_value(serde_json::Value::Object(
                invocation.call.arguments.clone(),
            ))
            .map_err(|_| HttpError::Policy("invalid HTTP arguments"))?;
            HttpRequest::new(&arguments.url, &arguments.method)
        })();
        match request {
            Ok(request) => self.scope.fetch(&request, context, output),
            Err(cause) => {
                output.abort();
                Err(HttpFailure {
                    cause,
                    requests_attempted: 0,
                })
            }
        }
    }
}
