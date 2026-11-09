use std::{
    io::{Read, Write},
    net::{TcpListener, TcpStream},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::Duration,
};
pub struct Server {
    pub origin: String,
    pub calls: Arc<Mutex<Vec<String>>>,
    stop: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
}
impl Server {
    pub fn new<F>(handler: F) -> Self
    where
        F: Fn(&str, &mut TcpStream, &AtomicBool) + Send + 'static,
    {
        Self::with_header_budget(handler, Duration::from_secs(3))
    }

    fn with_header_budget<F>(handler: F, header_budget: Duration) -> Self
    where
        F: Fn(&str, &mut TcpStream, &AtomicBool) + Send + 'static,
    {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let origin = format!("http://{}", listener.local_addr().unwrap());
        listener.set_nonblocking(true).unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let calls = Arc::new(Mutex::new(Vec::new()));
        let running = stop.clone();
        let observed = calls.clone();
        let worker = thread::spawn(move || {
            while !running.load(Ordering::Acquire) {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        // macOS inherits the nonblocking listener flag on accepted sockets.
                        // Use blocking reads so the configured per-read timeout applies.
                        stream.set_nonblocking(false).unwrap();
                        stream
                            .set_read_timeout(Some(Duration::from_millis(100)))
                            .unwrap();
                        stream
                            .set_write_timeout(Some(Duration::from_millis(100)))
                            .unwrap();
                        let mut bytes = Vec::new();
                        let mut block = [0u8; 512];
                        let header_started = std::time::Instant::now();
                        while !bytes.windows(4).any(|part| part == b"\r\n\r\n")
                            && bytes.len() < 8192
                        {
                            if running.load(Ordering::Acquire)
                                || header_started.elapsed() >= header_budget
                            {
                                break;
                            }
                            match stream.read(&mut block) {
                                Ok(0) => break,
                                Ok(n) => bytes.extend_from_slice(&block[..n]),
                                Err(error)
                                    if matches!(
                                        error.kind(),
                                        std::io::ErrorKind::WouldBlock
                                            | std::io::ErrorKind::TimedOut
                                            | std::io::ErrorKind::Interrupted
                                    ) =>
                                {
                                    continue;
                                }
                                Err(_) => break,
                            }
                        }
                        if !bytes.windows(4).any(|part| part == b"\r\n\r\n") {
                            continue;
                        }
                        let line = String::from_utf8_lossy(&bytes)
                            .lines()
                            .next()
                            .unwrap_or("")
                            .to_owned();
                        observed.lock().unwrap().push(line.clone());
                        handler(&line, &mut stream, &running);
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2))
                    }
                    Err(error) => panic!("fixture listener: {error}"),
                }
            }
        });
        Self {
            origin,
            calls,
            stop,
            worker: Some(worker),
        }
    }
}
impl Drop for Server {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            worker.join().unwrap();
        }
    }
}
pub fn response(stream: &mut TcpStream, status: u16, headers: &str, body: &[u8]) {
    let header = format!(
        "HTTP/1.1 {status} Fixture\r\nConnection: close\r\nContent-Length: {}\r\n{headers}\r\n",
        body.len()
    );
    let _ = stream.write_all(header.as_bytes());
    let _ = stream.write_all(body);
}

// Only test entrypoints acquire this guard, before creating contexts or servers.
// Server threads and concurrent fetch workers must never acquire it. This keeps
// short timing/cleanup fixtures independent of unrelated libtest scheduling.
static HTTP_TIMING_TESTS: std::sync::Mutex<()> = std::sync::Mutex::new(());
pub fn http_timing_test_guard() -> std::sync::MutexGuard<'static, ()> {
    let guard = HTTP_TIMING_TESTS
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    prepare_http_transport();
    guard
}

// Establish process-local loopback transport readiness once, before callers create
// the execution contexts whose short deadlines they are testing. On macOS the
// first outbound connect can remain pending after the listener accepts it; the
// same delay occurs with std TCP and ureq. This is infrastructure setup, not a
// retry or an extension of any tool's configured exchange/execution deadline.
// Use a separate endpoint so setup cannot change the tested server's call count.
fn prepare_http_transport() {
    static READY: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    READY.get_or_init(|| {
        let setup_budget = Duration::from_secs(10);
        let server = Server::with_header_budget(
            |_, stream, _| response(stream, 200, "", b"ready"),
            setup_budget,
        );
        let pinned: std::net::SocketAddr = server
            .origin
            .strip_prefix("http://")
            .unwrap()
            .parse()
            .unwrap();
        let agent = ureq::AgentBuilder::new()
            .redirects(0)
            .try_proxy_from_env(false)
            .max_idle_connections(0)
            .resolver(move |_: &str| Ok(vec![pinned]))
            .timeout_connect(setup_budget)
            .timeout_read(setup_budget)
            .timeout_write(setup_budget)
            .timeout(setup_budget)
            .build();
        let reply = agent
            .get(&server.origin)
            .call()
            .expect("loopback HTTP fixture readiness failed before tool deadlines were created");
        assert_eq!(reply.status(), 200);
        let mut body = Vec::new();
        reply.into_reader().take(6).read_to_end(&mut body).unwrap();
        assert_eq!(body, b"ready");
        assert_eq!(
            server.calls.lock().unwrap().len(),
            1,
            "readiness must not retry"
        );
        // Both the client reader and fixture-owned server/worker are closed here,
        // before READY is published and the caller allocates any timed context.
    });
}
