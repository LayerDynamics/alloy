use std::{env, fs::OpenOptions, io::{self, BufRead, Write}, thread, time::Duration};
fn main() {
    let args: Vec<_> = env::args().collect();
    let id = &args[1];
    let mode = &args[2];
    let log = &args[3];
    std::fs::write(format!("{log}.{id}.pid"), std::process::id().to_string()).unwrap();
    for line in io::stdin().lock().lines() {
        let line = line.unwrap();
        let start = line.find("\"id\":").unwrap() + 5;
        let number: String = line[start..].trim_start().chars().take_while(|c| c.is_ascii_digit()).collect();
        let method = if line.contains("\"initialize\"") { "initialize" }
            else if line.contains("\"tool.before_execute\"") { "before" } else { "after" };
        writeln!(OpenOptions::new().create(true).append(true).open(log).unwrap(), "{id}:{method}").unwrap();
        let result = if method == "initialize" {
            if mode == "handshake_hang" { thread::sleep(Duration::from_secs(60)); return; }
            let version = if mode == "version" { 2 } else { 1 };
            let identity = if mode == "identity" { "wrong" } else { id };
            let plugin_version = if mode == "plugin_version" { "2.0.0" } else { "1.0.0" };
            let hooks = match mode.as_str() {
                "unsupported_hook" => r#"["session.start"]"#,
                "duplicate_hook" => r#"["tool.before_execute","tool.before_execute"]"#,
                "before_only" => r#"["tool.before_execute"]"#,
                "after_only" => r#"["tool.after_execute"]"#,
                _ => r#"["tool.before_execute","tool.after_execute"]"#,
            };
            format!(r#"{{"protocol_version":{version},"plugin_id":"{identity}","plugin_version":"{plugin_version}","hooks":{hooks}}}"#)
        } else {
            match mode.as_str() {
                "hang" => { thread::sleep(Duration::from_secs(60)); return; },
                "crash" => std::process::exit(3),
                "malformed" => { println!("not json"); io::stdout().flush().unwrap(); continue; },
                "flood" => { println!("{}", "x".repeat(65536)); io::stdout().flush().unwrap(); continue; },
                "stderr_flood" => { eprint!("{}", "x".repeat(65536)); io::stderr().flush().unwrap(); thread::sleep(Duration::from_secs(60)); return; },
                "partial" => { print!("{{"); io::stdout().flush().unwrap(); thread::sleep(Duration::from_secs(60)); return; },
                "duplicate_annotation" => format!(r#"{{"decision":"allow","annotations":{{"{id}.x":1,"{id}.x":2}}}}"#),
                "escaped_duplicate" => format!(r#"{{"decision":"allow","annotations":{{"{id}.x":1,"{id}.\u0078":2}}}}"#),
                "namespace" => r#"{"decision":"allow","annotations":{"other.x":true}}"#.into(),
                "long_reason" => format!(r#"{{"decision":"allow","reason":"{}"}}"#, "x".repeat(512)),
                "long_annotation" => format!(r#"{{"decision":"allow","annotations":{{"{id}.x":"{}"}}}}"#, "x".repeat(2048)),
                "mutation" => r#"{"decision":"allow","arguments":{"changed":true}}"#.into(),
                "after_failure" if method == "after" => r#"{"decision":"allow"}"#.into(),
                "fd_probe" if method == "before" => {
                    unsafe extern "C" { fn fcntl(fd: i32, cmd: i32, ...) -> i32; }
                    let fd: i32 = args[4].parse().unwrap();
                    // SAFETY: F_GETFD observes the descriptor; no vararg is used.
                    assert_eq!(unsafe { fcntl(fd, 1) }, -1, "inherited descriptor survived exec");
                    assert_eq!(io::Error::last_os_error().raw_os_error(), Some(9));
                    format!(r#"{{"decision":"allow","annotations":{{"{id}.fd_closed":true}}}}"#)
                },
                "probe" if method == "before" => {
                    assert_eq!(std::fs::read(&args[4]).unwrap(), b"allowed");
                    assert!(std::fs::read(&args[5]).is_err());
                    assert!(std::fs::write(&args[5], b"changed").is_err());
                    assert!(std::net::TcpStream::connect_timeout(&args[6].parse().unwrap(), Duration::from_millis(100)).is_err());
                    #[cfg(target_os = "macos")]
                    {
                        unsafe extern "C" { fn fork() -> i32; }
                        // SAFETY: test-only probe expects fork to be denied by Seatbelt.
                        let pid = unsafe { fork() };
                        if pid == 0 { std::process::exit(87); }
                        assert_eq!(pid, -1, "sandbox must deny child creation");
                    }
                    format!(r#"{{"decision":"allow","annotations":{{"{id}.probes":true}}}}"#)
                },
                _ if method == "before" => {
                    let decision = if mode == "veto" { "veto" } else { "allow" };
                    format!(r#"{{"decision":"{decision}","annotations":{{"{id}.seen":true}}}}"#)
                },
                _ => format!(r#"{{"annotations":{{"{id}.seen":true}}}}"#),
            }
        };
        let response_id = if mode == "wrong_id" && method != "initialize" { "999" } else { &number };
        let frame = format!(r#"{{"jsonrpc":"2.0","id":{response_id},"result":{result}}}"#);
        if mode == "fragmented" {
            for byte in frame.bytes().chain(std::iter::once(b'\n')) { io::stdout().write_all(&[byte]).unwrap(); io::stdout().flush().unwrap(); }
        } else {
            println!("{frame}"); io::stdout().flush().unwrap();
        }
        if mode == "unsolicited" && method == "initialize" {
            thread::sleep(Duration::from_millis(20));
            println!(r#"{{"jsonrpc":"2.0","id":2,"result":{{"decision":"allow"}}}}"#);
            io::stdout().flush().unwrap();
            std::fs::write(format!("{log}.{id}.unsolicited"), b"sent").unwrap();
        }
        if mode == "not_reading_after_initialize" && method == "initialize" {
            thread::sleep(Duration::from_secs(60));
            return;
        }
    }
}
