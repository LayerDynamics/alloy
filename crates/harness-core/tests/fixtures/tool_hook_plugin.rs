use std::{env, fs::OpenOptions, io::{self, BufRead, Write}, thread, time::Duration};
fn main() {
    let args: Vec<_> = env::args().collect();
    let id = &args[1];
    let mode = &args[2];
    let log = &args[3];
    for line in io::stdin().lock().lines() {
        let line = line.unwrap();
        let start = line.find("\"id\":").unwrap() + 5;
        let number: String = line[start..].trim_start().chars().take_while(|c| c.is_ascii_digit()).collect();
        let method = if line.contains("\"initialize\"") { "initialize" }
            else if line.contains("\"tool.before_execute\"") { "before" } else { "after" };
        writeln!(OpenOptions::new().create(true).append(true).open(log).unwrap(), "{id}:{method}").unwrap();
        let result = if method == "initialize" {
            let version = if mode == "version" { 2 } else { 1 };
            format!(r#"{{"protocol_version":{version},"plugin_id":"{id}","plugin_version":"1.0.0","hooks":["tool.before_execute","tool.after_execute"]}}"#)
        } else {
            match mode.as_str() {
                "hang" => { thread::sleep(Duration::from_secs(60)); return; },
                "crash" => std::process::exit(3),
                "malformed" => { println!("not json"); io::stdout().flush().unwrap(); continue; },
                "flood" => { println!("{}", "x".repeat(65536)); io::stdout().flush().unwrap(); continue; },
                "mutation" => r#"{"decision":"allow","arguments":{"changed":true}}"#.into(),
                "after_failure" if method == "after" => r#"{"decision":"allow"}"#.into(),
                _ if method == "before" => {
                    let decision = if mode == "veto" { "veto" } else { "allow" };
                    format!(r#"{{"decision":"{decision}","annotations":{{"{id}.seen":true}}}}"#)
                },
                _ => format!(r#"{{"annotations":{{"{id}.seen":true}}}}"#),
            }
        };
        println!(r#"{{"jsonrpc":"2.0","id":{number},"result":{result}}}"#);
        io::stdout().flush().unwrap();
    }
}
