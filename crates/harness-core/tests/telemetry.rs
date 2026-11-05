//! Named acceptance: `telemetry_schema_conforms`, `telemetry_offline_drop`.

use harness_core::telemetry::{TELEMETRY_SCHEMA_V1, TelemetryTransport};
use harness_core::{HardwareTier, TelemetryClient, TelemetryPayload};
use std::sync::{Arc, Mutex};

struct RecordingTransport {
    posts: Mutex<Vec<(String, String)>>,
    fail: bool,
}

impl TelemetryTransport for RecordingTransport {
    fn post_json(&self, endpoint: &str, body: &str) -> Result<(), String> {
        self.posts
            .lock()
            .expect("lock")
            .push((endpoint.to_owned(), body.to_owned()));
        if self.fail {
            Err("unreachable".into())
        } else {
            Ok(())
        }
    }
}

#[test]
fn telemetry_schema_conforms() {
    let schema: serde_json::Value =
        serde_json::from_str(TELEMETRY_SCHEMA_V1).expect("schema parses");
    assert_eq!(schema["title"], "Alloy anonymous telemetry payload v1");
    assert_eq!(schema["additionalProperties"], false);
    let required = schema["required"].as_array().expect("required");
    for field in [
        "schema_version",
        "daemon_version",
        "hardware_tier",
        "session_count",
        "turn_count",
        "tokens_per_s",
        "ttft_ms",
        "cache_hit_rate",
        "tool_latency_ms",
        "memory_reserved_bytes",
        "memory_actual_bytes",
    ] {
        assert!(
            required.iter().any(|value| value == field),
            "schema missing required {field}"
        );
    }

    let transport = Arc::new(RecordingTransport {
        posts: Mutex::new(Vec::new()),
        fail: false,
    });
    let client = TelemetryClient::with_transport(
        "0.1.0",
        HardwareTier::Ram36Gb,
        "https://telemetry.example.test/v1",
        Box::new(CloneTransport(Arc::clone(&transport))),
    );
    client.record_session();
    client.record_turn();
    client.record_turn();

    let payload = client.snapshot().expect("snapshot");
    payload.assert_anonymous().expect("anonymous");
    assert_eq!(payload.schema_version, "telemetry-v1");
    assert_eq!(payload.session_count, 1);
    assert_eq!(payload.turn_count, 2);
    assert_eq!(payload.hardware_tier, HardwareTier::Ram36Gb);

    let json = serde_json::to_value(&payload).expect("json");
    for key in json.as_object().expect("object").keys() {
        let lower = key.to_ascii_lowercase();
        assert!(!lower.contains("prompt"));
        assert!(!lower.contains("path"));
        assert!(!lower.contains("model"));
        assert!(!lower.contains("artifact"));
        assert!(!lower.contains("tokenizer"));
    }

    // Required keys present; no extras beyond the schema property set.
    let properties = schema["properties"].as_object().expect("properties");
    for key in json.as_object().expect("object").keys() {
        assert!(
            properties.contains_key(key),
            "payload key {key} not in schema properties"
        );
    }
    for key in properties.keys() {
        assert!(
            json.get(key).is_some(),
            "payload missing schema property {key}"
        );
    }

    client.flush().expect("flush ok");
    let posts = transport.posts.lock().expect("lock");
    assert_eq!(posts.len(), 1);
    let body: TelemetryPayload = serde_json::from_str(&posts[0].1).expect("body");
    assert_eq!(body.turn_count, 2);
}

#[test]
fn telemetry_offline_drop() {
    let transport = Arc::new(RecordingTransport {
        posts: Mutex::new(Vec::new()),
        fail: true,
    });
    let client = TelemetryClient::with_transport(
        "0.1.0",
        HardwareTier::Ram8Gb,
        "https://127.0.0.1:1/unreachable",
        Box::new(CloneTransport(Arc::clone(&transport))),
    );
    client.record_session();
    client.record_turn();

    let err = client.flush().expect_err("unreachable must fail");
    assert!(
        err.to_string().contains("counts dropped"),
        "error should note offline drop: {err}"
    );

    // Counts were dropped — no on-disk queue; snapshot is zeroed.
    let after = client.snapshot().expect("snapshot after drop");
    assert_eq!(after.session_count, 0);
    assert_eq!(after.turn_count, 0);

    // A later successful path still works with a fresh client.
    let ok_transport = Arc::new(RecordingTransport {
        posts: Mutex::new(Vec::new()),
        fail: false,
    });
    let ok_client = TelemetryClient::with_transport(
        "0.1.0",
        HardwareTier::Ram8Gb,
        "https://telemetry.example.test/v1",
        Box::new(CloneTransport(Arc::clone(&ok_transport))),
    );
    ok_client.record_turn();
    ok_client.flush().expect("ok flush");
    assert_eq!(ok_transport.posts.lock().expect("lock").len(), 1);
}

/// `Box<dyn TelemetryTransport>` is not Clone; wrap Arc for dual ownership in tests.
struct CloneTransport(Arc<RecordingTransport>);

impl TelemetryTransport for CloneTransport {
    fn post_json(&self, endpoint: &str, body: &str) -> Result<(), String> {
        self.0.post_json(endpoint, body)
    }
}
