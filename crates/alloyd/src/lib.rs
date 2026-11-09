//! `alloyd` — Unix-domain JSON-RPC front door for harness-core + engine.
//!
//! Phase 1 exposes the B.6 chat/session/streaming subset over newline-delimited
//! JSON-RPC 2.0 with `api_version` negotiation.

#![deny(missing_docs)]

pub mod boot;
pub mod metrics;
pub mod rpc;
pub mod server;
pub mod service;
pub mod tool_config;

pub use boot::{BootRequest, BootedEngine, boot_qwen3};
pub use metrics::{LiveEngineMetrics, MetricsSource};
pub use rpc::{API_VERSION, ErrorKind, RpcError, RpcRequest, RpcResponse};
pub use server::{DaemonConfig, bind_listener, handle_connection, serve_listener};
pub use service::{DaemonService, FixtureEngine};
