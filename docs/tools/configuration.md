# Explicit tool configuration

Start the daemon with a local, daemon-owned JSON configuration:

```sh
alloy start --tools-config /absolute/path/to/tools.json
```

The equivalent native option is `alloyd --artifact /absolute/model --tools-config /absolute/path/to/tools.json`. Tool configuration is selected at startup; it cannot be supplied by a model or changed through approval RPC. Supplying this option to the CLI when a daemon is already running fails with an explicit restart action instead of silently reusing different policy.

A complete example follows. Replace the absolute paths and destination with resources you intend to allow. The example authorizes only the exact `fs.read` argument object shown. The other four tools remain denied by the empty set of matching rules, despite their implementations being registered. No shell command is implicitly approved.

```json
{
  "read_root": "/absolute/workspace",
  "write_root": "/absolute/workspace",
  "shell_executables": ["/bin/echo"],
  "shell_cwd_roots": ["/absolute/workspace"],
  "http_destinations": [
    {"origin": "http://fixture.invalid/", "address": "127.0.0.1"}
  ],
  "policy": {
    "rules": [
      {
        "tool_id": "fs.read",
        "allowed_arguments": [{"path": "fixture.txt", "offset": 0, "length": 4096}],
        "require_approval": false,
        "timeout_ms": 1000,
        "max_output_bytes": 65536
      }
    ]
  },
  "plugins": []
}
```

The loader caps input at 1 MiB and rejects duplicate keys, unknown fields, excessive nesting, invalid policy, missing scopes, and unregistered policy tool IDs. All five existing W4 implementations are constructed with explicit filesystem, executable/cwd, and pinned origin/IP scopes. Executable aliases are subject to the existing W4 canonical-path rules. The daemon does not resolve a missing network destination from ambient DNS.

To load ordered local plugins, add objects with `id`, `version`, absolute `program`, `args`, absolute `cwd`, and `sandbox`. Sandbox choices are `{"mode":"trusted"}` or `{"mode":"mac_os","read_files":[],"read_trees":[],"write_trees":[]}` with explicitly declared resources. Trusted mode retains W5's documented trust boundary. Restricted mode retains the existing macOS W5 sandbox; unsupported restricted execution fails closed. All plugins handshake before the configured runtime is published.

The coordinator defaults are eight generation rounds, 32 calls per turn, 120 seconds for the whole turn, 8 KiB inline output, and at most 1 MiB collected output per builtin. Registry/policy limits may tighten output and execution duration. Engine rendering, token counting, context admission and cache compatibility remain authoritative; these coordinator ceilings never increase model context capacity. Tool mode also installs a 2 MiB complete-notification ceiling and a five-second socket write timeout; explicit paired RPC transport options override these defaults. Oversized notifications are rejected before writing a partial frame. Whole-turn expiry cancels the exact receipt-owned operation. Cancellation during an HTTP exchange retains W4's bounded synchronous exchange behavior.

`approval_requested` shows the daemon's exact invocation. The CLI accepts explicit yes/no from interactive input through a second socket while the send socket waits. EOF, Ctrl-C, and noninteractive input cancel; none grants approval. Tool data is JSON quoted, bounded in the display, and written to stderr; assistant text remains on stdout. The CLI does not evaluate policy.

Artifacts are stored beside the session database at the database path with its extension replaced by `artifacts`. History carries opaque artifact references; subsequent rendering does not open artifact files or elevate result text into system instructions. Restart recovers existing W3 intent state and never automatically reexecutes an effect.

Before executing a proposed batch, the daemon checks whether its bounded results can fit the selected profile. If necessary, it lowers the inline threshold for that batch; larger complete streaming results are stored as artifacts. If even the bounded results cannot fit, or the selected tokenizer has no verified projection bound, the batch is rejected before any tool intent or effect. The current native projection bound supports the pinned Qwen3-4B corpus tokenizer.

Large error results remain intact in the append-only session history. When an error's serialized content exceeds 1,024 bytes, the model receives an explicitly marked short preview saying that the full result is retained in session history. `alloy sessions resume` exposes the original event; the preview does not replace it.
