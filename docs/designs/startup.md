# First-run setup and daemon startup

## Outcome and command boundaries

An unconfigured user should be able to run `alloy start` and reach an
API-ready foreground daemon without assembling a separate shell script.
The command reports the stage that failed and preserves existing artifacts
and configuration when provisioning fails. This is a bootstrap for the
existing frozen `qwen3-4b-4bit` profile.

| Command | Behavior |
| --- | --- |
| `alloy start` | Reuse a compatible daemon, or verify/configure the model, build/select the daemon and supervise it in the foreground |
| `alloy setup` / `alloy --setup` | Verify/configure a model and exit successfully without opening chat |
| `alloy setup --download` | Explicitly install the pinned starter, verify it and save configuration |
| `alloy doctor [--json]` | Read-only configuration, model preflight, API and launch-readiness report |
| `alloy monitor` | Read-only system/runtime monitoring, with a suggested next action when disconnected |
| `alloy chat` / bare `alloy` | Connect to an already-running compatible daemon |

From an uninstalled source checkout, use `node ts/cli/bin/alloy.js` in place
of `alloy` after `npm --prefix ts/cli ci` and `npm --prefix ts/cli run build`.

## Startup sequence

1. Probe the selected socket with a bounded `daemon.status` request. API
   version 1 and valid nonnegative status counters are required. Reuse a
   compatible listener without changing its configuration or taking ownership
   of its process. Permission errors, timeouts and incompatible listeners are
   reported before starting another process or provisioning a model.
2. Resolve an available launch command before provisioning. An explicit
   non-default `alloyd_bin` takes precedence, followed by `ALLOYD`. Otherwise
   the CLI's source checkout selects Cargo; installed mode selects `alloyd`
   from `PATH`. Missing tools produce an actionable failure. Finding Cargo
   does not establish that native compilation will succeed.
3. Verify the selected model against the bundled pinned runtime-file hashes.
   If unconfigured, try bounded local discovery, then ask for a local path,
   an explicit download, or cancellation. Noninteractive input never prompts
   or silently downloads.
4. Save a valid configuration atomically. Preserve valid saved socket,
   database, binary and corpus settings unless explicitly overridden. Invalid
   JSON/settings can be repaired through setup, but are replaced only after
   successful model verification. A missing inherited corpus file uses the
   bundled pins for the same starter; a missing explicitly supplied corpus
   file is an error.
5. In source mode, run `cargo build --locked -p alloyd` with structured Cargo
   output. Select the actual executable from its compiler-artifact message;
   never assume `target/debug`, and honor `CARGO_TARGET_DIR`. Build output and
   failures remain visible. The build has a separate thirty-minute deadline.
   Multi-GB model verification uses an optimized `sha2` dependency even in dev
   builds. The Qwen loader enables the pinned crate's ARM SHA backend on Apple
   Silicon; normal dependency feature unification applies it to engine and
   daemon verification too. Standalone Mac loader tests use the same backend.
   All expected digests and verification passes remain enforced.
6. Probe again after a build to avoid a duplicate listener if another daemon
   started meanwhile. Launch the owned native child with the selected model,
   profile, socket, database and corpus pins. New-launch readiness verifies
   both `daemon.status` and the `daemon.metrics` process ID against that child.
   A listener belonging to another process cannot satisfy this check. The
   readiness window is 120 seconds after the native build completes.
7. Print `READY — Alloy API 1`, keep supervising the process, and propagate
   its exit status. Ctrl+C/SIGTERM stop and await the owned child; build
   cancellation also stops its compiler descendants. An existing reused
   daemon is never stopped. There is no detached service, restart manager or
   background process left for the user to discover later.

The launcher supplies no CPU override. Native Apple-Silicon builds retain
the existing shared CPU/Metal behavior described in
[the monitor execution defaults](live-monitor.md#execution-defaults).

Run one foreground `start` for a socket and wait for its `READY` line. A
second launch during the first daemon's initialization can see no listener
yet; if the first daemon becomes ready before the second, the PID check
rejects the second child. A later `Already running` message refers to the
existing listener. Use its current `daemon.metrics.process_id` when collecting
process diagnostics; a PID printed before a restart is no longer authoritative.

Interactive chat is a text client; `monitor` is the read-only dashboard.
While a submitted turn has produced no text, chat reports that it is still
waiting on stderr, beginning at five seconds and then every fifteen seconds.
This reports a pending request, not a claim that loading or inference has
finished. Runtime warnings remain visible. A successful final response must
agree with streamed text; missing or inconsistent output is reported as an
error instead of returning an empty successful prompt.

## Model identity, discovery and publication

The bundled native-compatible manifest at `ts/cli/models/starter-model.json`
must exactly match the existing Qwen3 artifact identity in
`bench/corpus/models.json`: repository `mlx-community/Qwen3-4B-4bit`, revision
`4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25`, and its eleven per-file hashes.
A contract test enforces this relationship. This does not add another model
family, quantizer or registry.

Discovery checks only the selected/configured path, `ALLOY_QWEN3_ARTIFACT`,
`~/.alloy/models/qwen3-4b-4bit`, and
`~/alloy-omlx-models/alloy-qwen3-4b-4bit`. It does not recursively scan a user's
home directory. These are candidates, not a claim that they exist on a
particular machine.

Quick preflight checks bounded local metadata, the expected Qwen3/4-bit/group-64
configuration, tokenizer format, weight index, safetensors header and flat
artifact layout. Full setup verification additionally streams the runtime
files through SHA-256. All nine runtime files are required. `README.md` and
`.gitattributes` may be absent from an existing runtime artifact; if present,
their hashes are checked too. This is pinned runtime-file verification, not
a claim that an incomplete decorative snapshot has the full corpus aggregate.

Downloading requires `--download` on setup/start or an explicit prompt choice.
It uses Node's built-in HTTP facilities and the frozen full revision, streams
all eleven files into owned private storage, and verifies their hashes.
There are file, total-size, connection, inactivity and overall-time bounds;
progress is printed to stderr. No Python environment or model-conversion
step is required for this artifact.

Verified storage is published using an exclusive directory symlink at the
stable destination (`~/.alloy/models/qwen3-4b-4bit` by default). The native
loader canonicalizes the artifact root, so the files remain inside its
accepted flat directory. A destination that already exists, including one
created during download, is never overwritten. An already verified destination
can be reused. External per-file cache symlinks are rejected by the native
confinement rule; a root alias to an otherwise valid flat directory is allowed.

Failure, timeout or cancellation removes only owned unpublished storage.
Once a complete model is published it remains available even if a later
settings save fails. Configuration writes use an exclusive temporary file,
mode 0600, and atomic rename; failure preserves the prior settings file.

## Readiness and recovery

`doctor` has a stable JSON report (`schema_version: 1`) containing independent
configuration, model, daemon and launch findings; a readiness classification;
`next_command`; `next_step`; and an explicit validation scope. Exit status 0
means a compatible daemon is ready for chat. Other states exit 1.

A ready daemon may serve chat without a locally configured model; doctor
reports that usable state while retaining the local setup findings for a
future restart. Malformed settings or an unsupported profile still require
configuration repair. Socket access/API problems and unavailable launch tools
take precedence over asking the user to provision a new model.

Doctor never downloads, builds, starts a daemon or hashes all model weights.
Its bounded model preflight is distinct from setup's full runtime-file
verification. Finding a source launch plan is distinct from a successful
native build. The monitor caches similar setup advice and reloads saved
settings while disconnected, while keeping an explicit `--socket` fixed.
Neither path waits on the native execution queue or changes admission policy.

## Validation scope

Focused CLI coverage includes standalone setup, invalid/noninteractive
configuration, explicit download selection, interrupted/hash-failed installs,
publication races, settings preservation, hung RPCs, correct-PID readiness,
existing-listener reuse, foreground child/build cleanup, Cargo's executable
path, read-only diagnostics, and monitor ages after readiness advice.

Installer tests use small injected pinned artifacts and transport streams;
they do not download multi-GB weights in CI. API readiness is not proof of
native weight loading or generation: a real chat request must validate those
on the target host. Neither model provisioning nor this monitor/startup flow
completes the representative two-large-model constrained-budget M9 acceptance.
The broader release bootstrap, model registry, and Phase-5 environment and
wired-limit doctor report remain separate work.
