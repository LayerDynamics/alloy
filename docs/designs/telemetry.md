# Anonymous telemetry delivery (FR-35)

The harness collects only the aggregate fields in
[`telemetry-v1.json`](../events/telemetry-v1.json). The schema and the required
always-on collection contract are unchanged. Prompts, conversation events,
model identities, artifact identities, and filesystem paths are not added to
telemetry. The telemetry client never reads session history or writes a disk
queue.

## Worker and batch ownership

One `TelemetryClient` owns one transport worker. Recording a session or turn
only updates fixed-size, saturating in-memory counters. Completing a turn
calls `request_flush`, which submits a nonblocking signal through a
capacity-one channel. Repeated requests coalesce while a POST is running.
The queue carries one signal rather than a list of per-turn payloads; there
is at most one serialized batch in flight alongside the fixed-size counters.

The worker also wakes at a 24-hour deadline. Session-created counts are
therefore flushed even when no turn completes. Each new deadline is measured
from the start of the previous attempt, so transport latency does not keep
pushing the cadence later. Empty batches do not send requests.

The worker atomically takes the current batch before serialization and I/O.
No counter or session-store lock is held across a network call. Counts
recorded during that call form the next batch. Success or failure of the old
batch cannot erase those newer counts. A failed batch is dropped and is
never retried or written to disk.

## Transport and failure boundaries

The default transport enforces HTTPS, follows no redirects, and sets
explicit 10-second connection, read, write, and overall request deadlines.
Responses are checked for success without collecting their bodies. Logs and
explicit flush errors contain fixed failure classes; they omit payloads,
endpoint URLs or credentials, and arbitrary transport error text.

The pinned `ureq` implementation cannot interrupt the operating system's
DNS resolver. That limitation cannot block generation: all transport calls
remain on the one worker. No per-attempt timeout helper thread is spawned.
Dropping the client signals shutdown and disconnects the request channel;
it does not join an in-flight transport. A normal idle worker exits when the
channel disconnects, and an in-flight worker exits after its call returns.
No new batch is sent after the shutdown signal is observed.

`flush()` remains available for explicit administrative use and tests. It
uses the same worker, returns immediately when the signal queue is already
full, and waits at most 12 seconds for an accepted request's acknowledgment.
An acknowledgment timeout does not restart the request or assert that the
remote server did not receive it. Production turn completion uses
`request_flush()` and does not wait for an acknowledgment.

## Verification and release prerequisites

The harness tests cover the published schema and offline drop, scheduled
flush with no completed turn, two completed/persisted turns while the
transport stays blocked, one worker with coalesced requests, preservation of
new counts across a failed POST, redacted failures, bounded explicit waiting
and client destruction, rejection of cleartext, and an actual loopback TLS
peer that accepts TCP but never completes the handshake.

These tests establish client behavior; they do not establish a deployed
collection endpoint. The current daemon default is the non-routable
`https://telemetry.invalid/v1`, with `--telemetry-endpoint` available for an
explicit configured destination. A real project endpoint and published
privacy policy are release prerequisites before a tagged emitting
pre-release. This change does not invent either or claim that test traffic
proves production delivery.
