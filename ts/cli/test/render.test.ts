import test from "node:test";
import assert from "node:assert/strict";
import { formatRequestFailed, formatRpcError, textFromEvent } from "../src/render.js";

test("textFromEvent reads text_delta body.text", () => {
  assert.equal(
    textFromEvent({
      method: "session.event",
      params: { event: "text_delta", body: { text: "Hi" } },
    }),
    "Hi",
  );
});

test("engine guidance distinguishes active work, protected ownership and each memory constraint", () => {
  const cases = [
    { cause: { kind: "busy" }, expected: /Wait for the active request/, absent: /caller holding these model pins/ },
    { cause: { kind: "protected_resources" }, expected: /caller holding these model pins or session-state leases/, absent: /Wait for the active request/ },
    { cause: { kind: "memory_limit_exceeded", reason: "host_preflight" }, expected: /planned startup exceeds the current safe memory ceiling/, absent: /Wait for/ },
    { cause: { kind: "memory_limit_exceeded", reason: "reservation_budget", required_bytes: 101, available_bytes: 20, budget_bytes: 100 }, expected: /exceeds the full reservation budget/, absent: /Not enough unreserved/ },
    { cause: { kind: "memory_limit_exceeded", reason: "reservation_budget", required_bytes: 50, available_bytes: 20, budget_bytes: 100 }, expected: /Not enough unreserved memory/, absent: /exceeds the full reservation budget/ },
    { cause: { kind: "memory_limit_exceeded", reason: "allocator_ceiling" }, expected: /native allocation exceeds the allocator ceiling/, absent: /full reservation budget/ },
    { cause: { kind: "memory_limit_exceeded", reason: "advisory_estimate" }, expected: /advisory estimate exceeds currently available memory/, absent: /full reservation budget/ },
    { cause: { kind: "memory_limit_exceeded", reason: "admissions_paused", required_bytes: 10, available_bytes: 100 }, expected: /paused by the memory guard/, absent: /Not enough unreserved|exceeds the full/ },
    { cause: { kind: "memory_limit_exceeded", reason: "buffer_limit" }, expected: /configured buffer limit/, absent: /paused by the memory guard/ },
  ];
  for (const { cause, expected, absent } of cases) {
    const diagnostic = { ...cause, message: "server explanation" };
    const final = formatRpcError({ code: -32000, message: diagnostic.message, data: { kind: "engine_failed", engine_error: diagnostic } });
    const event = formatRequestFailed({ message: diagnostic.message, engine_error: diagnostic, partial_usage: { input_tokens: 1, output_tokens: 0 } });
    assert.match(final, expected);
    assert.doesNotMatch(final, absent);
    assert.match(event, expected);
    assert.equal(final.split("\n")[1], event.split("\n")[1], "event and response give identical guidance");
    assert.match(final, /server explanation/);
  }
});

test("legacy, malformed and unknown engine diagnostics preserve the original server explanation", () => {
  for (const engine_error of [undefined, null, [], { kind: "protected_resources" }, { kind: "future_kind", message: "future detail" }, { kind: "memory_limit_exceeded", reason: "future_reason", message: "future detail" }]) {
    const formatted = formatRpcError({ code: -32000, message: "original server explanation", data: { kind: "engine_failed", engine_error } });
    assert.match(formatted, /original server explanation/);
    assert.doesNotMatch(formatted, /\n/, "unknown data must not invent recovery guidance");
  }
  assert.equal(formatRequestFailed({ message: "legacy terminal failure" }), "request_failed: legacy terminal failure");
});

test("textFromEvent ignores non-delta", () => {
  assert.equal(
    textFromEvent({
      method: "session.event",
      params: { event: "generation_completed", body: {} },
    }),
    null,
  );
});

test("formatRpcError includes kind when present", () => {
  const message = formatRpcError({
    code: -32000,
    message: "nope",
    data: { kind: "session_not_found" },
  });
  assert.match(message, /session_not_found/);
  assert.match(message, /nope/);
});
