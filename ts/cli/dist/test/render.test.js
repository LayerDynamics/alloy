import test from "node:test";
import assert from "node:assert/strict";
import { formatRpcError, textFromEvent } from "../src/render.js";
test("textFromEvent reads text_delta body.text", () => {
    assert.equal(textFromEvent({
        method: "session.event",
        params: { event: "text_delta", body: { text: "Hi" } },
    }), "Hi");
});
test("textFromEvent ignores non-delta", () => {
    assert.equal(textFromEvent({
        method: "session.event",
        params: { event: "generation_completed", body: {} },
    }), null);
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
//# sourceMappingURL=render.test.js.map