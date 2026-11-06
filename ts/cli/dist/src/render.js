export function textFromEvent(line) {
    if (line.method !== "session.event") {
        return null;
    }
    if (line.params?.event !== "text_delta") {
        return null;
    }
    const text = line.params.body?.text;
    return typeof text === "string" ? text : null;
}
export function formatRpcError(error) {
    const kind = error.data?.kind;
    return kind
        ? `RPC ${error.code} (${kind}): ${error.message}`
        : `RPC ${error.code}: ${error.message}`;
}
//# sourceMappingURL=render.js.map