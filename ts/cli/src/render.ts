export type JsonRpcError = {
  code: number;
  message: string;
  data?: { kind?: string; [key: string]: unknown };
};

export function textFromEvent(line: {
  method?: string;
  params?: { event?: string; body?: { text?: string } };
}): string | null {
  if (line.method !== "session.event") {
    return null;
  }
  if (line.params?.event !== "text_delta") {
    return null;
  }
  const text = line.params.body?.text;
  return typeof text === "string" ? text : null;
}

export function formatRpcError(error: JsonRpcError): string {
  const kind = error.data?.kind;
  return kind
    ? `RPC ${error.code} (${kind}): ${error.message}`
    : `RPC ${error.code}: ${error.message}`;
}
