export type JsonRpcError = {
  code: number;
  message: string;
  data?: { kind?: string; [key: string]: unknown };
};

/** Additive API v1 engine diagnostics; unknown kinds retain their message. */
export type EngineDiagnostic = {
  kind: string;
  message: string;
  [key: string]: unknown;
};

export function engineDiagnostic(value: unknown): EngineDiagnostic | undefined {
  if (value === null || typeof value !== "object" || Array.isArray(value)) return undefined;
  const diagnostic = value as Record<string, unknown>;
  return typeof diagnostic.kind === "string" && typeof diagnostic.message === "string"
    ? diagnostic as EngineDiagnostic : undefined;
}

function bytes(value: unknown): number | undefined {
  return typeof value === "number" && Number.isSafeInteger(value) && value >= 0 ? value : undefined;
}

function engineGuidance(error: EngineDiagnostic): string | undefined {
  if (error.kind === "busy") return "Wait for the active request to finish, or cancel that request explicitly.";
  if (error.kind === "protected_resources") {
    return "The caller holding these model pins or session-state leases must release the relevant handles before retrying.";
  }
  if (error.kind !== "memory_limit_exceeded") return undefined;
  switch (error.reason) {
    case "host_preflight":
      return "The planned startup exceeds the current safe memory ceiling. Reduce memory use or model/request demand, then retry with a fresh observation.";
    case "reservation_budget": {
      const required = bytes(error.required_bytes), budget = bytes(error.budget_bytes);
      return required !== undefined && budget !== undefined && required > budget
        ? "The request exceeds the full reservation budget. Use a smaller model or reduce the request; reclaiming idle resources cannot make it fit under the same budget."
        : "Not enough unreserved memory is available. Release eligible idle resources or reduce the request before retrying.";
    }
    case "allocator_ceiling":
      return "The projected native allocation exceeds the allocator ceiling. Reduce the model or request, or release eligible idle resources before retrying.";
    case "advisory_estimate":
      return "The advisory estimate exceeds currently available memory. Reduce the request or release eligible idle resources before retrying.";
    case "admissions_paused":
      return "New admissions are paused by the memory guard. Wait for memory pressure to fall before retrying.";
    case "buffer_limit":
      return "An allocation exceeds its configured buffer limit. Reduce the model or request, or inspect that limit before retrying.";
    default:
      return undefined;
  }
}

function withEngineGuidance(message: string, diagnostic: EngineDiagnostic | undefined): string {
  const guidance = diagnostic === undefined ? undefined : engineGuidance(diagnostic);
  return guidance === undefined ? message : `${message}\n${guidance}`;
}

/** Failure notifications and final RPC errors use the same typed guidance. */
export function formatRequestFailed(body: Record<string, unknown>): string {
  const diagnostic = engineDiagnostic(body.engine_error);
  const message = typeof body.message === "string" ? body.message : diagnostic?.message ?? JSON.stringify(body);
  const kind = diagnostic === undefined ? "" : ` (${diagnostic.kind})`;
  return withEngineGuidance(`request_failed${kind}: ${message}`, diagnostic);
}

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
  const diagnostic = engineDiagnostic(error.data?.engine_error);
  const classification = [kind, diagnostic?.kind].filter((value) => typeof value === "string" && value.length > 0).join("/");
  return withEngineGuidance(classification
    ? `RPC ${error.code} (${classification}): ${error.message}`
    : `RPC ${error.code}: ${error.message}`, diagnostic);
}
