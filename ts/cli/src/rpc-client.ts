import net from "node:net";
import { formatRpcError, type JsonRpcError } from "./render.js";

const API_VERSION = 1;

export type RpcConnectionOptions = {
  timeoutMs?: number;
  signal?: AbortSignal;
};

export type SessionEvent = {
  session_id: string;
  event: string;
  body: Record<string, unknown>;
};

type Pending = {
  resolve: (value: unknown) => void;
  reject: (error: Error) => void;
  onEvent?: (event: SessionEvent) => void;
};

/** Preserve server codes and additive data while remaining an ordinary Error. */
export class RpcError extends Error {
  readonly code: number;
  readonly data: JsonRpcError["data"];

  constructor(error: JsonRpcError) {
    super(formatRpcError(error));
    this.name = "RpcError";
    this.code = error.code;
    this.data = error.data;
  }
}

export class RpcClient {
  #socket: net.Socket;
  #buf = "";
  #nextId = 1;
  #pending = new Map<number, Pending>();
  #streamId: number | null = null;

  private constructor(socket: net.Socket) {
    this.#socket = socket;
    socket.setEncoding("utf8");
    socket.on("data", (chunk: string) => {
      this.#onData(chunk);
    });
    socket.on("error", (error) => {
      this.#failAll(error);
    });
    socket.on("close", () => {
      this.#failAll(new Error("socket closed"));
    });
  }

  static connect(socketPath: string, options: RpcConnectionOptions = {}): Promise<RpcClient> {
    return new Promise((resolve, reject) => {
      if (options.signal?.aborted) {
        reject(new Error("RPC connection cancelled"));
        return;
      }
      if (options.timeoutMs !== undefined && (!Number.isFinite(options.timeoutMs) || options.timeoutMs <= 0)) {
        reject(new Error("RPC connection timeout must be positive"));
        return;
      }
      const socket = net.createConnection({ path: socketPath });
      let settled = false;
      let timer: ReturnType<typeof setTimeout> | undefined;
      const cleanup = (): void => {
        clearTimeout(timer);
        socket.off("error", onError);
        socket.off("connect", onConnect);
        socket.off("close", onClose);
        options.signal?.removeEventListener("abort", onAbort);
      };
      const onError = (error: Error): void => {
        if (settled) return;
        settled = true;
        cleanup();
        socket.destroy();
        reject(error);
      };
      const onAbort = (): void => { onError(new Error("RPC connection cancelled")); };
      const onClose = (): void => { onError(new Error("socket closed before RPC connection was established")); };
      const onConnect = (): void => {
        if (settled) return;
        settled = true;
        cleanup();
        resolve(new RpcClient(socket));
      };
      socket.once("error", onError);
      socket.once("connect", onConnect);
      socket.once("close", onClose);
      options.signal?.addEventListener("abort", onAbort, { once: true });
      if (options.timeoutMs !== undefined) {
        timer = setTimeout(() => { onError(new Error(`RPC connection timed out after ${options.timeoutMs} ms`)); }, options.timeoutMs);
      }
      if (options.signal?.aborted) onAbort();
    });
  }

  call(method: string, params: Record<string, unknown> = {}): Promise<unknown> {
    return this.#request(method, params);
  }

  callStreaming(
    method: string,
    params: Record<string, unknown>,
    onEvent: (event: SessionEvent) => void,
  ): Promise<unknown> {
    if (this.#streamId !== null) {
      return Promise.reject(new Error("overlapping streaming RPC is not supported"));
    }
    return this.#request(method, params, onEvent);
  }

  close(): void {
    this.#socket.destroy();
  }

  #request(
    method: string,
    params: Record<string, unknown>,
    onEvent?: (event: SessionEvent) => void,
  ): Promise<unknown> {
    const id = this.#nextId++;
    if (onEvent) {
      this.#streamId = id;
    }
    const line =
      JSON.stringify({
        jsonrpc: "2.0",
        id,
        api_version: API_VERSION,
        method,
        params,
      }) + "\n";

    return new Promise((resolve, reject) => {
      const pending: Pending = {
        resolve: (value) => {
          if (this.#streamId === id) {
            this.#streamId = null;
          }
          resolve(value);
        },
        reject: (error) => {
          if (this.#streamId === id) {
            this.#streamId = null;
          }
          reject(error);
        },
      };
      if (onEvent) {
        pending.onEvent = onEvent;
      }
      this.#pending.set(id, pending);
      this.#socket.write(line, (error) => {
        if (error) {
          this.#pending.delete(id);
          if (this.#streamId === id) {
            this.#streamId = null;
          }
          reject(error);
        }
      });
    });
  }

  #onData(chunk: string): void {
    this.#buf += chunk;
    let idx = this.#buf.indexOf("\n");
    while (idx >= 0) {
      const raw = this.#buf.slice(0, idx).trim();
      this.#buf = this.#buf.slice(idx + 1);
      if (raw) {
        this.#handleLine(raw);
      }
      idx = this.#buf.indexOf("\n");
    }
  }

  #handleLine(raw: string): void {
    let msg: Record<string, unknown>;
    try {
      const parsed: unknown = JSON.parse(raw);
      if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
        throw new Error("malformed JSON-RPC line: expected an object");
      }
      msg = parsed as Record<string, unknown>;
    } catch (error) {
      this.#failAll(
        error instanceof Error
          ? error
          : new Error(`malformed JSON-RPC line: ${raw}`),
      );
      return;
    }

    if (msg.method === "session.event" && msg.params && typeof msg.params === "object") {
      const params = msg.params as SessionEvent;
      if (this.#streamId !== null) {
        this.#pending.get(this.#streamId)?.onEvent?.(params);
      }
      return;
    }

    if (!("id" in msg) || msg.id === null || msg.id === undefined) {
      return;
    }

    const id = Number(msg.id);
    const pending = this.#pending.get(id);
    if (!pending) {
      return;
    }
    this.#pending.delete(id);
    if (msg.error) {
      pending.reject(new RpcError(msg.error as JsonRpcError));
      return;
    }
    pending.resolve(msg.result);
  }

  #failAll(error: Error): void {
    for (const pending of this.#pending.values()) {
      pending.reject(error);
    }
    this.#pending.clear();
    this.#streamId = null;
  }
}
