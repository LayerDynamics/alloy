import net from "node:net";
import { formatRpcError, type JsonRpcError } from "./render.js";

const API_VERSION = 1;

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

  static connect(socketPath: string): Promise<RpcClient> {
    return new Promise((resolve, reject) => {
      const socket = net.createConnection({ path: socketPath });
      const onError = (error: Error) => {
        reject(error);
      };
      socket.once("error", onError);
      socket.once("connect", () => {
        socket.off("error", onError);
        resolve(new RpcClient(socket));
      });
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
      msg = JSON.parse(raw) as Record<string, unknown>;
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
      pending.reject(new Error(formatRpcError(msg.error as JsonRpcError)));
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
