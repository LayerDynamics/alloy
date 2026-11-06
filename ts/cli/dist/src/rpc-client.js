import net from "node:net";
import { formatRpcError } from "./render.js";
const API_VERSION = 1;
export class RpcClient {
    #socket;
    #buf = "";
    #nextId = 1;
    #pending = new Map();
    #streamId = null;
    constructor(socket) {
        this.#socket = socket;
        socket.setEncoding("utf8");
        socket.on("data", (chunk) => {
            this.#onData(chunk);
        });
        socket.on("error", (error) => {
            this.#failAll(error);
        });
        socket.on("close", () => {
            this.#failAll(new Error("socket closed"));
        });
    }
    static connect(socketPath) {
        return new Promise((resolve, reject) => {
            const socket = net.createConnection({ path: socketPath });
            const onError = (error) => {
                reject(error);
            };
            socket.once("error", onError);
            socket.once("connect", () => {
                socket.off("error", onError);
                resolve(new RpcClient(socket));
            });
        });
    }
    call(method, params = {}) {
        return this.#request(method, params);
    }
    callStreaming(method, params, onEvent) {
        if (this.#streamId !== null) {
            return Promise.reject(new Error("overlapping streaming RPC is not supported"));
        }
        return this.#request(method, params, onEvent);
    }
    close() {
        this.#socket.destroy();
    }
    #request(method, params, onEvent) {
        const id = this.#nextId++;
        if (onEvent) {
            this.#streamId = id;
        }
        const line = JSON.stringify({
            jsonrpc: "2.0",
            id,
            api_version: API_VERSION,
            method,
            params,
        }) + "\n";
        return new Promise((resolve, reject) => {
            const pending = {
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
    #onData(chunk) {
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
    #handleLine(raw) {
        let msg;
        try {
            msg = JSON.parse(raw);
        }
        catch (error) {
            this.#failAll(error instanceof Error
                ? error
                : new Error(`malformed JSON-RPC line: ${raw}`));
            return;
        }
        if (msg.method === "session.event" && msg.params && typeof msg.params === "object") {
            const params = msg.params;
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
            pending.reject(new Error(formatRpcError(msg.error)));
            return;
        }
        pending.resolve(msg.result);
    }
    #failAll(error) {
        for (const pending of this.#pending.values()) {
            pending.reject(error);
        }
        this.#pending.clear();
        this.#streamId = null;
    }
}
//# sourceMappingURL=rpc-client.js.map