import { RpcClient } from "./rpc-client.js";
async function canConnect(socketPath) {
    try {
        const client = await RpcClient.connect(socketPath);
        client.close();
        return true;
    }
    catch {
        return false;
    }
}
/** Require a reachable alloyd socket; do not spawn the daemon. */
export async function ensureDaemon(settings) {
    if (await canConnect(settings.socket)) {
        return;
    }
    throw new Error(`alloyd is not running at ${settings.socket}; start alloyd, then retry`);
}
//# sourceMappingURL=daemon.js.map