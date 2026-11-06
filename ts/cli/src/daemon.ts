import type { AlloySettings } from "./settings.js";
import { RpcClient } from "./rpc-client.js";

async function canConnect(socketPath: string): Promise<boolean> {
  try {
    const client = await RpcClient.connect(socketPath);
    client.close();
    return true;
  } catch {
    return false;
  }
}

/** Require a reachable alloyd socket; do not spawn the daemon. */
export async function ensureDaemon(settings: AlloySettings): Promise<void> {
  if (await canConnect(settings.socket)) {
    return;
  }
  throw new Error(
    `alloyd is not running at ${settings.socket}; start alloyd, then retry`,
  );
}
