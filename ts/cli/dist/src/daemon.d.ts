import type { AlloySettings } from "./settings.js";
/** Require a reachable alloyd socket; do not spawn the daemon. */
export declare function ensureDaemon(settings: AlloySettings): Promise<void>;
