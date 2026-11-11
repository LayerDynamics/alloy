/** Node never signals a PID or process group: the live group leader owns it. */
import { spawn, spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import path from "node:path";

const SCRIPT = fileURLToPath(new URL("./group_supervisor.py", import.meta.url));
const withoutLoaderInjection = env => Object.fromEntries(Object.entries(env)
  .filter(([key]) => !key.startsWith("DYLD_") && !key.startsWith("LD_")));
// The installed smoke intentionally clears PATH. Resolve the supervisor's
// interpreter from the driver environment, not from the target environment.
// Isolated startup without site initialization keeps Python customization from
// starting competing waiters inside the supervisor.
// Apple's /usr/bin/python3 3.9 can lack os.waitid even though WNOWAIT exists.
const PYTHON = (() => {
  for (const candidate of ["python3", "/opt/homebrew/bin/python3", "/usr/local/bin/python3"]) {
    const probe = spawnSync(candidate, ["-I", "-S", "-c", "import os,sys; assert hasattr(os,'waitid') and hasattr(os,'WNOWAIT'); print(sys.executable)"],
      { env: withoutLoaderInjection(process.env), encoding: "utf8", timeout: 5000 });
    const absolute = probe.stdout?.trim();
    if (probe.status === 0 && absolute && path.isAbsolute(absolute)) return absolute;
  }
  return undefined;
})();


export function spawnOwned(command, args, { cwd, env }) {
  if (!PYTHON) throw new Error("Python 3 with waitid(WNOWAIT) is required for safe process supervision");
  const targetEnv = Object.fromEntries(Object.entries(env ?? process.env)
    .filter(([, value]) => value !== undefined).map(([key, value]) => [key, String(value)]));
  const child = spawn(PYTHON, ["-I", "-S", SCRIPT, command, ...args], {
    cwd, env: withoutLoaderInjection(targetEnv), detached: true,
    stdio: ["ignore", "pipe", "pipe", "pipe", "pipe"],
  });
  // The complete target environment is delivered privately after Python has
  // started with loader injection disabled; it is never put in argv or logs.
  const environment = child.stdio[4];
  environment.on("error", () => {}); // missing bootstrap is reported by done
  environment.end(JSON.stringify(targetEnv));
  const control = child.stdio[3];
  let buffer = "", outcome = null, launchError = null, retired = false;
  let readyResolve;
  const ready = new Promise(resolve => { readyResolve = resolve; });
  let targetPid = null;
  control.setEncoding("utf8");
  control.on("error", error => { if (error.code !== "EPIPE") launchError ??= error; });
  control.on("data", text => {
    buffer += text;
    let end;
    while ((end = buffer.indexOf("\n")) !== -1) {
      const line = buffer.slice(0, end); buffer = buffer.slice(end + 1);
      try {
        const item = JSON.parse(line);
        if (item.event === "ready" && Number.isSafeInteger(item.pid) && item.pid > 1) {
          targetPid = item.pid; readyResolve(targetPid);
        } else if (item.event === "exit") {
          outcome = item;
        } else { launchError ??= new Error("Invalid supervisor response"); }
      } catch (error) { launchError ??= error; }
    }
  });
  child.on("error", error => { launchError ??= error; });
  const done = new Promise(resolve => child.once("close", (code, signal) => {
    retired = true; readyResolve(targetPid);
    if (!outcome && !launchError) launchError = new Error(`Process supervisor exited without target status (${signal ?? code})`);
    if (outcome && !outcome.error && outcome.cleanup_verified !== true) launchError ??= new Error("Process-group cleanup was not verified");
    resolve({ code: outcome?.code ?? null, signal: outcome?.signal ?? null,
      error: launchError?.message ?? outcome?.error ?? null,
      live_before_cleanup: outcome?.live_before_cleanup ?? null });
  }));
  function request(signal, targetOnly = false) {
    if (retired || control.destroyed || !control.writable) return false;
    const command = targetOnly ? ({ SIGINT: "I", SIGTERM: "Q" })[signal] : ({ SIGTERM: "T", SIGKILL: "K" })[signal];
    if (!command) throw new Error(`Unsupported supervisor signal ${signal}`);
    control.write(command);
    return true;
  }
  return { child, ready, done, request, get group() { return targetPid; }, get targetPid() { return targetPid; }, get retired() { return retired; } };
}
