import test from "node:test";
import assert from "node:assert/strict";
import { runCli, type CliDependencies } from "../src/main.js";
import { defaultSettings } from "../src/settings.js";

const absent = { status: "unavailable" as const, error: "connect ENOENT", daemon_status: null, can_start: true };

function dependencies(): Partial<CliDependencies> {
  return {
    loadSettings: () => null,
    probeDaemon: async () => absent,
    resolveDaemonLaunch: async () => ({ kind: "source", command: "/cargo", cwd: "/checkout" }),
    runSetup: async () => { throw new Error("unexpected setup"); },
    startDaemon: async () => { throw new Error("unexpected daemon launch"); },
    ensureDaemon: async () => { throw new Error("unexpected chat connection"); },
    runChat: async () => { throw new Error("unexpected chat"); },
  };
}

test("setup and --setup complete successfully without connecting or falling through to chat", async () => {
  for (const command of ["setup", "--setup"]) {
    let configured = 0;
    const exit = await runCli(["node", "alloy", command, "--artifact", "/models/qwen"], { ...dependencies(),
      runSetup: async (partial) => { configured++; assert.equal(partial?.artifact, "/models/qwen"); return defaultSettings(); },
    });
    assert.equal(exit, 0);
    assert.equal(configured, 1);
  }
});

test("start performs setup before launching and propagates the actual daemon outcome", async () => {
  const calls: string[] = [];
  const configured = { ...defaultSettings(), artifact: "/verified/model" };
  const exit = await runCli(["node", "alloy", "start"], { ...dependencies(),
    runSetup: async () => { calls.push("setup"); return configured; },
    startDaemon: async (settings) => { calls.push("start"); assert.equal(settings, configured); return 7; },
  });
  assert.deepEqual(calls, ["setup", "start"]);
  assert.equal(exit, 7);
});

test("already ready daemon is reused without setup or download", async () => {
  const exit = await runCli(["node", "alloy", "start"], { ...dependencies(),
    probeDaemon: async () => ({ status: "ready", can_start: false, error: null,
      daemon_status: { api_version: 1, models_resident: 0, memory_reserved: 0, sessions_active: 0 } }),
    startDaemon: async () => 0,
  });
  assert.equal(exit, 0);
});

test("a ready daemon does not conceal configuration errors that would still block chat", async () => {
  const ready: NonNullable<Partial<CliDependencies>["probeDaemon"]> = async () => ({ status: "ready", can_start: false, error: null,
    daemon_status: { api_version: 1, models_resident: 0, memory_reserved: 0, sessions_active: 0 } });
  await assert.rejects(runCli(["node", "alloy", "start"], { ...dependencies(), probeDaemon: ready,
    loadSettings: () => { throw new Error("Invalid settings; run alloy setup"); },
  }), /Invalid settings/);
  await assert.rejects(runCli(["node", "alloy", "start"], { ...dependencies(), probeDaemon: ready,
    loadSettings: () => ({ ...defaultSettings(), profile: "unsupported" }),
  }), /alloy setup --profile qwen3-4b-4bit/);
});

test("blocked listener or missing build tools prevents provisioning and another launch", async () => {
  await assert.rejects(runCli(["node", "alloy", "start"], { ...dependencies(),
    probeDaemon: async () => ({ ...absent, can_start: false, error: "permission denied" }),
  }), /permission denied/);
  await assert.rejects(runCli(["node", "alloy", "start", "--download"], { ...dependencies(),
    resolveDaemonLaunch: async () => ({ kind: "unavailable", error: "cargo is missing" }),
  }), /cargo is missing/);
});

test("monitor and doctor preserve read-only behavior even with invalid settings", async () => {
  for (const command of ["monitor", "doctor"]) {
    let invoked = false;
    const deps = { ...dependencies(), loadSettings: () => { throw new Error("invalid settings"); } };
    if (command === "monitor") deps.runMonitor = async (_settings, _args, options) => {
      invoked = true; assert.match(options!.configurationError!, /invalid settings/); return 0;
    };
    else deps.runDoctor = async (_settings, options) => {
      invoked = true; assert.match(options!.configurationError!, /invalid settings/); return 0;
    };
    assert.equal(await runCli(["node", "alloy", command], deps), 0);
    assert.equal(invoked, true);
  }
});
