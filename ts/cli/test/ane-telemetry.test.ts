import test from "node:test";
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { PowerMetricsCollector, type PowerMetricsSnapshot } from "../src/powermetrics.js";

type Progress = { event: "progress"; at_unix_ms: number; completed_requests: number };
type Ready = Omit<Progress, "event"> & {
  event: "ready"; effective_uid: number; pid: number; request: string; revision: number;
  selected_device: string; device_class: string; neural_engine_cores: number;
  load_duration_ms: number; warmup_completed_requests: number;
};
type Complete = Omit<Progress, "event"> & { event: "complete"; duration_ms: number };
type WorkloadFrame = Ready | Progress | Complete;

function parseFrame(line: string): WorkloadFrame {
  assert.ok(Buffer.byteLength(line) <= 4096, "Workload JSON line exceeds its bound.");
  const value: unknown = JSON.parse(line);
  assert.ok(value !== null && typeof value === "object" && !Array.isArray(value), "Invalid workload JSON.");
  const frame = value as Record<string, unknown>;
  for (const field of ["at_unix_ms", "completed_requests"]) {
    assert.ok(Number.isSafeInteger(frame[field]) && (frame[field] as number) >= 0, `Invalid workload ${field}.`);
  }
  assert.ok(["ready", "progress", "complete"].includes(String(frame.event)), "Unexpected workload event.");
  if (frame.event === "ready") {
    for (const field of ["effective_uid", "pid", "revision", "neural_engine_cores", "load_duration_ms", "warmup_completed_requests"]) {
      assert.ok(Number.isSafeInteger(frame[field]) && (frame[field] as number) > 0, `Invalid workload ${field}.`);
    }
    for (const field of ["request", "selected_device", "device_class"]) {
      assert.ok(typeof frame[field] === "string" && frame[field].length > 0 && frame[field].length <= 128, `Invalid workload ${field}.`);
    }
  }
  if (frame.event === "complete") {
    assert.ok(typeof frame.duration_ms === "number" && Number.isFinite(frame.duration_ms) && frame.duration_ms > 0,
      "Invalid workload duration.");
  }
  return frame as WorkloadFrame;
}

/** Own both the compiler and any descendants; never leave a test workload running. */
function startOwned(command: string, args: string[], signal: AbortSignal, timeoutMs: number, onLine?: (line: string) => void) {
  const child = spawn(command, args, { detached: true, stdio: ["ignore", "pipe", "pipe"] });
  let failure: Error | null = null;
  let stderr = "";
  let pending = "";
  let outputBytes = 0;
  let closed = false;
  let escalation: ReturnType<typeof setTimeout> | undefined;
  const killGroup = (name: NodeJS.Signals): void => {
    if (child.pid === undefined) return;
    try { process.kill(-child.pid, name); }
    catch (error) { if ((error as NodeJS.ErrnoException).code !== "ESRCH") failure ??= error as Error; }
  };
  const stop = (error: Error): void => {
    if (closed || failure) return;
    failure = error;
    killGroup("SIGTERM");
    escalation = setTimeout(() => { killGroup("SIGKILL"); }, 250);
  };
  const onAbort = (): void => { stop(new Error("ANE acceptance was cancelled.")); };
  const deadline = setTimeout(() => { stop(new Error(`${path.basename(command)} exceeded ${timeoutMs} ms.`)); }, timeoutMs);
  const completion = new Promise<void>((resolve, reject) => {
    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    const consume = (chunk: string, isStdout: boolean): void => {
      if (failure) return;
      outputBytes += Buffer.byteLength(chunk);
      if (outputBytes > 128 * 1024) { stop(new Error("ANE test child output exceeded 128 KiB.")); return; }
      if (!isStdout) { stderr = (stderr + chunk).slice(-8192); return; }
      if (!onLine) return;
      pending += chunk;
      let boundary: number;
      try {
        while ((boundary = pending.indexOf("\n")) !== -1) {
          const line = pending.slice(0, boundary);
          pending = pending.slice(boundary + 1);
          onLine(line);
        }
        assert.ok(Buffer.byteLength(pending) <= 4096, "Workload JSON line exceeds its bound.");
      } catch (error) { stop(error instanceof Error ? error : new Error(String(error))); }
    };
    child.stdout.on("data", (chunk: string) => { consume(chunk, true); });
    child.stderr.on("data", (chunk: string) => { consume(chunk, false); });
    child.once("error", (error) => { failure ??= error; });
    // Compiler descendants cannot keep inherited pipes alive after their parent exits.
    child.once("exit", () => { killGroup("SIGKILL"); });
    child.once("close", (code, name) => {
      closed = true;
      clearTimeout(deadline);
      clearTimeout(escalation);
      signal.removeEventListener("abort", onAbort);
      killGroup("SIGKILL");
      if (failure) reject(new Error(`${failure.message}${stderr ? `\n${stderr}` : ""}`));
      else if (code !== 0) reject(new Error(`${path.basename(command)} failed (${name ?? code}): ${stderr}`));
      else if (pending) reject(new Error("Workload left an unterminated JSON record."));
      else resolve();
    });
  });
  // The workload can fail while the test awaits an independent telemetry sample.
  void completion.catch(() => {});
  signal.addEventListener("abort", onAbort, { once: true });
  if (signal.aborted) onAbort();
  return {
    completion, pid: child.pid,
    get closed(): boolean { return closed; },
    async dispose(): Promise<void> { stop(new Error("ANE test child cleanup.")); await completion.catch(() => {}); },
  };
}

test("Apple Silicon ANE power becomes positive during explicitly assigned Vision execution without elevation", {
  skip: process.platform !== "darwin" || process.arch !== "arm64",
  timeout: 55_000,
}, async (t) => {
  assert.equal(typeof process.geteuid, "function");
  assert.notEqual(process.geteuid!(), 0, "Run ANE acceptance as a normal user.");
  const temporary = await fs.mkdtemp(path.join(os.tmpdir(), "alloy-ane-acceptance-"));
  const children: Array<ReturnType<typeof startOwned>> = [];
  const evidence: Record<string, unknown> = {};
  const samples: Array<{ phase: "baseline" | "load" | "recovery"; snapshot: PowerMetricsSnapshot }> = [];
  const collector = new PowerMetricsCollector({ enabled: true, intervalMs: 1000 });
  const sample = async (phase: "baseline" | "load" | "recovery"): Promise<PowerMetricsSnapshot> => {
    const snapshot = await collector.sample(t.signal);
    samples.push({ phase, snapshot });
    assert.equal(snapshot.status, "ready", `${phase}: ${snapshot.error ?? "native sample unavailable"}`);
    assert.equal(snapshot.source, "ioreport");
    assert.equal(snapshot.effective_uid, process.geteuid!());
    assert.ok(snapshot.ane_power_mw !== null && Number.isFinite(snapshot.ane_power_mw) && snapshot.ane_power_mw >= 0,
      `${phase}: ANE power missing; ${snapshot.warnings?.join("; ")}`);
    assert.ok(typeof snapshot.power_sources?.ane === "string" && snapshot.power_sources.ane.length > 0,
      `${phase}: ANE power needs an identified measured source.`);
    return snapshot;
  };
  try {
    const executable = path.join(temporary, "ane-workload");
    const source = fileURLToPath(new URL("../../test/fixtures/ane-workload.m", import.meta.url));
    const compiler = startOwned("/usr/bin/xcrun", ["--sdk", "macosx", "clang", "-fobjc-arc", "-O2",
      "-arch", "arm64", "-mmacosx-version-min=14.0", "-framework", "Foundation", "-framework", "Vision",
      "-framework", "CoreML", "-framework", "CoreGraphics", source, "-o", executable], t.signal, 30_000);
    children.push(compiler);
    await compiler.completion;
    await sample("baseline");

    const frames: WorkloadFrame[] = [];
    let readyFrame: Ready | undefined;
    let completeFrame: Complete | undefined;
    let readyResolve!: (frame: Ready) => void;
    let readyReject!: (error: unknown) => void;
    const readiness = new Promise<Ready>((resolve, reject) => { readyResolve = resolve; readyReject = reject; });
    const workload = startOwned(executable, [], t.signal, 15_000, (line) => {
      const frame = parseFrame(line);
      assert.ok(frames.length < 64, "Too many workload protocol records.");
      assert.equal(completeFrame, undefined, "Workload sent another record after completion.");
      const previous = frames.at(-1);
      if (previous) {
        assert.ok(frame.at_unix_ms >= previous.at_unix_ms, "Workload timestamps moved backwards.");
        assert.ok(frame.completed_requests >= previous.completed_requests, "Workload completion count moved backwards.");
      }
      frames.push(frame);
      if (frame.event === "ready") {
        assert.equal(readyFrame, undefined, "Workload became ready more than once.");
        assert.equal(frame.completed_requests, 0, "Warmup must not count as measured execution.");
        readyFrame = frame;
        readyResolve(frame);
      } else {
        assert.ok(readyFrame, "Workload progress arrived before readiness.");
        if (frame.event === "complete") completeFrame = frame;
      }
    });
    children.push(workload);
    void workload.completion.then(() => {
      if (!readyFrame) readyReject(new Error("Workload exited without successful ANE warmup."));
    }, readyReject);
    const ready = await readiness;
    evidence.workload = ready;
    assert.equal(ready.effective_uid, process.geteuid!());
    assert.equal(ready.pid, workload.pid);
    assert.equal(ready.selected_device, "MLNeuralEngineComputeDevice");
    assert.ok(["VNClassifyImageRequest", "VNGenerateImageFeaturePrintRequest"].includes(ready.request));
    assert.equal(ready.warmup_completed_requests, 1);
    assert.equal(ready.load_duration_ms, 6000);
    for (let index = 0; index < 4 && !workload.closed; index++) await sample("load");
    await workload.completion;
    assert.ok(completeFrame, "Workload exited without a completion record.");
    const complete = completeFrame as Complete;
    evidence.completion = complete;
    assert.ok(complete.completed_requests > 0, "No successful ANE requests completed after warmup.");
    await sample("recovery");

    const overlapping = samples.filter(({ phase, snapshot }) => {
      if (phase !== "load" || snapshot.sampled_at_unix_ms === null || snapshot.sample_duration_ms === null) return false;
      const end = snapshot.sampled_at_unix_ms;
      const start = end - snapshot.sample_duration_ms;
      if (start < ready.at_unix_ms || end > complete.at_unix_ms) return false;
      const progress = frames.filter((frame) => frame.event === "progress" && frame.at_unix_ms >= start && frame.at_unix_ms <= end);
      // Two distinct completion markers inside the measurement window prove
      // repeated successful execution during that actual counter interval.
      return progress.length >= 2 && progress.at(-1)!.completed_requests > progress[0]!.completed_requests;
    });
    evidence.overlapping_samples = overlapping.length;
    assert.ok(overlapping.length > 0, "No complete telemetry window overlapped repeated successful ANE execution.");
    assert.ok(overlapping.some(({ snapshot }) => snapshot.ane_power_mw! > 0),
      "ANE power stayed zero while ANE-selected Vision requests completed; the selected counter did not demonstrate responsiveness.");
    // Baseline/recovery may remain positive: this telemetry is system-wide and
    // other applications can use ANE. Do not invent an idle threshold or claim
    // that every operation, or all observed power, belongs to this workload.
  } finally {
    await Promise.all(children.map((child) => child.dispose()));
    await fs.rm(temporary, { recursive: true, force: true });
    evidence.samples = samples.map(({ phase, snapshot }) => ({ phase, status: snapshot.status,
      effective_uid: snapshot.effective_uid, at_unix_ms: snapshot.sampled_at_unix_ms,
      duration_ms: snapshot.sample_duration_ms, ane_power_mw: snapshot.ane_power_mw,
      ane_source: snapshot.power_sources?.ane, warnings: snapshot.warnings, error: snapshot.error }));
    t.diagnostic(JSON.stringify(evidence));
  }
});
