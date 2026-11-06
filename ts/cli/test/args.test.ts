import test from "node:test";
import assert from "node:assert/strict";
import { parseArgv } from "../src/args.js";

test("parseArgv bare alloy defaults", () => {
  const args = parseArgv(["node", "alloy"]);
  assert.equal(args.forceSetup, false);
  assert.equal(args.help, false);
  assert.equal(args.message, undefined);
});

test("parseArgv one-shot message", () => {
  const args = parseArgv(["node", "alloy", "-m", "hello"]);
  assert.equal(args.message, "hello");
});

test("parseArgv chat alias with message", () => {
  const args = parseArgv([
    "node",
    "alloy",
    "chat",
    "--profile",
    "qwen3-4b-4bit",
    "-m",
    "hello",
  ]);
  assert.equal(args.profile, "qwen3-4b-4bit");
  assert.equal(args.message, "hello");
});

test("parseArgv --setup and --socket", () => {
  const args = parseArgv([
    "node",
    "alloy",
    "--setup",
    "--socket",
    "/tmp/x.sock",
    "--resume",
    "sess-9",
  ]);
  assert.equal(args.forceSetup, true);
  assert.equal(args.socket, "/tmp/x.sock");
  assert.equal(args.resume, "sess-9");
});

test("parseArgv rejects unknown argument", () => {
  assert.throws(() => parseArgv(["node", "alloy", "doctor"]), /unknown argument/);
});

test("monitor flags select a read-only one-shot JSON capture", () => {
  const args = parseArgv(["node", "alloy", "monitor", "--once", "--json", "--powermetrics", "--socket", "/tmp/monitor.sock", "--interval", "500"]);
  assert.equal(args.command, "monitor");
  assert.equal(args.once, true);
  assert.equal(args.json, true);
  assert.equal(args.powermetrics, true);
  assert.equal(args.intervalMs, 500);
  assert.equal(args.socket, "/tmp/monitor.sock");
});

test("monitor defaults and interval bounds are explicit", () => {
  const args = parseArgv(["node", "alloy", "monitor"]);
  assert.equal(args.intervalMs, 1000);
  assert.equal(args.once, false);
  assert.equal(args.json, false);
  assert.equal(args.powermetrics, false);
  assert.equal(parseArgv(["node", "alloy", "monitor", "--interval", "60000"]).intervalMs, 60000);
  for (const value of ["499", "60001", "-1", "1.5", "1e3", "NaN", ""]) {
    assert.throws(() => parseArgv(["node", "alloy", "monitor", "--interval", value]), /500 to 60000/);
  }
});

test("monitor rejects chat/setup mutations and chat rejects monitor flags", () => {
  for (const options of [["--setup"], ["-m", "hello"], ["--resume", "session"], ["--artifact", "model"]]) {
    assert.throws(() => parseArgv(["node", "alloy", "monitor", ...options]), /chat\/setup options do not apply/);
  }
  assert.throws(() => parseArgv(["node", "alloy", "--json"]), /only valid with alloy monitor/);
});
