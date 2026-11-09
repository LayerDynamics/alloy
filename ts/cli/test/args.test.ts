import test from "node:test";
import assert from "node:assert/strict";
import { parseArgv } from "../src/args.js";

test("parseArgv bare alloy defaults", () => {
  const args = parseArgv(["node", "alloy"]);
  assert.equal(args.forceSetup, false);
  assert.equal(args.help, false);
  assert.equal(args.version, false);
  assert.equal(args.message, undefined);
});

test("help and version flags are available without a subcommand", () => {
  for (const flag of ["--help", "-h"]) {
    assert.equal(parseArgv(["node", "alloy", flag]).help, true);
  }
  for (const flag of ["--version", "-V"]) {
    assert.equal(parseArgv(["node", "alloy", flag]).version, true);
  }
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
  ]);
  assert.equal(args.forceSetup, true);
  assert.equal(args.socket, "/tmp/x.sock");
  assert.equal(args.command, "setup");
});

test("parseArgv rejects unknown argument", () => {
  assert.throws(() => parseArgv(["node", "alloy", "unknown-command"]), /unknown argument/);
});

test("startup commands keep provisioning explicit and read-only commands reject it", () => {
  const setup = parseArgv(["node", "alloy", "setup", "--download", "--artifact", "/models/qwen", "--corpus-models", "/pins.json"]);
  assert.equal(setup.command, "setup");
  assert.equal(setup.download, true);
  assert.equal(setup.artifact, "/models/qwen");
  assert.equal(setup.corpusModels, "/pins.json");
  assert.equal(parseArgv(["node", "alloy", "start"]).command, "start");
  assert.equal(parseArgv(["node", "alloy", "doctor", "--json"]).json, true);
  for (const command of ["chat", "doctor", "monitor"]) {
    assert.throws(() => parseArgv(["node", "alloy", command, "--download"]));
  }
  for (const command of ["setup", "start", "doctor"]) {
    assert.throws(() => parseArgv(["node", "alloy", command, "-m", "hello"]), /chat messages/);
  }
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

test("tool configuration is an explicit startup option and cannot silently affect chat", () => {
  assert.equal(parseArgv(["node", "alloy", "start", "--tools-config", "/fixture/tools.json"]).toolsConfig, "/fixture/tools.json");
  assert.throws(() => parseArgv(["node", "alloy", "--tools-config", "/fixture/tools.json"]), /only valid with alloy start/);
});
