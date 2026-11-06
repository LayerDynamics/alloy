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
