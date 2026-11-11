import assert from "node:assert/strict";
import { spawn, spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { test } from "node:test";
import { spawnOwned } from "./group-supervisor.mjs";

const script = fileURLToPath(new URL("./group_supervisor.py", import.meta.url));
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
function alive(pid) {
  const p = spawnSync("/bin/ps", ["-p", String(pid), "-o", "stat="], { encoding: "utf8" });
  return p.status === 0 && p.stdout.trim() !== "" && !p.stdout.trim().startsWith("Z");
}
async function waitUntil(fn, message) {
  const deadline = Date.now() + 5000;
  while (!fn()) { assert(Date.now() < deadline, message); await sleep(20); }
}
function temp(t) {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "alloy-owned-supervisor-"));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  return root;
}

test("natural leader exit cleans stdio-holding descendant and late requests cannot hit another group", { timeout: 10000 }, async (t) => {
  const root = temp(t), marker = path.join(root, "descendant");
  const sentinel = spawnOwned(process.execPath, ["-e", "setInterval(() => {}, 1000)"], { env: process.env });
  t.after(async () => { sentinel.request("SIGKILL"); await sentinel.done; });
  const sentinelPid = await sentinel.ready;
  const code = `const {spawn}=require('node:child_process'); const fs=require('node:fs');\n` +
    `const c=spawn(process.execPath,['-e','process.on("SIGTERM",()=>{});setInterval(()=>{},1000)'],{stdio:'inherit'});` +
    `fs.writeFileSync(${JSON.stringify(marker)},String(c.pid)); process.exit(0);`;
  const owner = spawnOwned(process.execPath, ["-e", code], { env: process.env });
  const result = await owner.done;
  assert.equal(result.code, 0, result.error);
  const descendant = Number(fs.readFileSync(marker, "utf8"));
  assert(result.live_before_cleanup.includes(descendant));
  assert(!alive(descendant));
  assert.equal(owner.request("SIGKILL"), false);
  assert(alive(sentinelPid), "late requests must not reach an unrelated group");
});

test("TERM then KILL is delivered only by the owner and reports actual termination", { timeout: 10000 }, async () => {
  const p = spawnOwned(process.execPath, ["-e", 'process.on("SIGTERM",()=>{});process.stdout.write("ready");setInterval(()=>{},1000)'], { env: process.env });
  let output = ""; p.child.stdout.on("data", s => { output += s; });
  await p.ready; await waitUntil(() => output.includes("ready"), "target did not start");
  assert(p.request("SIGTERM")); await sleep(100);
  assert(p.request("SIGKILL"));
  const result = await p.done;
  assert.equal(result.signal, "SIGKILL", result.error);
  assert.equal(result.error, null);
});

test("parent death closes the private control channel and cleans the target", { timeout: 10000 }, async (t) => {
  const root = temp(t), marker = path.join(root, "pid");
  const url = new URL("./group-supervisor.mjs", import.meta.url).href;
  const code = `import {spawnOwned} from ${JSON.stringify(url)}; import fs from 'node:fs';\n` +
    `const p=spawnOwned(process.execPath,['-e',${JSON.stringify(`require('fs').writeFileSync(${JSON.stringify(marker)},String(process.pid));setInterval(()=>{},1000)`)}],{env:process.env});` +
    `while(!fs.existsSync(${JSON.stringify(marker)})) await new Promise(r=>setTimeout(r,10)); process.exit(0);`;
  const parent = spawn(process.execPath, ["--input-type=module", "-e", code], { stdio: "ignore" });
  await new Promise((resolve, reject) => { parent.once("error", reject); parent.once("close", resolve); });
  const pid = Number(fs.readFileSync(marker, "utf8"));
  await waitUntil(() => !alive(pid), "target survived loss of owner control pipe");
});

test("process-table failure kills the still-owned group and reports failure", { timeout: 10000 }, async (t) => {
  const root = temp(t), marker = path.join(root, "descendant");
  const command = `sleep 30 & echo $! > ${JSON.stringify(marker)}; exit 0`;
  const inject = `import runpy,subprocess,sys\ndef fail(*a,**k): raise OSError('injected process-table failure')\nsubprocess.run=fail\nsys.argv=[${JSON.stringify(script)},'/bin/sh','-c',${JSON.stringify(command)}]\nrunpy.run_path(sys.argv[0],run_name='__main__')`;
  const child = spawn("python3", ["-c", inject], { detached: true, stdio: ["ignore", "pipe", "pipe", "pipe", "pipe"] });
  child.stdio[3].on("data", () => {});
  child.stdio[4].end(JSON.stringify(process.env));
  const result = await new Promise(resolve => child.once("close", (code, signal) => resolve({ code, signal })));
  assert.notEqual(result.code, 0, "owner must report failed verification");
  const descendant = Number(fs.readFileSync(marker, "utf8"));
  await waitUntil(() => !alive(descendant), "failure path left a live owned descendant");
});

test("broken control channel during ready publication cleans the target", { timeout: 10000 }, async (t) => {
  const root = temp(t), marker = path.join(root, "target");
  const target = `require('fs').writeFileSync(${JSON.stringify(marker)},String(process.pid));setInterval(()=>{},1000)`;
  const inject = `import runpy,subprocess,sys,os,time\n` +
    `realpopen=subprocess.Popen\ndef start(*a,**k):\n p=realpopen(*a,**k)\n while not os.path.exists(${JSON.stringify(marker)}): time.sleep(.01)\n return p\n` +
    `subprocess.Popen=start\nrealwrite=os.write\ndef write(fd,data):\n if fd==3: raise BrokenPipeError('injected ready failure')\n return realwrite(fd,data)\nos.write=write\n` +
    `sys.argv=[${JSON.stringify(script)},${JSON.stringify(process.execPath)},'-e',${JSON.stringify(target)}]\nrunpy.run_path(sys.argv[0],run_name='__main__')`;
  const child = spawn("python3", ["-c", inject], { detached: true, stdio: ["ignore", "pipe", "pipe", "pipe", "pipe"] });
  child.stdio[3].on("data", () => {});
  child.stdio[4].end(JSON.stringify(process.env));
  const result = await new Promise(resolve => child.once("close", code => resolve(code)));
  assert.notEqual(result, 0);
  const pid = Number(fs.readFileSync(marker, "utf8"));
  await waitUntil(() => !alive(pid), "failed ready publication left its target alive");
});

test("isolated target retains its exact PATH while supervisor uses a qualified interpreter", async () => {
  const env = { ...process.env, PATH: "/usr/bin:/bin" };
  const p = spawnOwned(process.execPath, ["-e", "process.stdout.write(process.env.PATH)"], { env });
  let stdout = ""; p.child.stdout.on("data", data => { stdout += data; });
  const result = await p.done;
  assert.equal(result.code, 0, result.error);
  assert.equal(stdout.trim(), env.PATH);
});

test("supervisor resets inherited SIGCHLD auto-reaping before owning its child", { timeout: 10000 }, async () => {
  const wrapper = `import os,signal,sys\nsignal.signal(signal.SIGCHLD,signal.SIG_IGN)\nos.execv(sys.executable,[sys.executable,${JSON.stringify(script)},'/usr/bin/true'])`;
  const child = spawn("python3", ["-c", wrapper], { stdio: ["ignore", "pipe", "pipe", "pipe", "pipe"] });
  let protocol = "", stderr = "";
  child.stdio[3].setEncoding("utf8").on("data", text => { protocol += text; });
  child.stdio[4].end(JSON.stringify(process.env));
  child.stderr.setEncoding("utf8").on("data", text => { stderr += text; });
  const code = await new Promise(resolve => child.once("close", resolve));
  assert.equal(code, 0, stderr);
  const messages = protocol.trim().split("\n").map(line => JSON.parse(line));
  assert.equal(messages[0].event, "ready");
  assert.equal(messages[1].event, "exit");
  assert.equal(messages[1].code, 0);
  assert.equal(messages[1].cleanup_verified, true);
});

test("Python customization cannot start inside the supervisor while target receives its environment", { timeout: 10000 }, async (t) => {
  const root = temp(t), marker = path.join(root, "sitecustomize-ran");
  fs.writeFileSync(path.join(root, "sitecustomize.py"), `open(${JSON.stringify(marker)}, 'w').write('executed')\n`);
  const env = { ...process.env, PYTHONPATH: root };
  const target = spawnOwned(process.execPath, ["-e", "process.stdout.write(process.env.PYTHONPATH)"], { env });
  let stdout = ""; target.child.stdout.on("data", data => { stdout += data; });
  const result = await target.done;
  assert.equal(result.code, 0, result.error);
  assert.equal(stdout, root, "target must receive its intended environment");
  assert(!fs.existsSync(marker), "Python customization ran inside the supervisor");
});

test("native loader injection reaches target but cannot run in supervisor", { timeout: 10000 }, async (t) => {
  const root = temp(t), marker = path.join(root, "loader-pids"), source = path.join(root, "probe.c");
  const library = path.join(root, process.platform === "darwin" ? "probe.dylib" : "probe.so");
  fs.writeFileSync(source, `#include <stdio.h>\n#include <stdlib.h>\n#include <unistd.h>\n__attribute__((constructor)) static void probe(void) { const char *p=getenv("ALLOY_LOADER_MARKER"); if(p){ FILE *f=fopen(p,"a"); if(f){fprintf(f,"%ld\\n",(long)getpid());fclose(f);} } }\n`);
  const args = process.platform === "darwin" ? ["-dynamiclib", source, "-o", library] : ["-shared", "-fPIC", source, "-o", library];
  const built = spawnSync("cc", args, { encoding: "utf8", timeout: 30000 });
  assert.equal(built.status, 0, built.stderr);
  const key = process.platform === "darwin" ? "DYLD_INSERT_LIBRARIES" : "LD_PRELOAD";
  const env = { ...process.env, [key]: library, ALLOY_LOADER_MARKER: marker };
  const p = spawnOwned(process.execPath, ["-e", `process.stdout.write(process.env.${key})`], { env });
  let output = ""; p.child.stdout.on("data", data => { output += data; });
  const targetPid = await p.ready;
  const result = await p.done;
  assert.equal(result.code, 0, result.error);
  assert.equal(output, library, "target did not receive its native loader setting");
  assert.deepEqual(fs.readFileSync(marker, "utf8").trim().split("\n").map(Number), [targetPid],
    "native loader code ran in the supervisor or another process");
});
