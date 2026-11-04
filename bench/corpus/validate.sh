#!/usr/bin/env bash
# Corpus structural checks for Phase 0 tests 13-16.
# Run from bench/corpus, or from anywhere: scripts resolve relative to this file.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

fail=0
say() { printf '%s\n' "$*"; }
bad() { printf 'FAIL %s\n' "$*" >&2; fail=1; }

need() {
  command -v "$1" >/dev/null 2>&1 || { echo "missing $1" >&2; exit 2; }
}
need python3

sha256_file() {
  if command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | awk '{print $1}'
  else
    sha256sum "$1" | awk '{print $1}'
  fi
}

python3 - <<'PY' || bad "corpus_manifest_validates"
import json, sys
from pathlib import Path
root = Path(".")
manifest = json.loads((root / "manifest.json").read_text())
schema = json.loads((root / "manifest.schema.json").read_text())
try:
    import jsonschema
except ImportError:
    print("SKIP schema: python package jsonschema is not installed")
    print("structural manifest parse: ok")
    sys.exit(0)
try:
    jsonschema.validate(manifest, schema)
except jsonschema.ValidationError as exc:
    print(f"schema: {exc.message}")
    sys.exit(1)
print("ok corpus_manifest_validates")
PY

python3 - <<'PY' || bad "scenario_counts_match_freeze"
import json, sys
from pathlib import Path
m = json.loads(Path("manifest.json").read_text())
s = m["scenarios"]
checks = [
    ("code_qa", "count", 50),
    ("agentic_loops", "count", 20),
    ("prefill_heavy", "count", 25),
]
ok = True
for key, field, expected in checks:
    path = Path(s[key]["file"])
    data = json.loads(path.read_text())
    if key == "code_qa":
        actual = len(data.get("questions", []))
    elif key == "agentic_loops":
        actual = len(data.get("tasks", []))
    else:
        actual = len(data.get("prompts", []))
    declared = s[key][field]
    if actual != expected or declared != expected:
        print(f"{key}: file={actual} manifest={declared} expected={expected}")
        ok = False
docs = set(s["long_context"]["document_ids"])
if docs != {"document-8k", "document-32k"}:
    print(f"long_context documents: {sorted(docs)}")
    ok = False
if not ok:
    sys.exit(1)
print("ok scenario_counts_match_freeze")
PY

python3 - <<'PY' || bad "corpus_models_pinned"
import json, sys
from pathlib import Path
m = json.loads(Path("manifest.json").read_text())
models = json.loads(Path("models.json").read_text())
ok = True
if m["models"]["sha256"] != __import__("hashlib").sha256(Path("models.json").read_bytes()).hexdigest():
    print("models.json sha256 does not match manifest")
    ok = False
ids = [p["id"] for p in models["profiles"]]
if ids != m["models"]["profile_ids"]:
    print(f"profile id order mismatch: {ids}")
    ok = False
for p in models["profiles"]:
    art = p["artifact"]
    if not art.get("revision") or not art.get("sha256") or not art.get("files"):
        print(f"{p['id']}: artifact revision, sha256, or files missing")
        ok = False
if not ok:
    sys.exit(1)
print("ok corpus_models_pinned")
PY

python3 - <<'PY' || bad "fixtures_hashed"
import hashlib, json, sys
from pathlib import Path
m = json.loads(Path("manifest.json").read_text())
ok = True

def check(path, expected, label):
    global ok
    file = Path(path)
    if not file.is_file():
        print(f"missing {label}: {path}")
        ok = False
        return
    actual = hashlib.sha256(file.read_bytes()).hexdigest()
    if actual != expected:
        print(f"hash mismatch {label}")
        print(f"  manifest {expected}")
        print(f"  file     {actual}")
        ok = False

repo = m["fixtures"]["repository"]
check(repo["archive"], repo["content_sha256"], "repo snapshot")
for name, doc in m["fixtures"]["documents"].items():
    check(doc["file"], doc["sha256"], name)
    if doc.get("actual_tokens") != doc.get("target_tokens"):
        print(f"{name}: actual_tokens {doc.get('actual_tokens')} != target {doc.get('target_tokens')}")
        ok = False
for key, scenario in m["scenarios"].items():
    check(scenario["file"], scenario["sha256"], key)
if not ok:
    sys.exit(1)
print("ok fixtures_hashed")
PY

if [[ "$fail" -eq 0 ]]; then
  echo "ok corpus checks"
else
  echo "corpus checks failed" >&2
  exit 1
fi