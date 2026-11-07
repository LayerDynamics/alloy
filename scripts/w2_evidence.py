#!/usr/bin/env python3
"""Record W2 test evidence against a clean candidate. Never self-certifies W2 complete."""
import argparse, datetime, hashlib, json, re, subprocess
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    root=Path(subprocess.check_output(["git","rev-parse","--show-toplevel"],text=True).strip())
    def git(*args): return subprocess.check_output(["git","-C",str(root),*args],text=True).strip()
    if git("status","--porcelain"): raise SystemExit("Commit changes before evidence collection.")
    sha=git("rev-parse","HEAD")
    output=a.out.expanduser().resolve()
    if output == root or root in output.parents: raise SystemExit("Evidence must be outside the checkout.")
    output.mkdir(parents=True,exist_ok=False)
    cases=[
        ("P2-03",["cargo","test","--locked","-p","qwen3","--test","tool_calls","qwen3_tool_call_chunk_boundaries","--","--exact"],"qwen3_tool_call_chunk_boundaries"),
        ("P2-04",["cargo","test","--locked","-p","engine","--lib","generation::w2_completion_gate_tests::qwen3_malformed_tool_calls_never_execute","--","--exact"],"generation::w2_completion_gate_tests::qwen3_malformed_tool_calls_never_execute"),
        ("P2-05",["cargo","test","--locked","-p","engine","--test","tool_rendering","tool_context_exact_count_and_identity","--","--exact"],"tool_context_exact_count_and_identity"),
        ("completion-gates",["cargo","test","--locked","-p","engine","--lib","w2_completion_gate_tests"],None),
        ("tool-rendering",["cargo","test","--locked","-p","engine","--test","tool_rendering"],None),
        ("cache-schema-gates",["cargo","test","--locked","-p","engine","--lib","w2_tool_schema_cache_tests"],None),
        ("qwen-renderer",["cargo","test","--locked","-p","engine","--lib","tool_context_reaches_the_production_qwen_renderer"],"backends::qwen::shared_rendering_wiring_tests::tool_context_reaches_the_production_qwen_renderer"),
    ]
    results=[]
    for name,command,test in cases:
        run=subprocess.run(command,cwd=root,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        log=run.stdout
        (output/(name+".log")).write_text(log)
        tests=re.findall(r"^test (\S+) \.\.\. ok$",log,re.M)
        passed=run.returncode==0 and bool(tests) and (test is None or test in tests)
        results.append({"gate":name,"command":command,"passed":passed,"returncode":run.returncode,
            "observed_passed_tests":tests,"log_sha256":hashlib.sha256(log.encode()).hexdigest()})
        print(name, "PASS" if passed else "FAIL", flush=True)
    clean=not git("status","--porcelain") and git("rev-parse","HEAD")==sha
    report={
        "schema_version":1,"candidate_sha":sha,"candidate_unchanged_and_clean":clean,
        "recorded_at":datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "rustc":subprocess.check_output(["rustc","--version"],text=True).strip(),
        "results":results,"recorded_tests_passed":clean and all(r["passed"] for r in results),
        "w2_complete":False,
        "remaining_required_review":[
            "Exact proposal/parser storage admission and overlap accounting are NOT implemented by this patch.",
            "Review supported Qwen3/Coder syntax fixtures against the frozen family scope.",
            "Native retained-cache reuse/SSD restore acceptance for changed schemas remains unrecorded.",
            "P2-04 here is scripted-inference generation lifecycle evidence, not real-model or builtin execution evidence."
        ],
    }
    (output/"w2-evidence.json").write_text(json.dumps(report,indent=2)+"\n")
    print("Evidence:",output)
    if not report["recorded_tests_passed"]: raise SystemExit(1)
if __name__=="__main__":main()
