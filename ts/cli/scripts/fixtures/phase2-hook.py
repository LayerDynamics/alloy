#!/usr/bin/env python3
"""Controlled plugin that records real hook invocations without modifying them."""
import json
import sys

with open(sys.argv[1], "a", encoding="utf-8", buffering=1) as log:
    for line in sys.stdin:
        request = json.loads(line)
        method = request["method"]
        log.write(json.dumps({"method": method, "params": request.get("params")}) + "\n")
        if method == "initialize":
            result = {"protocol_version": 1, "plugin_id": "w7", "plugin_version": "1.0.0", "hooks": ["tool.before_execute", "tool.after_execute"]}
        elif method == "tool.before_execute":
            result = {"decision": "allow", "annotations": {"w7.observed": True}}
        elif method == "tool.after_execute":
            result = {"annotations": {"w7.observed": True}}
        else:
            raise ValueError("unexpected method")
        print(json.dumps({"jsonrpc": "2.0", "id": request["id"], "result": result}), flush=True)
