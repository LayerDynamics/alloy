"""Offline publisher render capture. Never used by production inference."""
import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path
from transformers.utils.chat_template_utils import _compile_jinja_template

assert version("transformers") == "5.23.0"
assert version("jinja2") == "3.1.6"
parser = argparse.ArgumentParser()
parser.add_argument("--ornith", type=Path, required=True)
parser.add_argument("--upstream", type=Path, required=True)
args = parser.parse_args()
root = Path(__file__).resolve().parent
data = json.loads((root / "input.json").read_text())
messages = list(data["base_messages"])
for item in data["history"]:
    if item["kind"] == "call":
        messages.append({"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": item["tool_id"], "arguments": item["arguments"]}}
        ]})
    elif item["kind"] == "result":
        content = {key: item[key] for key in ("call_id", "content", "is_error")}
        messages.append({"role": "tool", "content": json.dumps(
            content, ensure_ascii=False, sort_keys=True, separators=(",", ":"))})
    else:
        messages.append({"role": item["role"], "content": item["content"]})
for name, artifact, profile_name in [
    ("ornith", args.ornith, "ornith_1_5_9b_profile.json"),
    ("upstream", args.upstream, "qwen35_0_8b_profile.json"),
]:
    profile = json.loads((root.parents[3] / "inference-infra/tests/fixtures" / profile_name).read_text())
    pin = next(f for f in profile["artifact"]["files"] if f["path"] == "chat_template.jinja")
    template = (artifact / "chat_template.jinja").read_bytes()
    assert len(template) == pin["bytes"] and hashlib.sha256(template).hexdigest() == pin["sha256"]
    result = _compile_jinja_template(template.decode()).render(
        messages=messages, tools=data["tools"], enable_thinking=False, add_generation_prompt=True)
    output = result.encode()
    (root / (name + ".txt")).write_bytes(output)
    print(name, len(output), hashlib.sha256(output).hexdigest())
