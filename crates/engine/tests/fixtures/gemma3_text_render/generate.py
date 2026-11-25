"""Group logical history blocks, then use the pinned publisher's text template."""
import hashlib
import json
import os
from pathlib import Path

from jinja2 import Environment

root = Path(__file__).parent
artifact = Path(os.environ["ALLOY_GEMMA3_SMALL_ARTIFACT"])
metadata = (artifact / "tokenizer_config.json").read_bytes()
assert hashlib.sha256(metadata).hexdigest() == "bfe25c2735e395407beb78456ea9a6984a1f00d8c16fa04a8b75f2a614cf53e1"
messages = json.loads((root / "input.json").read_text())
prefix = []
if messages[0]["role"] == "system":
    prefix.append(messages.pop(0))
grouped = []
for message in messages:
    assert message["role"] in ("user", "assistant")
    content = message["content"].strip()
    if grouped and grouped[-1]["role"] == message["role"]:
        grouped[-1]["content"] += "\n" + content
    else:
        grouped.append(dict(role=message["role"], content=content))

def fail(message):
    raise ValueError(message)

environment = Environment()
environment.globals["raise_exception"] = fail
template = environment.from_string(json.loads(metadata)["chat_template"])
rendered = template.render(messages=prefix + grouped, bos_token="<bos>", add_generation_prompt=True)
(root / "expected.txt").write_text(rendered)
print("expected_sha256", hashlib.sha256(rendered.encode()).hexdigest())
