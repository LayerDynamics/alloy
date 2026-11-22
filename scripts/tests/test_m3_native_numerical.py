from hashlib import sha256
import os
import json
from unittest.mock import patch
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

spec = importlib.util.spec_from_file_location("m3_native_numerical", Path(__file__).resolve().parents[1] / "ci/m3_native_numerical.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class NativeEvidenceTests(unittest.TestCase):
    def test_family_owned_numerical_pins(self):
        for profile, package in [("llama-3.1-8b-4bit", "llama3"), ("llama-3.1-8b-8bit", "llama3"),
                                 ("smollm3-3b-4bit", "smollm")]:
            suite = runner.SUITES[profile]
            pin = json.loads((runner.ROOT / suite.manifest).read_bytes())
            self.assertEqual((suite.package, suite.target, pin["id"]), (package, "quantized_loading", profile))
            self.assertEqual(len(pin["prompt_tokens"]), 2)
            self.assertEqual(pin["decode_tokens"], [19, 13])

    def test_candidates_and_checkouts_have_separate_cargo_targets(self):
        with TemporaryDirectory() as tmp, patch.dict(os.environ, {"CARGO_TARGET_DIR": tmp}):
            root = Path(tmp)
            a, b = "a" * 40, "b" * 40
            t = runner.cargo_target(a, root, root / "first")
            self.assertNotEqual(t, runner.cargo_target(b, root, root / "first"))
            self.assertNotEqual(t, runner.cargo_target(a, root, root / "second"))
            self.assertEqual(t, runner.cargo_target(a, root, root / "first"))
            self.assertEqual(os.environ["CARGO_TARGET_DIR"], tmp)
            with self.assertRaises(ValueError):
                runner.cargo_target("../old", root)

    def test_only_exact_named_execution_passes(self):
        output = "running 1 test\ntest native ... ok\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 4 filtered out\n"
        runner.require_single_pass(output, "native")
        for old, new in [("native ... ok", "similar ... ok"), ("1 test", "0 tests"), ("1 passed", "0 passed"), ("0 ignored", "1 ignored")]:
            with self.subTest(new=new), self.assertRaises(ValueError):
                runner.require_single_pass(output.replace(old, new), "native")

    def test_rejects_mutated_artifact_and_reference(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            weights = folder / "weights"
            reference = folder / "reference"
            weights.write_bytes(b"weights")
            reference.write_bytes(b"reference")
            manifest = {"artifact": {"files": [{"path": "weights", "bytes": 7, "sha256": sha256(b"weights").hexdigest()}]},
                        "reference": {"sha256": sha256(b"reference").hexdigest()}}
            runner.verify_inputs(manifest, folder, reference)
            weights.write_bytes(b"Weights")
            with self.assertRaisesRegex(ValueError, "artifact"):
                runner.verify_inputs(manifest, folder, reference)
            weights.write_bytes(b"weights")
            reference.write_bytes(b"Reference")
            with self.assertRaisesRegex(ValueError, "reference"):
                runner.verify_inputs(manifest, folder, reference)
