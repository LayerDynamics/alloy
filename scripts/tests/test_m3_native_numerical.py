from hashlib import sha256
import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

spec = importlib.util.spec_from_file_location("m3_native_numerical", Path(__file__).resolve().parents[1] / "ci/m3_native_numerical.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class NativeEvidenceTests(unittest.TestCase):
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
