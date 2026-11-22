import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ci"))
import m3_native_matrix as matrix


class NativeMatrixTests(unittest.TestCase):
    def test_missing_profile_and_wrong_receipt_keep_independent_results(self):
        profiles = list(matrix.native.SUITES)
        self.assertGreaterEqual(len(profiles), 3)
        candidate = "a" * 40
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = root / "inputs"
            inputs.write_text(json.dumps({p: {"artifact": str(root), "reference": str(root / "ref")}
                                          for p in profiles[1:]}))

            def stub(scope):
                def run(args):
                    args.output.mkdir()
                    if args.profile == profiles[1] and scope == "behavior":
                        return
                    revision = "old" if args.profile == profiles[1] else candidate
                    receipt = {"version": 1, "candidate": revision, "scope": scope,
                               "profile": args.profile, "status": "passed"}
                    (args.output / "receipt.json").write_text(json.dumps(receipt))
                return run

            runners = {scope: stub(scope) for scope in matrix.RUNNERS}
            with patch.dict(matrix.RUNNERS, runners), patch.object(matrix.native, "verify_candidate"):
                index = matrix.run(candidate, inputs, root / "out")
                with self.assertRaises(FileExistsError):
                    matrix.run(candidate, inputs, root / "out")
            self.assertEqual(index, json.loads((root / "out/index.json").read_text()))
            self.assertEqual(index["status"], "failed")
            self.assertEqual(len(index["runs"]), 2 * len(profiles))
            for row in index["runs"]:
                if row["profile"] == profiles[0]:
                    self.assertIn("profile input", row["error"])
                elif row["profile"] == profiles[1]:
                    self.assertIn("receipt_error", row)
                else:
                    self.assertEqual(row["status"], "passed")
                    self.assertEqual(matrix.native.sha256(root / "out" / row["receipt"]), row["receipt_sha256"])

    def test_invalid_inventory_retains_terminal_failure_index(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(matrix.native, "verify_candidate"):
            root = Path(tmp)
            config = root / "config"
            config.write_text('{"unknown": {}}')
            index = matrix.run("candidate", config, root / "out")
            self.assertEqual(index["runs"], [])
            self.assertEqual(index["status"], "failed")
            self.assertIn("unexpected profiles", index["error"])
