import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ci"))
import m3_native_package as package


class NativePackageTests(unittest.TestCase):
    def test_incomplete_or_contradictory_runs_cannot_grant_profiles(self):
        q = {"profile": "profile", "status": "passed"}
        n = dict(q, scope="numerical")
        b = dict(q, scope="behavior")
        for rows in ([n], [n, n], [n, dict(b, status="failed")]):
            with self.assertRaisesRegex(ValueError, "contradicts"):
                list(package.selected({"qualifications": [q], "runs": rows}))
        self.assertEqual(list(package.selected({"qualifications": [dict(q, status="failed")], "runs": [n, b]})), [])
        self.assertEqual(list(package.selected({"qualifications": [q], "runs": [n, b]})), [("profile", q, [n, b])])

    def test_package_rejects_stale_receipt_and_changed_copies(self):
        name = "ornith-1.5-9b-4bit"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            receipt = root / (name + "-qualification") / "receipt.json"
            receipt.parent.mkdir()
            receipt.write_text(json.dumps(dict(candidate="old", scope="qualification", profile=name, status="passed")))
            q = {"receipt_sha256": package.native.sha256(receipt)}
            with self.assertRaisesRegex(ValueError, "stale"):
                package.package_profile(root, root, "new", name, q, [])
            receipt.write_text("{}")
            with self.assertRaisesRegex(ValueError, "changed"):
                package.checked_copy(receipt, root / "copy", q["receipt_sha256"])

    def test_mismatched_colocated_kernels_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            lib = root / "lib/mlx.metallib"
            lib.parent.mkdir()
            lib.write_bytes(b"compiler output")
            artifacts = [dict(reason="build-script-executed", package_id="path+file:///crates/mlx-ffi#1", out_dir=str(root))]
            actual = package.native.metal_library(artifacts, root / "alloyd")
            self.assertEqual(actual["sha256"], package.native.sha256(lib))
            (root / "mlx.metallib").write_bytes(b"unexpected kernels")
            with self.assertRaisesRegex(ValueError, "conflicting"):
                package.native.metal_library(artifacts, root / "alloyd")

    def test_package_publishes_only_after_compiler_and_digest_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            lib = root / "linked.metallib"
            lib.write_bytes(b"compiled kernel library")
            binary = root / "daemon"
            binary.write_bytes(b"compiled daemon")
            kernel = {"path": str(lib), "sha256": package.native.sha256(lib)}
            entry = {"context": 32768}
            index = {"candidate": "a" * 40, "qualifications": [{"profile":"x", "status":"passed"}],
                     "runs":[{"profile":"x", "scope":s, "status":"passed"} for s in ("numerical", "behavior")]}
            out = root / "cargo.json"
            out.write_text(json.dumps({"reason":"compiler-artifact", "executable":str(binary),
                                       "manifest_path": str(package.native.ROOT / "crates/alloyd/Cargo.toml"),
                                       "target":{"name":"alloyd","kind":["bin"]}}))
            with patch.object(package.native, "verify_candidate"), patch.object(package.native, "cargo_target", return_value=root), \
                    patch.object(package, "host", return_value={}), \
                    patch.object(package, "package_profile", return_value=(entry, kernel)), \
                    patch.object(package.native, "execute", return_value=out), \
                    patch.object(package.native, "metal_library", side_effect=[{},kernel]):
                with self.assertRaisesRegex(ValueError, "differs"):
                    package.publish(index,root)
                self.assertFalse((root / "package").exists())
                result=package.publish(index,root)
            self.assertEqual(result["catalog_sha256"], package.native.sha256(root / "package/qualification/catalog.json"))
            self.assertEqual(result["profiles"], ["x"])
            self.assertEqual((root / "package/alloyd").read_bytes(),binary.read_bytes())
            self.assertEqual((root / "package/mlx.metallib").read_bytes(),lib.read_bytes())
