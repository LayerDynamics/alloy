import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ci"))
import m3_artifact_archive as archive_tool


class M3ArtifactArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "matrix"
        (self.root / "sample-numerical" / "report").mkdir(parents=True)
        (self.root / "sample-numerical" / "raw").mkdir()
        (self.root / "package" / "qualification").mkdir(parents=True)
        (self.root / "index.json").write_text(json.dumps({"status": "failed"}))
        (self.root / "sample-numerical" / "receipt.json").write_text('{"status":"passed"}')
        self.payload = self.root / "sample-numerical" / "report" / "numerical.json"
        self.payload.write_bytes(b'{"logits":[' + b"1.25," * 1000 + b"0]}")
        self.raw = self.root / "sample-numerical" / "raw" / "case-0-step-0.json"
        self.raw.write_bytes(b'{"logits":[' + b"2.50," * 1000 + b"0]}")
        self.package = self.root / "package" / "qualification" / "spec.json"
        self.package.write_bytes(b"packaged bytes")

    def test_archive_and_restore_exact_bytes(self):
        before = {p: p.read_bytes() for p in (self.payload, self.raw, self.package)}
        with patch.object(archive_tool, "MIN_BYTES", 1):
            self.assertEqual(archive_tool.archive(self.root), 2)
        self.assertFalse(self.payload.exists())
        self.assertFalse(self.raw.exists())
        self.assertEqual(self.package.read_bytes(), before[self.package])
        self.assertTrue((self.root / "sample-numerical" / "receipt.json").exists())
        self.assertEqual(archive_tool.restore(self.root), 2)
        for path, expected in before.items():
            self.assertEqual(path.read_bytes(), expected)

    def test_tampered_archive_does_not_restore(self):
        with patch.object(archive_tool, "MIN_BYTES", 1):
            archive_tool.archive(self.root)
        source = self.root / archive_tool.ARCHIVE
        with source.open("ab") as stream:
            stream.write(b"tampered")
        with self.assertRaisesRegex(ValueError, "archive digest differs"):
            archive_tool.restore(self.root)
        self.assertFalse(self.payload.exists())


if __name__ == "__main__":
    unittest.main()
