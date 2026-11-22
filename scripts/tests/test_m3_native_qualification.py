import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ci"))
import m3_native_behavior as behavior
import m3_native_qualification as qualification


class QualificationTests(unittest.TestCase):
    def test_verify_native_log_and_propagate_failed_assertions(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            exe = out / "native"
            exe.write_bytes(b"binary")
            binary = {"path": str(exe), "sha256": qualification.sha256(exe)}
            name = "native_validated_context_boundary"
            keys = ["profile", "candidate", "manifest_sha256", "artifact_revision", "reference_sha256",
                    "cargo_lock_sha256", "pins_sha256", "host", "architecture"]
            identity = dict.fromkeys(keys, "example")
            (out / "stdout").write_text("running 1 test\ntest native_validated_context_boundary ... ok\n"
                                        "test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out\n")
            (out / "stderr").write_text("")
            logs = dict(stdout="stdout", stderr="stderr", stdout_sha256=qualification.sha256(out / "stdout"),
                        stderr_sha256=qualification.sha256(out / "stderr"))
            build = dict(logs, exit_code=0, argv=["cargo", "test", "--no-run", "--locked", "--release", "-p",
                                                   "pkg", "--test", "target", "--message-format=json"])
            command = dict(logs, argv=[str(exe), name, "--exact", "--ignored", "--test-threads=1", "--show-output"])
            recorder = dict(identity, fixture_environment={}, binaries={"target": binary}, commands=[build, command])
            evidence = dict(identity, environment={}, fixture_id=name, executable=binary, build=build, execution=command)
            path = out / (name + ".json")
            fixtures = out / "fixtures.json"

            def save(passed):
                evidence.update(passed=passed, detail="native assertion")
                path.write_text(json.dumps(evidence))
                row = {k: evidence[k] for k in ("fixture_id", "passed", "detail")}
                fixtures.write_text(json.dumps([dict(row, evidence_sha256=qualification.sha256(path))]))
                recorder["fixtures_sha256"] = qualification.sha256(fixtures)

            with patch.object(qualification, "inventory", return_value=[("pkg", "target", [name])]):
                for rc in [0, 101]:
                    command["exit_code"] = rc
                    save(rc == 0)
                    required, observed = qualification.fixture_rows(out, recorder, "profile")
                    self.assertEqual(set(required.values()), set(behavior.COVERAGE[name]))
                    self.assertTrue(all(r["passed"] == (rc == 0) for r in observed))
                save(True)
                with self.assertRaisesRegex(ValueError, "assertion result"):
                    qualification.fixture_rows(out, recorder, "profile")
                save(False)
                (out / "stdout").write_text("edited")
                with self.assertRaisesRegex(ValueError, "changed execution log"):
                    qualification.fixture_rows(out, recorder, "profile")
                fixtures.write_text("[]")
                recorder["fixtures_sha256"] = qualification.sha256(fixtures)
                with self.assertRaisesRegex(ValueError, "inventory"):
                    qualification.fixture_rows(out, recorder, "profile")

    def test_source_mismatch_rejects_otherwise_terminal_receipt(self):
        record = dict.fromkeys(["profile", "candidate", "manifest_sha256", "artifact_directory", "artifact_revision",
                                "reference", "reference_sha256", "cargo_lock_sha256", "pins_sha256", "host", "architecture"], "match")
        other = dict(record, version=1, scope="behavior", status="failed")
        qualification.check_inputs(record, other, "behavior")
        other["candidate"] = "old"
        with self.assertRaisesRegex(ValueError, "candidate"):
            qualification.check_inputs(record, other, "behavior")
