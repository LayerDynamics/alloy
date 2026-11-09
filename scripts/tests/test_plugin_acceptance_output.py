"""Exercise the real collector's output lifecycle through genuine preflight failures."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
COLLECTOR = ROOT / "scripts/plugin_acceptance.sh"


class PluginAcceptanceOutputTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="alloy-collector-output-")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.source = self.base / "source"
        self.source.mkdir()
        self.output = self.base / "evidence"
        self.output.mkdir()

    def run_collector(self, candidate="invalid", output=None):
        env = dict(os.environ, ALLOY_W5_OUTPUT=str(output or self.output),
                   ALLOY_W5_CANDIDATE=candidate)
        return subprocess.run(["/bin/bash", str(COLLECTOR)], cwd=self.source,
                              env=env, text=True, capture_output=True, timeout=15)

    def seed_previous_run(self):
        names = ["report.json", "report.json.tmp", "candidate.txt", "platform.txt",
                 "source-before.sha256", "source-after.sha256", "protocol-discovery.txt",
                 "unit-discovery.txt", "sandbox-discovery.txt", "sandbox-platform.txt",
                 "hardening::plugin_blocked_stdin_deadline_reaps_owned_process.log",
                 "hardening::plugin_blocked_stdin_deadline_reaps_owned_process.command.txt",
                 "hardening_plugin_blocked_stdin_deadline_reaps_owned_process.log",
                 "plugin_blocked_stdin_deadline_reaps_owned_process.log"]
        for name in names:
            (self.output / name).write_text("OLD RUN\n")
        return names

    def test_failed_preflight_removes_previous_evidence(self):
        names = self.seed_previous_run()
        (self.output / "notes.txt").write_text("caller-owned\n")
        result = self.run_collector()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Invalid candidate SHA", result.stderr)
        self.assertEqual([name for name in names if (self.output / name).exists()], [])
        self.assertEqual((self.output / "notes.txt").read_text(), "caller-owned\n")
        self.assertFalse((self.output / ".plugin-acceptance.lock").exists())

    def test_cleanup_unlinks_evidence_symlinks_without_touching_targets(self):
        preserved = self.base / "preserved.txt"
        preserved.write_text("preserve\n")
        (self.output / "report.json").symlink_to(preserved)
        owned_log = self.output / "format.log"
        owned_log.symlink_to(preserved)
        result = self.run_collector()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.output / "report.json").is_symlink())
        self.assertFalse(owned_log.is_symlink())
        self.assertEqual(preserved.read_text(), "preserve\n")

    def test_second_collector_cannot_erase_an_active_runs_evidence(self):
        lock = self.output / ".plugin-acceptance.lock"
        lock.mkdir()
        report = self.output / "report.json"
        report.write_text("active owner\n")
        result = self.run_collector()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("already in use", result.stderr)
        self.assertEqual(report.read_text(), "active owner\n")
        self.assertTrue(lock.is_dir())

    def test_current_wrapper_log_survives_the_collectors_failure(self):
        wrapper = self.output / "driver.log"
        env = dict(os.environ, ALLOY_W5_OUTPUT=str(self.output),
                   ALLOY_W5_CANDIDATE="invalid")
        with wrapper.open("w") as log:
            result = subprocess.run(["/bin/bash", str(COLLECTOR)], cwd=self.source,
                                    env=env, stdout=log, stderr=subprocess.STDOUT, timeout=15)
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(wrapper.is_file())
        self.assertIn("Invalid candidate SHA", wrapper.read_text())

    def test_manifest_removes_retired_collector_files(self):
        names = ["retired-gate.log", "retired-gate.command.txt"]
        for name in names:
            (self.output / name).write_text("old collector output\n")
        (self.output / "collector-files.txt").write_text("\n".join(names) + "\n")
        result = self.run_collector()
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue(all(not (self.output / name).exists() for name in names))
        self.assertEqual((self.output / "collector-files.txt").read_text(), "")

    def test_invalid_manifest_cannot_remove_files_outside_output(self):
        preserved = self.base / "outside.log"
        preserved.write_text("preserve\n")
        (self.output / "report.json").write_text("previous success\n")
        (self.output / "collector-files.txt").write_text("../outside.log\n")
        result = self.run_collector()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Invalid collector output manifest", result.stderr)
        self.assertEqual(preserved.read_text(), "preserve\n")
        self.assertFalse((self.output / "report.json").exists())

    def test_source_directory_is_rejected_before_cleanup(self):
        evidence = self.source / "report.json"
        evidence.write_text("source file\n")
        result = self.run_collector(output=self.source)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("outside the source tree or under target", result.stderr)
        self.assertEqual(evidence.read_text(), "source file\n")
        self.assertFalse((self.source / ".plugin-acceptance.lock").exists())

    def test_source_ancestor_is_rejected_before_cleanup(self):
        preserved = self.base / "caller.log"
        preserved.write_text("preserve\n")
        result = self.run_collector(output=self.base)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("outside the source tree or under target", result.stderr)
        self.assertEqual(preserved.read_text(), "preserve\n")
        self.assertFalse((self.base / ".plugin-acceptance.lock").exists())

    def test_filesystem_root_is_rejected_before_cleanup(self):
        result = self.run_collector(output=Path("/"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("outside the source tree or under target", result.stderr)


if __name__ == "__main__":
    unittest.main()
