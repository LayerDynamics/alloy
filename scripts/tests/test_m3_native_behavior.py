import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ci"))
import m3_native_behavior as behavior


class NativeBehaviorTests(unittest.TestCase):
    def test_shared_lifecycle_is_mandatory_for_each_pinned_profile(self):
        active = {"gemma-4-e4b-it-4bit", "ornith-1.5-9b-4bit", "qwen3.5-0.8b-4bit"}
        for profile in behavior.SUITES.keys() - active:
            with self.subTest(profile=profile), self.assertRaisesRegex(ValueError, "missing native behavioral inventory"):
                behavior.inventory(profile)
        for profile in active:
            names = [n for _, _, group in behavior.inventory(profile) for n in group]
            self.assertEqual(len(names), len(set(names)))
            kinds = {k for name in names for k in behavior.COVERAGE[name]}
            self.assertEqual(kinds, {"tool_schema", "stops", "exact_counting", "long_context", "compaction",
                                     "cached_uncached", "streaming", "cancellation", "restart",
                                     "memory8_gi_b", "memory36_gi_b", "policy_gated_task"})
            self.assertIn("real_model_compaction_preserves_history_and_continues", names)
            self.assertIn("llama3_tool_stop_context_cache", names)
            self.assertIn("native_validated_context_boundary", names)
            self.assertEqual("qwen35_native_memory_tiers" in names, profile in ("ornith-1.5-9b-4bit", "qwen3.5-0.8b-4bit"))
            self.assertEqual("gemma4_native_memory_tiers" in names, profile == "gemma-4-e4b-it-4bit")
        with self.assertRaises(ValueError):
            behavior.inventory("unknown")

    def test_inherited_fixture_overrides_are_discarded(self):
        with patch.dict(os.environ, {"ALLOY_TEST_REAL_SSD_PHASE": "read", "ALLOY_TEST_COLD": "1",
                                     "ALLOY_REAL_PROFILE": "wrong"}):
            selected = {"ALLOY_REAL_PROFILE": "expected"}
            env = behavior.fixture_env(selected)
            self.assertEqual({k: v for k, v in env.items() if k.startswith("ALLOY_")}, selected)

    def test_failed_and_filtered_fixtures_do_not_mask_later_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            binary = out / "test"
            binary.write_text(
                "#!/bin/sh\ncase \"$1\" in\n"
                "fail) echo assertion >&2; exit 101;;\n"
                "filtered) echo 'running 0 tests';;\n"
                "pass) echo 'running 1 test'; echo 'test pass ... ok'; "
                "echo 'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 2 filtered out';;\nesac\n"
            )
            binary.chmod(0o700)
            record = dict.fromkeys(["candidate", "profile", "manifest_sha256", "artifact_revision",
                                    "reference_sha256", "cargo_lock_sha256", "pins_sha256", "host", "architecture"], "fixture")
            record.update(commands=[{"exit_code": 0}], binaries={"test": {"sha256": behavior.sha256(binary)}})
            with patch.object(behavior, "build", return_value=binary):
                rows = behavior.run_suite(out, record, {}, [("p", "test", ("fail", "filtered", "pass"))])
            self.assertEqual([row["passed"] for row in rows], [False, False, True])
            for row, status in zip(rows, (101, 0, 0)):
                path = out / (row["fixture_id"] + ".json")
                self.assertEqual(row["evidence_sha256"], behavior.sha256(path))
                evidence = json.loads(path.read_text())
                command = evidence["execution"]
                self.assertEqual(command["exit_code"], status)
                self.assertEqual(command["argv"][1], row["fixture_id"])
                for stream in ("stdout", "stderr"):
                    self.assertEqual(command[stream + "_sha256"], behavior.sha256(out / command[stream]))
