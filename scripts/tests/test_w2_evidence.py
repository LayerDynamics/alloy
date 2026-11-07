"""Collector regression tests; run with python3 -B -m unittest discover."""
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location(
    "_w2_evidence_under_test", Path(__file__).resolve().parents[1] / "w2_evidence.py"
)
w2 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = w2
SPEC.loader.exec_module(w2)


def output(names, status="ok", ignored=0):
    lines = [f"test {name} ... {status}" for name in names]
    passed = len(names) if status == "ok" else 0
    failed = len(names) if status == "FAILED" else 0
    verdict = "FAILED" if failed else "ok"
    lines.append(f"test result: {verdict}. {passed} passed; {failed} failed; {ignored} ignored; 0 measured; 0 filtered out; finished in 0.00s")
    return "\n".join(lines)


class EvidenceParsingTests(unittest.TestCase):
    def test_listing_has_real_test_names_only(self):
        self.assertEqual(w2.parse_listing("Compiling x\na::b: test\nc: benchmark\n1 test, 1 benchmark"), ["a::b"])

    def test_empty_listing_fails(self):
        gate = w2.TestGate("x", "engine", "lib", "x")
        self.assertIn("zero tests discovered", w2.discovery_errors(gate, []))

    def test_missing_required_test_fails(self):
        gate = w2.TestGate("x", "engine", "lib", "x", ("needed",))
        self.assertTrue(w2.discovery_errors(gate, ["mod::different"]))

    def test_required_test_matches_exact_name_or_qualified_suffix(self):
        gate = w2.TestGate("x", "engine", "lib", "x", ("needed",))
        self.assertEqual(w2.discovery_errors(gate, ["mod::needed"]), [])
        self.assertTrue(w2.discovery_errors(gate, ["mod::not_needed"]))

    def test_duplicate_discovery_fails(self):
        gate = w2.TestGate("x", "engine", "lib", "x")
        self.assertTrue(w2.discovery_errors(gate, ["x", "x"]))

    def test_ambiguous_required_suffix_fails(self):
        gate = w2.TestGate("x", "engine", "lib", "x", ("needed",))
        self.assertTrue(w2.discovery_errors(gate, ["a::needed", "b::needed"]))

    def test_minimum_suite_size_is_enforced(self):
        gate = w2.TestGate("x", "engine", "lib", "x", minimum=2)
        self.assertTrue(w2.discovery_errors(gate, ["one"]))

    def test_named_passing_results_and_summary_match_discovery(self):
        evidence = w2.execution_evidence(output(["a", "b"]), ["a", "b"])
        self.assertEqual(evidence["validation_errors"], [])

    def test_successful_zero_test_run_is_rejected(self):
        evidence = w2.execution_evidence(output([]), [])
        self.assertIn("zero tests observed passing", evidence["validation_errors"])

    def test_missing_discovered_result_is_rejected(self):
        self.assertTrue(w2.execution_evidence(output(["a"]), ["a", "b"])["validation_errors"])

    def test_unexpected_executed_test_is_rejected(self):
        self.assertTrue(w2.execution_evidence(output(["a", "other"]), ["a"])["validation_errors"])

    def test_failed_tests_are_rejected(self):
        self.assertTrue(w2.execution_evidence(output(["a"], "FAILED"), ["a"])["validation_errors"])

    def test_ignored_tests_are_rejected(self):
        self.assertTrue(w2.execution_evidence(output(["a"], "ignored", 1), ["a"])["validation_errors"])

    def test_duplicate_execution_is_rejected(self):
        self.assertTrue(w2.execution_evidence(output(["a", "a"]), ["a"])["validation_errors"])

    def test_missing_summary_is_rejected(self):
        self.assertTrue(w2.execution_evidence("test a ... ok", ["a"])["validation_errors"])

    def test_multiple_summaries_are_rejected(self):
        self.assertTrue(w2.execution_evidence(output(["a"]) + "\n" + output([]), ["a"])["validation_errors"])

    def test_inconsistent_summary_count_is_rejected(self):
        log = output(["a"]).replace("1 passed", "2 passed")
        self.assertTrue(w2.execution_evidence(log, ["a"])["validation_errors"])

    def test_primary_rejection_case_targets_the_slab_delivery_path(self):
        gate = next(g for g in w2.gates() if g.name == "P2-04-slab")
        self.assertIn("w2_slab_delivery_tests", gate.selector)
        self.assertTrue(gate.exact)

    def test_selftest_zero_run_is_rejected(self):
        self.assertTrue(w2.python_selftest_errors("Ran 0 tests in 0.0s\n\nOK\n"))

    def test_selftest_positive_summary_is_accepted(self):
        self.assertEqual(w2.python_selftest_errors("Ran 32 tests in 0.1s\n\nOK\n"), [])

    def test_selftest_skips_are_not_accepted_as_full_coverage(self):
        self.assertTrue(w2.python_selftest_errors("Ran 32 tests in 0.1s\n\nOK (skipped=4)\n"))

    def test_binary_target_selects_cargo_bin_not_integration_test(self):
        gate = w2.TestGate("binary", "alloyd", "bin:alloyd", "rpc_transport_config::tests", ("example",))
        command = gate.command(True)
        self.assertIn("--bin", command)
        self.assertEqual(command[command.index("--bin") + 1], "alloyd")
        self.assertNotIn("--test", command)

    def test_all_lower_layers_do_not_become_trained_model_evidence(self):
        names = ["qwen-decoder-hooks", "slab-delivery", "runtime-slab-handoff", "rpc-borrowed-transport",
                 "rpc-notification-dispatch", "rpc-write-failure-ownership", "daemon-transport-options",
                 "daemon-transport-cli", "parser-shared-fixture-parity"]
        boundary = w2.acceptance_boundary([{"gate": name, "passed": True} for name in names])
        self.assertTrue(all(layer["passed"] for layer in boundary["layers"].values()))
        self.assertTrue(all(not layer["trained_model_tool_loop_evidence"] for layer in boundary["layers"].values()))
        self.assertEqual(boundary["candidate_built_trained_model_daemon_tool_loop"]["status"], "not_collected")

    def test_all_case_identifiers_are_unique(self):
        names = [g.name for g in w2.gates()]
        self.assertEqual(len(names), len(set(names)))
        self.assertTrue(all(g.minimum > 0 and g.required for g in w2.gates()))

    def test_acceptance_distinguishes_pending_review_from_missing_implementation(self):
        gaps = w2.acceptance_gaps()
        self.assertTrue(any(g["id"] == "mlx-schema-cache-restore" for g in gaps))
        self.assertFalse(any("NOT implemented" in g["detail"] for g in gaps))


class GateRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.calls = []

    def runner(self, listing, execution, discovery_rc=0, execution_rc=0):
        def run(phase, command, overrides):
            self.calls.append(phase)
            path = self.root / (phase + ".log")
            path.write_text(listing if phase == "list" else execution)
            rc = discovery_rc if phase == "list" else execution_rc
            return w2.CommandRun(command, rc, path, w2.hash_file(path), 0.01)
        return run

    def test_no_execution_after_zero_test_discovery(self):
        gate = w2.TestGate("x", "engine", "lib", "absent")
        result = w2.collect_test_gate(gate, self.runner("0 tests, 0 benchmarks", output([])))
        self.assertFalse(result["passed"])
        self.assertEqual(self.calls, ["list"])

    def test_nonzero_execution_exit_rejects_passing_looking_log(self):
        gate = w2.TestGate("x", "engine", "lib", "a", ("a",))
        result = w2.collect_test_gate(gate, self.runner("a: test", output(["a"]), execution_rc=1))
        self.assertFalse(result["passed"])

    def test_success_records_discovery_and_execution_log_digests(self):
        gate = w2.TestGate("x", "engine", "lib", "a", ("a",))
        result = w2.collect_test_gate(gate, self.runner("a: test", output(["a"])))
        self.assertTrue(result["passed"])
        self.assertEqual(self.calls, ["list", "run"])
        self.assertEqual(result["execution"]["log_sha256"], w2.hash_file(self.root / "run.log"))

    def test_failed_discovery_command_does_not_run_tests(self):
        gate = w2.TestGate("x", "engine", "lib", "a", ("a",))
        result = w2.collect_test_gate(gate, self.runner("a: test", output(["a"]), discovery_rc=1))
        self.assertFalse(result["passed"])
        self.assertEqual(self.calls, ["list"])

    def test_output_cannot_live_inside_checkout(self):
        with self.assertRaises(ValueError):
            w2.validate_output_directory(self.root, self.root / "evidence")

    def test_checkout_alias_cannot_hide_internal_output(self):
        actual = self.root / "real-checkout"
        actual.mkdir()
        alias = self.root / "checkout-alias"
        alias.symlink_to(actual.resolve(), target_is_directory=True)
        with self.assertRaises(ValueError):
            w2.validate_output_directory(alias, actual / "new-evidence")
        with self.assertRaises(ValueError):
            w2.validate_output_directory(actual, alias / "new-evidence")

    def test_output_alias_into_checkout_is_rejected(self):
        actual = self.root / "checkout"
        actual.mkdir()
        outside = self.root / "outside"
        outside.mkdir()
        alias = outside / "back-into-checkout"
        alias.symlink_to(actual.resolve(), target_is_directory=True)
        with self.assertRaises(ValueError):
            w2.validate_output_directory(actual, alias / "new-evidence")

    def test_existing_output_is_not_overwritten(self):
        with self.assertRaises(ValueError):
            w2.validate_output_directory(self.root / "checkout", self.root)

    def test_spawn_failure_is_recorded(self):
        result = w2.run_logged([str(self.root / "missing-command")], self.root, self.root / "spawn.log", 2)
        self.assertFalse(result.ok())
        self.assertIsNotNone(result.spawn_error)
        self.assertEqual(result.log_sha256, w2.hash_file(result.log))

    def test_timeout_is_recorded_as_failure(self):
        result = w2.run_logged([sys.executable, "-c", "import time; time.sleep(20)"],
                              self.root, self.root / "timeout.log", 0.1)
        self.assertFalse(result.ok())
        self.assertTrue(result.timed_out)


@unittest.skipUnless(shutil.which("git"), "Git is required for candidate-integrity tests")
class CandidateIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        def command(*args):
            return subprocess.check_output(["git", "-C", str(self.root), *args],
                                           text=True, stderr=subprocess.STDOUT).strip()
        self.git = command
        self.git("init")
        self.git("config", "user.email", "w2-evidence-test@example.invalid")
        self.git("config", "user.name", "W2 evidence test")
        (self.root / "source.txt").write_text("original\n")
        self.git("add", "source.txt")
        self.git("-c", "commit.gpgsign=false", "commit", "-m", "fixture")
        self.sha = self.git("rev-parse", "HEAD")

    def test_clean_candidate_is_accepted(self):
        w2.ensure_candidate(self.root, self.sha)

    def test_tracked_source_drift_is_rejected(self):
        (self.root / "source.txt").write_text("changed\n")
        with self.assertRaises(w2.CandidateChanged):
            w2.ensure_candidate(self.root, self.sha)

    def test_untracked_source_is_rejected(self):
        (self.root / "new.txt").write_text("untracked\n")
        with self.assertRaises(w2.CandidateChanged):
            w2.ensure_candidate(self.root, self.sha)

    def test_changed_head_is_rejected_even_when_tree_is_clean(self):
        (self.root / "source.txt").write_text("new commit\n")
        self.git("add", "source.txt")
        self.git("-c", "commit.gpgsign=false", "commit", "-m", "second")
        with self.assertRaises(w2.CandidateChanged):
            w2.ensure_candidate(self.root, self.sha)


if __name__ == "__main__":
    unittest.main()
