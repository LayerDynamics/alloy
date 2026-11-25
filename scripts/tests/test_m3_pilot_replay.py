from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ci"))
import m3_pilot_replay as replay

NAME = "llama3_tool_stop_context_cache"
MARKERS = {"tool_order": {"location": "tests/tools.rs:10:2", "lines": ["expected tool result"]},
           "tool_count": {"location": "tests/tools.rs:20:2", "lines": ["wrong count", "  left: 2"]}}


def failure(defect):
    marker = MARKERS[defect]
    return "\n".join(["running 1 test", f"test {NAME} ... FAILED",
                      f"thread '{NAME}' (21) panicked at {marker['location']}:",
                      *marker["lines"],
                      "test result: FAILED. 0 passed; 1 failed; 0 ignored; 0 measured; 2 filtered out; finished in 0.06s", "", ""])


class ArchiveReplayTests(unittest.TestCase):
    def test_one_rejection_can_still_miss_a_failing_fixture(self):
        from m3_native_pilot import FIXED
        a, b, c, d = [case._asdict() for case in FIXED]
        selection = {"base": "b", "merge_base": "b", "candidate": "h", "rule_version": 1,
                     "path_counts": {"mapped": 1, "total": 2}, "complete": False,
                     "fixed_comparator": [a, b, c, d], "selected": [a, c]}
        archived = {"candidate": "h", "observations": [
            {**a, "outcome": "product_assertion", "matches_expected_defect": False},
            {**b, "outcome": "product_assertion", "matches_expected_defect": True},
            {**c, "outcome": "inconclusive", "matches_expected_defect": False}]}
        report = replay.compare(selection, archived)
        mapped = report["strategies"]["change_based"]
        self.assertEqual(mapped["archive_advice"], "reject")
        self.assertEqual(mapped["known_failing_fixture_denominator"], 2)
        self.assertEqual(mapped["detected_failing_fixtures"], [a])
        self.assertEqual(mapped["missed_known_failing_fixtures"], [b])
        self.assertEqual(mapped["expected_defect_matches"], 0)
        self.assertEqual(mapped["unknown_native_cases"], [c])
        fixed = report["strategies"]["fixed"]
        self.assertEqual(fixed["expected_defect_matches"], 1)
        self.assertCountEqual(fixed["unknown_native_cases"], [c, d])
        archived["observations"] = [{**row, "outcome": "passed"} for row in (a, b, c, d)]
        self.assertEqual(replay.compare(selection, archived)["strategies"]["fixed"]["archive_advice"],
                         "inconclusive")
        archived["candidate"] = "other"
        with self.assertRaisesRegex(ValueError, "candidates differ"):
            replay.compare(selection, archived)

    def test_exact_product_signatures_and_unknown_outputs(self):
        for defect in MARKERS:
            self.assertEqual(replay.classify(NAME, 101, failure(defect), "", MARKERS),
                             ("product_assertion", defect))
        for code, log in [(1, failure("tool_order")), (101, failure("tool_order").replace("expected tool result", "oops")),
                          (101, failure("tool_order") + "truncated"), (0, "running 0 tests\n"),
                          (101, failure("tool_count").replace("  left: 2", "  left: 9"))]:
            self.assertEqual(replay.classify(NAME, code, log, "", MARKERS), ("inconclusive", None))

    def test_bound_evidence_and_different_defect(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = {"profile": "smollm3-3b-4bit", "package": "alloyd", "target": "llama_real_model",
                    "fixture": NAME, "expected_defect": "tool_order", "markers": MARKERS,
                    "manifest_sha256": "b" * 64, "reference_sha256": "c" * 64, "fixture_environment": {}}
            claim = {"version": 1, "candidate": "a" * 40, "cases": [case]}
            directory = root / (case["profile"] + "-behavior")
            directory.mkdir()

            def write(name, content):
                data = content.encode() if isinstance(content, str) else json.dumps(content).encode()
                (directory / name).write_bytes(data)
                return hashlib.sha256(data).hexdigest()

            def log(label, argv, text, code):
                return {"argv": argv, "stdout": label + ".stdout", "stderr": label + ".stderr",
                        "stdout_sha256": write(label + ".stdout", text), "stderr_sha256": write(label + ".stderr", ""),
                        "exit_code": code}

            binary = {"path": str(directory / "test"), "sha256": write("test", "test binary")}
            build = log("build", ["cargo", "test", "--no-run", "--locked", "--release", "-p", case["package"],
                                  "--test", case["target"], "--message-format=json"], "", 0)
            command = log(NAME, [binary["path"], NAME, "--exact", "--ignored", "--test-threads=1", "--show-output"],
                          failure("tool_count"), 101)
            identity = {k: case[k] for k in ("profile", "reference_sha256", "manifest_sha256")}
            identity.update(candidate=claim["candidate"], architecture="arm64", host="test", artifact_revision="revision",
                            pins_sha256="pins", cargo_lock_sha256="lock")
            event = dict(identity, fixture_id=NAME, build=build, execution=command,
                         environment={}, executable=binary, passed=False)
            rows = [{"fixture_id": NAME, "evidence_sha256": write(NAME + ".json", event), "passed": False}]
            receipt = dict(identity, status="failed", scope="behavior", version=1, fixture_environment={},
                           commands=[build, command], binaries={case["target"]: binary},
                           fixtures_sha256=write("fixtures.json", rows))
            write("receipt.json", receipt)

            def score(manifest):
                result = replay.replay(root, json.dumps(manifest).encode())
                self.assertFalse(result["approval_credit"])
                return result["observations"][0]

            self.assertEqual(score(claim)["observed_defect"], "tool_count")
            self.assertFalse(score(claim)["matches_expected_defect"])
            case["expected_defect"] = "tool_count"
            self.assertTrue(score(claim)["matches_expected_defect"])
            altered = deepcopy(claim)
            altered["candidate"] = "d" * 40
            self.assertEqual(score(altered)["outcome"], "inconclusive")
            write(NAME + ".stderr", "modified")
            self.assertEqual(score(claim)["outcome"], "inconclusive")
            (directory / "receipt.json").unlink()
            self.assertEqual(score(claim)["outcome"], "inconclusive")


if __name__ == "__main__":
    unittest.main()
