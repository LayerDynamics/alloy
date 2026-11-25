from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ci"))
from m3_pr_timing import cohort, evaluate


def at(minute):
    return f"2026-11-25T10:{minute:02}:00Z"


def job(name, minute, sha):
    return {"name": name, "conclusion": "success", "status": "completed",
            "checkout_sha": sha, "created_at": at(1), "started_at": at(2),
            "completed_at": at(minute), "published_at": at(minute)}


def example():
    key = {"repository": "example/alloy", "pr": 12, "head_sha": "a" * 40, "event_id": "delivery-A"}
    ci = ["mac-metal", "approval-storage-macos-arm64"]
    linux = "approval-storage-linux-arm64"
    common = {**key, "attempt": 1, "created_at": at(1), "updated_at": at(25), "status": "completed"}
    return {"request": {**key, "base_sha": "c" * 40, "event": "pull_request",
                        "same_repository": True, "requested_at": at(0),
                        "advisory_published_at": at(12)},
            "as_of": at(30), "policy": {"captured_at": at(0), "required": {
                "ci": ci.copy(), "approval-storage-evidence": [linux]}},
            "attempts": [
                {**common, "workflow": "ci", "run_id": 10,
                 "jobs": [job(n, 25, key["head_sha"]) for n in ci]},
                {**common, "workflow": "approval-storage-evidence", "run_id": 11,
                 "jobs": [job(linux, 20, "b" * 40),
                          {**job("approval-storage-macos-standalone", 2, None), "conclusion": "skipped"}]}]}


class DeadlineTests(unittest.TestCase):
    def test_deadline_uses_last_required_publication(self):
        s = example()
        self.assertEqual(evaluate(s)["deadline"], "on_time")
        s["attempts"][0]["jobs"][0]["published_at"] = at(26)
        result = evaluate(s)
        self.assertEqual((result["deadline"], result["elapsed_seconds"]), ("miss", 1560))
        self.assertEqual(result["verdict"], "approved")

    def test_rerun_only_ci_keeps_original_clock_and_other_workflow(self):
        s = example()
        ci = s["attempts"][0]
        ci["jobs"][0]["conclusion"] = "failure"
        s["attempts"].append({**ci, "attempt": 2, "updated_at": at(27),
                              "jobs": [job("mac-metal", 27, "a" * 40)]})
        result = evaluate(s)
        self.assertEqual((result["verdict"], result["elapsed_seconds"]), ("approved", 1620))
        self.assertEqual({(j["checkout_sha"], j["run_id"], j["attempt"]) for j in result["required_checks"]},
                         {("a" * 40, 10, 1), ("a" * 40, 10, 2), ("b" * 40, 11, 1)})

    def test_pending_and_superseded_requests(self):
        for minute, expected in [(24, "superseded_before_deadline"), (26, "miss")]:
            s = example()
            s["request"]["superseded_at"] = at(minute)
            s["attempts"][0]["jobs"] = []
            self.assertEqual(evaluate(s)["deadline"], expected)
        s = example()
        s["request"]["superseded_at"] = at(26)
        self.assertEqual(evaluate(s)["deadline"], "on_time")

    def test_missing_skipped_and_canceled_checks(self):
        s = example()
        s["attempts"][0]["jobs"][1]["conclusion"] = "cancelled"
        self.assertEqual(evaluate(s)["deadline"], "miss")
        s["attempts"][0]["jobs"].pop()
        self.assertIn(["ci", "approval-storage-macos-arm64"], evaluate(s)["missing_checks"])
        self.assertEqual(evaluate(s)["deadline"], "miss")
        s["attempts"].pop()
        self.assertEqual(evaluate(s)["verdict"], "unknown")

    def test_uncertain_correlation_and_proxy(self):
        s = example()
        s["request"]["requested_at"] = None
        result = evaluate(s)
        self.assertEqual(result["deadline"], "unscorable")
        self.assertEqual(result["workflow_creation_to_update_proxy_seconds"], 1440)
        for change in ({"pr": 99}, {"event_id": "other"}):
            altered = deepcopy(s)
            altered["attempts"][0].update(change)
            self.assertIn("ambiguous", evaluate(altered)["reason"])
        s["request"]["event"] = "push"
        self.assertEqual(evaluate(s)["deadline"], "ineligible")

    def test_denominator_excludes_early_supersession_and_ambiguous_requests(self):
        snapshots = [example() for _ in range(4)]
        for n, s in enumerate(snapshots):
            for row in [s["request"], *s["attempts"]]:
                row["event_id"] = f"delivery-{n}"
        snapshots[1]["request"]["superseded_at"] = at(24)
        snapshots[2]["attempts"][0]["jobs"] = []
        snapshots[2]["request"]["superseded_at"] = at(26)
        snapshots[3]["request"]["event_id"] = None
        result = cohort(snapshots)
        self.assertEqual(result["scored_deadline_denominator"], 2)
        self.assertEqual(result["superseded_requests"], 2)
        self.assertEqual(result["counts"], {"on_time": 1, "superseded_before_deadline": 1,
                                            "miss": 1, "unscorable": 1})
        with self.assertRaisesRegex(ValueError, "duplicate logical request"):
            cohort([snapshots[0], snapshots[0]])

    def test_newly_queued_retry_cannot_inherit_old_success(self):
        s = example()
        ci = s["attempts"][0]
        s["attempts"].append({**ci, "attempt": 2, "status": "queued", "jobs": []})
        self.assertEqual(evaluate(s)["deadline"], "miss")
        self.assertEqual(evaluate(s)["verdict"], "unknown")

    def test_github_retry_check_created_after_original_execution(self):
        s = example()
        ci = s["attempts"][0]
        ci["jobs"][0]["id"] = 101
        copied = {**ci["jobs"][0], "id": 202, "created_at": at(26),
                  "published_at": None, "checkout_sha": None}
        retry = {**ci, "attempt": 2, "created_at": at(26), "updated_at": at(28),
                 "jobs": [copied, job("approval-storage-macos-arm64", 28, "c" * 40)]}
        retry["jobs"][1].update(created_at=at(26), started_at=at(27))
        s["attempts"].append(retry)
        result = evaluate(s)
        first = next(row for row in result["required_checks"] if row["name"] == "mac-metal")
        self.assertEqual((first["id"], first["attempt"], first["checkout_sha"]), (101, 1, "a" * 40))
        self.assertEqual(result["deadline"], "miss")
        for field, value in (("started_at", at(3)), ("checkout_sha", "b" * 40), ("id", 101)):
            bad = deepcopy(s)
            bad["attempts"][-1]["jobs"][0][field] = value
            with self.assertRaisesRegex(ValueError, "unverified carried-forward"):
                evaluate(bad)

    def test_bad_or_missing_publication(self):
        s = example()
        s["attempts"][0]["jobs"][0]["published_at"] = None
        self.assertEqual(evaluate(s)["deadline"], "unscorable")
        s["attempts"][0]["jobs"][0]["published_at"] = at(31)
        with self.assertRaises(ValueError):
            evaluate(s)

    def test_superseded_after_completion_with_unknown_publication_order(self):
        s = example()
        s["request"]["superseded_at"] = at(26)
        s["attempts"][0]["jobs"][0]["published_at"] = None
        result = evaluate(s)
        self.assertEqual(result["deadline"], "unscorable")
        self.assertIn("unknown publication order", result["reason"])
        s["request"]["superseded_at"] = at(24)
        self.assertEqual(evaluate(s)["deadline"], "superseded_before_deadline")

    def test_reversed_workflow_times_are_invalid(self):
        s = example()
        s["attempts"][0]["updated_at"] = at(0)
        with self.assertRaisesRegex(ValueError, "update precedes"):
            evaluate(s)


if __name__ == "__main__":
    unittest.main()
