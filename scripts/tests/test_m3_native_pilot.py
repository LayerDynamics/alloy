from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ci"))
import m3_native_pilot as pilot


# Frozen v1 expectations, independent of the runner's mapping and suite tables.
PROFILES = """
gemma-4-e4b-it-4bit ornith-1.5-9b-4bit qwen3.5-0.8b-4bit gemma-3-4b-4bit
gemma-3-12b-4bit phi4-mini-instruct-4bit ministral-3-3b-instruct-2512-4bit
ministral-3-3b-instruct-2512-8bit llama-3.1-8b-4bit llama-3.1-8b-8bit
smollm3-3b-4bit qwen3-4b-4bit qwen3-14b-4bit qwen3-coder-30b-a3b-instruct-4bit
""".split()
TOOL = "crates/alloyd/tests/llama_real_model.rs"
SKILL = "crates/alloyd/tests/compaction_real_model.rs"
EXPECTED = {
    TOOL: ("llama_real_model", ("llama3_tool_stop_context_cache",
                                "llama3_long_context_stream_cancel_memory",
                                "native_validated_context_boundary")),
    SKILL: ("compaction_real_model", ("real_model_compaction_preserves_history_and_continues",
                                     "real_model_ssd_prefix_survives_process_restart",
                                     "real_model_generated_history_has_verified_canonical_bytes",
                                     "real_skill_change_invalidates_stale_cache_and_matches_cold",
                                     "real_rpc_skill_load_changes_next_generation")),
}


def tuples(rows):
    return {(r["profile"], r["package"], r["target"], r["fixture"]) for r in rows}


class PilotPlanTests(unittest.TestCase):
    def test_exact_sets_and_comparator(self):
        for paths in ([TOOL], [SKILL], [TOOL, SKILL, TOOL]):
            expected = set()
            for path in paths:
                target, fixtures = EXPECTED[path]
                expected.update((p, "alloyd", target, f) for p in PROFILES for f in fixtures)
            plan = pilot.select(paths, pilot.SUITES)
            self.assertTrue(plan["complete"])
            self.assertEqual(plan["rule_version"], 1)
            self.assertEqual(tuples(plan["selected"]), expected)
            self.assertEqual(len(plan["selected"]), len(expected))
            self.assertTrue(tuples(plan["omitted"]).isdisjoint(expected))
            self.assertEqual(set(plan["mapped_paths"]), set(paths))
            self.assertEqual(tuples(plan["fixed_comparator"]), {
                (p, "alloyd", "llama_real_model", "llama3_tool_stop_context_cache")
                for p in ("smollm3-3b-4bit", "qwen3.5-0.8b-4bit")
            } | {(p, "alloyd", "compaction_real_model", "real_skill_change_invalidates_stale_cache_and_matches_cold")
                 for p in ("qwen3-14b-4bit", "llama-3.1-8b-8bit")})

    def test_empty_unmapped_and_mixed(self):
        shared = "crates/engine/src/lib.rs"
        for paths in ([], [shared], [TOOL, shared, shared]):
            plan = pilot.select(paths, pilot.SUITES)
            self.assertFalse(plan["complete"])
            self.assertEqual(bool(plan["selected"]), TOOL in paths)
            self.assertEqual(plan["path_counts"], {"mapped": int(TOOL in paths), "total": len(set(paths))})
            self.assertEqual([r["path"] for r in plan["unknown_paths"]], [shared] if shared in paths else [])

    def test_diverged_base_and_both_rename_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def git(*args):
                return subprocess.check_output(["git", "-C", tmp, "-c", "user.name=CI Test",
                                                "-c", "user.email=ci@example.invalid",
                                                "-c", "commit.gpgsign=false",
                                                "-c", "core.hooksPath=/dev/null", *args], text=True).strip()

            git("init", "-q", "-b", "base")
            path = root / TOOL
            path.parent.mkdir(parents=True)
            path.write_text("fixture\n")
            git("add", ".")
            git("commit", "-qm", "initial")
            ancestor = git("rev-parse", "HEAD")
            git("switch", "-qc", "candidate")
            renamed = "renamed fixture.rs"
            path.rename(root / renamed)
            git("add", "-A")
            git("commit", "-qm", "rename")
            candidate = git("rev-parse", "HEAD")
            git("switch", "-q", "base")
            (root / "base-only").touch()
            git("add", ".")
            git("commit", "-qm", "diverge")
            merge_base, paths = pilot.revisions(root, git("rev-parse", "HEAD"), candidate)
            self.assertEqual(merge_base, ancestor)
            self.assertEqual(set(paths), {TOOL, renamed})
            self.assertFalse(pilot.select(paths, pilot.SUITES)["complete"])


if __name__ == "__main__":
    unittest.main()
