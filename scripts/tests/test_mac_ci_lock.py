import os
import pathlib
import signal
import subprocess
import sys
import tempfile
import time
import unittest

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "ci/mac_locked_step.py"


class MacCiLockTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)
        self.env = os.environ.copy()
        self.env.pop("GITHUB_ACTIONS", None)
        self.env["ALLOY_CI_LOCK_TEST_PATH"] = str(self.root / "host.lock")

    def launch(self, code):
        return subprocess.Popen(
            [sys.executable, str(SCRIPT), "--", sys.executable, "-c", code],
            env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )

    def test_steps_on_same_host_do_not_overlap(self):
        events = self.root / "events"
        code = (
            "import sys,time\n"
            f"p={str(events)!r}\n"
            "with open(p,'a') as f: f.write(sys.argv[1]+' start\\n')\n"
            "time.sleep(.25)\n"
            "with open(p,'a') as f: f.write(sys.argv[1]+' end\\n')\n"
        )
        first = self.launch(code.replace("sys.argv[1]", repr("first")))
        second = self.launch(code.replace("sys.argv[1]", repr("second")))
        for child in (first, second):
            _, err = child.communicate(timeout=10)
            self.assertEqual(child.returncode, 0, err)
        lines = events.read_text().splitlines()
        self.assertIn(lines, [["first start", "first end", "second start", "second end"],
                              ["second start", "second end", "first start", "first end"]])

    def test_cancelled_waiter_does_not_run_and_symlink_is_rejected(self):
        ready = self.root / "ready"
        marker = self.root / "unexpected"
        holder = self.launch(f"import pathlib,time; pathlib.Path({str(ready)!r}).touch(); time.sleep(1)")
        self.addCleanup(lambda: holder.poll() is None and holder.kill())
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertTrue(ready.exists())
        waiter = self.launch(f"import pathlib; pathlib.Path({str(marker)!r}).touch()")
        time.sleep(.15)
        waiter.send_signal(signal.SIGTERM)
        _, err = waiter.communicate(timeout=5)
        self.assertEqual(waiter.returncode, 128 + signal.SIGTERM, err)
        self.assertFalse(marker.exists())
        holder.communicate(timeout=5)
        self.assertEqual(holder.returncode, 0)
        os.unlink(self.env["ALLOY_CI_LOCK_TEST_PATH"])
        os.symlink(str(self.root / "target"), self.env["ALLOY_CI_LOCK_TEST_PATH"])
        rejected = subprocess.run(
            [sys.executable, str(SCRIPT), "--", sys.executable, "-c", "pass"],
            env=self.env, capture_output=True, text=True, timeout=5,
        )
        self.assertNotEqual(rejected.returncode, 0)
        self.assertFalse((self.root / "target").exists())

    def test_cancelled_holder_keeps_lock_until_child_finishes_cleanup(self):
        ready, cleaned = self.root / "ready", self.root / "cleaned"
        code = (
            "import pathlib,signal,sys,time\n"
            "def stop(*_):\n"
            f" time.sleep(.3); pathlib.Path({str(cleaned)!r}).touch(); sys.exit(0)\n"
            "signal.signal(signal.SIGTERM,stop)\n"
            f"pathlib.Path({str(ready)!r}).touch()\n"
            "while True: time.sleep(.05)\n"
        )
        holder = self.launch(code)
        self.addCleanup(lambda: holder.poll() is None and holder.kill())
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertTrue(ready.exists())
        holder.send_signal(signal.SIGTERM)
        waiter = self.launch(f"import pathlib,sys; sys.exit(0 if pathlib.Path({str(cleaned)!r}).exists() else 7)")
        _, err = waiter.communicate(timeout=5)
        self.assertNotEqual(waiter.returncode, 0)
        self.assertIn("lease ended unexpectedly", err)
        _, err = holder.communicate(timeout=5)
        self.assertEqual(holder.returncode, 128 + signal.SIGTERM, err)
        self.assertTrue(cleaned.exists(), "waiter acquired the lock before child cleanup")

    def test_killed_wrapper_does_not_release_live_step_lock(self):
        ready, done = self.root / "ready", self.root / "done"
        holder = self.launch(
            f"import pathlib,time; pathlib.Path({str(ready)!r}).touch(); time.sleep(.5); pathlib.Path({str(done)!r}).touch()"
        )
        self.addCleanup(lambda: holder.poll() is None and holder.kill())
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertTrue(ready.exists())
        holder.kill()
        waiter = self.launch(f"import pathlib,sys; sys.exit(0 if pathlib.Path({str(done)!r}).exists() else 7)")
        _, err = waiter.communicate(timeout=5)
        self.assertNotEqual(waiter.returncode, 0)
        self.assertIn("lease ended unexpectedly", err)
        self.assertTrue(done.exists())
        holder.communicate(timeout=5)
        self.assertEqual(holder.returncode, -signal.SIGKILL)

    def test_dirty_lease_blocks_after_detached_child_closes_descriptors(self):
        ready, done = self.root / "ready", self.root / "done"
        grandchild = f"import pathlib,time; time.sleep(1); pathlib.Path({str(done)!r}).touch()"
        parent = (
            "import pathlib,subprocess,sys,time\n"
            f"subprocess.Popen([sys.executable,'-c',{grandchild!r}], start_new_session=True, "
            "close_fds=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
            f"pathlib.Path({str(ready)!r}).touch()\n"
            "time.sleep(.25)\n"
        )
        holder = self.launch(parent)
        self.addCleanup(lambda: holder.poll() is None and holder.kill())
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertTrue(ready.exists())
        holder.kill()
        waiter = self.launch("raise SystemExit(0)")
        _, err = waiter.communicate(timeout=5)
        self.assertNotEqual(waiter.returncode, 0)
        self.assertIn("lease ended unexpectedly", err)
        self.assertFalse(done.exists(), "test did not observe the still-running detached child")
        holder.communicate(timeout=5)
        while not done.exists() and time.monotonic() < deadline:
            time.sleep(.02)
        self.assertTrue(done.exists(), "detached fixture did not finish")

    def test_failed_step_leaves_dirty_and_cleanup_cannot_clear_it(self):
        failed = self.launch("raise SystemExit(7)")
        failed.communicate(timeout=5)
        self.assertEqual(failed.returncode, 7)
        marker = self.root / "cleanup"
        cleanup = subprocess.run(
            [sys.executable, str(SCRIPT), "--cleanup", "--", sys.executable, "-c",
             f"import pathlib; pathlib.Path({str(marker)!r}).touch()"],
            env=self.env, capture_output=True, text=True, timeout=5,
        )
        self.assertEqual(cleanup.returncode, 0, cleanup.stderr)
        self.assertTrue(marker.exists())
        refused = self.launch("raise SystemExit(0)")
        _, err = refused.communicate(timeout=5)
        self.assertNotEqual(refused.returncode, 0)
        self.assertIn("lease ended unexpectedly", err)

    def test_empty_or_interrupted_state_transitions_fail_closed(self):
        for transition in ("empty", "partial_dirty", "before_clean", "partial_clean", "signal_before_clean"):
            with self.subTest(transition=transition):
                env = self.env.copy()
                lock = self.root / f"lock-{transition}"
                env["ALLOY_CI_LOCK_TEST_PATH"] = str(lock)
                if transition == "empty":
                    lock.touch(mode=0o666)
                    lock.chmod(0o666)
                else:
                    env["ALLOY_CI_LOCK_TEST_TRANSITION"] = transition
                    broken = subprocess.run(
                        [sys.executable, str(SCRIPT), "--", sys.executable, "-c", "pass"],
                        env=env, capture_output=True, text=True, timeout=5,
                    )
                    self.assertEqual(broken.returncode, 143 if transition == "signal_before_clean" else 86, broken.stderr)
                    env.pop("ALLOY_CI_LOCK_TEST_TRANSITION")
                refused = subprocess.run(
                    [sys.executable, str(SCRIPT), "--", sys.executable, "-c", "pass"],
                    env=env, capture_output=True, text=True, timeout=5,
                )
                self.assertNotEqual(refused.returncode, 0)
                self.assertIn("lease ended unexpectedly", refused.stderr)

    def test_simultaneous_first_use_publishes_complete_record(self):
        self.env["ALLOY_CI_LOCK_TEST_TRANSITION"] = "initial_race"
        children = [self.launch("pass"), self.launch("pass")]
        for child in children:
            _, err = child.communicate(timeout=5)
            self.assertEqual(child.returncode, 0, err)

    def test_signal_after_successful_step_reap_may_leave_clean_lease(self):
        self.env["ALLOY_CI_LOCK_TEST_TRANSITION"] = "signal_during_clean"
        child = self.launch("pass")
        _, err = child.communicate(timeout=5)
        self.assertEqual(child.returncode, 143, err)
        self.assertEqual(pathlib.Path(self.env["ALLOY_CI_LOCK_TEST_PATH"]).read_bytes(), b"clean\n")


if __name__ == "__main__":
    unittest.main()
