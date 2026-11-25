import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ci"))
from m3_reserved_window import run_cases


class WindowTests(unittest.TestCase):
    def test_stop_during_active_fixture_prevents_next_admission(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            command = out / "native"
            stop = out / "STOP"
            command.write_text(
                "#!/bin/sh\n"
                f"while [ ! -e '{stop}' ]; do sleep .05; done\n"
                "echo 'running 1 test'\necho \"test $1 ... ok\"\n"
                "echo 'test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 1 filtered out'\n"
            )
            command.chmod(0o700)
            observed = []
            def signal():
                until = time.monotonic() + 5
                while time.monotonic() < until:
                    log = out / "events.jsonl"
                    if log.exists() and '"fixture_started"' in log.read_text():
                        observed.append(True)
                        break
                    time.sleep(.01)
                stop.touch()
            observer = threading.Thread(target=signal)
            observer.start()
            try:
                rows = run_cases(out, ["first", "second"], {"path": str(command)}, {})
            finally:
                observer.join()
            self.assertEqual(observed, [True])
            self.assertEqual([row["status"] for row in rows], ["passed", "unstarted"])
            events = [json.loads(line) for line in (out / "events.jsonl").read_text().splitlines()]
            self.assertEqual([row["fixture"] for row in events], ["first", "first"])
            self.assertFalse((out / "second.stdout").exists())
