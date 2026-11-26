"""Conservative runner-account process checks shared by Mac CI entry points."""
import os
import subprocess
import time


def runner_processes():
    # Same real UID includes detached descendants even after reparenting or
    # environment clearing. Start time distinguishes reused process IDs.
    with subprocess.Popen(
        ["/bin/ps", "-U", str(os.getuid()), "-o", "pid=,lstart="],
        stdout=subprocess.PIPE, text=True, env={**os.environ, "LC_ALL": "C"},
    ) as probe:
        try:
            output, _ = probe.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            probe.kill()
            probe.communicate()
            raise
        if probe.returncode:
            raise RuntimeError("Could not inspect runner processes")
        processes = {}
        for line in output.splitlines():
            pid, started = line.split(maxsplit=1)
            if int(pid) != probe.pid:
                processes[int(pid)] = started
        if os.getpid() not in processes:
            raise RuntimeError("Runner process snapshot is incomplete")
        return processes


def check_quiescence(before):
    for sample in range(2):
        if sample:
            time.sleep(0.2)
        after = runner_processes()
        survivors = sorted(pid for pid, start in after.items() if before.get(pid) != start)
        if survivors:
            raise RuntimeError(f"Runner processes remain after test failure: {survivors}")


