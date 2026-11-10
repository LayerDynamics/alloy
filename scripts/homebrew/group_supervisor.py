#!/usr/bin/env python3
"""Single-threaded owner of a session leader; control messages arrive on fd 3.

The target remains our unreaped child through all numeric signals and group
liveness checks. No other process can reap it or recycle its PID/PGID then.
"""
import json
import os
import select
import signal
import subprocess
import sys
import time

CONTROL = 3
ENVIRONMENT = 4
MAX_ENVIRONMENT = 2 * 1024 * 1024


def report(value):
    data = (json.dumps(value, separators=(",", ":")) + "\n").encode()
    view = memoryview(data)
    while view:
        view = view[os.write(CONTROL, view):]


def live_group(group):
    members = []
    result = subprocess.run(["/bin/ps", "-axo", "pid=,pgid=,stat="], check=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            timeout=3)
    for line in result.stdout.splitlines():
        columns = line.split()
        if len(columns) != 3:
            raise RuntimeError("invalid process table")
        if int(columns[1]) == group and not columns[2].startswith("Z"):
            members.append(int(columns[0]))
    return members


def main():
    if len(sys.argv) < 2:
        raise RuntimeError("missing supervised command")
    signal.signal(signal.SIGCHLD, signal.SIG_DFL)
    os.set_inheritable(CONTROL, False)
    os.set_inheritable(ENVIRONMENT, False)
    env_bytes = bytearray()
    while True:
        chunk = os.read(ENVIRONMENT, 65536)
        if not chunk:
            break
        env_bytes.extend(chunk)
        if len(env_bytes) > MAX_ENVIRONMENT:
            raise RuntimeError("target environment exceeds supervisor limit")
    os.close(ENVIRONMENT)
    target_env = json.loads(env_bytes)
    if not isinstance(target_env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in target_env.items()):
        raise RuntimeError("invalid target environment")
    try:
        child = subprocess.Popen(sys.argv[1:], stdin=subprocess.DEVNULL,
                                 env=target_env, start_new_session=True, close_fds=True)
    except BaseException as error:
        report({"event": "exit", "code": None, "signal": None, "error": str(error)})
        return 1
    group = child.pid
    try:
        return supervise(child, group)
    except BaseException:
        # The helper is still the sole waiter, so the child remains unreaped.
        # A broken control pipe or a failed process-table probe must not leave a
        # live target without attempting cleanup. No numeric signal follows a
        # successful wait/reap; supervise clears child.pid in its final step.
        if child.pid is not None:
            try:
                os.killpg(group, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass
        raise


def observe(child, group):
    try:
        return os.waitid(os.P_PID, group, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    except ChildProcessError:
        # Lost ownership must never be followed by a numeric signal.
        child.pid = None
        raise


def supervise(child, group):
    report({"event": "ready", "pid": group})
    target_exited = False
    eof = False
    forced = False
    forced_deadline = None
    while not target_exited:
        info = observe(child, group)
        if info is not None and info.si_pid == group:
            target_exited = True
            break
        if forced_deadline is not None and time.monotonic() >= forced_deadline:
            raise RuntimeError("target did not exit after SIGKILL; cleanup not established")
        readable, _, _ = select.select([CONTROL], [], [], 0.02)
        if not readable:
            continue
        data = os.read(CONTROL, 256)
        if not data:
            eof = True
            data = b"K"
        for command in data:
            if command in (ord("T"), ord("K")):
                sig = signal.SIGTERM if command == ord("T") else signal.SIGKILL
                try:
                    os.killpg(group, sig)
                except ProcessLookupError:
                    pass
                except PermissionError:
                    if live_group(group):
                        raise
                if command == ord("K"):
                    forced = True
                    forced_deadline = forced_deadline or time.monotonic() + 3
            elif command in (ord("I"), ord("Q")):
                try:
                    os.kill(group, signal.SIGINT if command == ord("I") else signal.SIGTERM)
                except ProcessLookupError:
                    pass
        if eof:
            # Do not repeatedly select an EOF descriptor while awaiting exit.
            while True:
                if time.monotonic() >= forced_deadline:
                    raise RuntimeError("orphaned target did not exit")
                info = observe(child, group)
                if info is not None and info.si_pid == group:
                    target_exited = True
                    break
                time.sleep(0.02)
    # The WNOWAIT child pins the numeric group leader through the final kill,
    # every liveness check, and all possible retries.
    survivors_before_cleanup = live_group(group)
    if survivors_before_cleanup:
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError:
            if live_group(group):
                raise
    deadline = time.monotonic() + 3
    while live_group(group):
        if time.monotonic() >= deadline:
            raise RuntimeError("owned process group cleanup not established")
        time.sleep(0.025)
    # Python turns SIGINT into KeyboardInterrupt. Block asynchronous signals
    # across waitpid and retiring the numeric identity so an exception cannot
    # enter the outer cleanup handler after the child has been reaped.
    mask = {signal.SIGINT, signal.SIGTERM, signal.SIGHUP}
    old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, mask)
    try:
        try:
            status = child.wait()
        except ChildProcessError:
            child.pid = None
            raise
        child.pid = None  # no numeric group signal is allowed after reaping
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
    if not eof:
        report({"event": "exit", "code": status if status >= 0 else None,
                "signal": None if status >= 0 else signal.Signals(-status).name,
                "forced": forced, "cleanup_verified": True,
                "live_before_cleanup": survivors_before_cleanup})
    return 0


if __name__ == "__main__":
    sys.exit(main())
