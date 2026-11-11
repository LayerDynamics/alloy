#!/usr/bin/env python3
"""Serialize cooperating CI shell steps on one physical host.

The advisory lock is per host; distinct Macs do not share it.
Hold it until the step and its signal-aware cleanup have completed.
"""
import fcntl
import os
import signal
import stat
import subprocess
import sys
import tempfile
import time


def main():
    args = sys.argv[1:]
    cleanup_only = bool(args and args[0] == "--cleanup")
    if cleanup_only:
        args = args[1:]
    if not args:
        raise SystemExit("usage: mac_locked_step.py [--cleanup] STEP_SCRIPT | -- COMMAND [ARG ...]")
    if args[0] == "--":
        if len(args) < 2:
            raise SystemExit("missing command")
        command = args[1:]
    else:
        command = ["/bin/bash", "--noprofile", "--norc", "-e", "-o", "pipefail", args[0]]
    # Ignore per-runner TMPDIR: runner agents on the same host must share a lock.
    path = "/tmp/alloy-ci-mac-host.lock"
    if os.environ.get("GITHUB_ACTIONS") != "true":
        path = os.environ.get("ALLOY_CI_LOCK_TEST_PATH", path)
    injection = os.environ.get("ALLOY_CI_LOCK_TEST_TRANSITION") if os.environ.get("GITHUB_ACTIONS") != "true" else None
    flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except FileNotFoundError:
        # Publish a complete record atomically, never an empty visible file.
        initial, tmp = tempfile.mkstemp(prefix=".alloy-ci-lease-", dir=os.path.dirname(path))
        try:
            os.fchmod(initial, 0o666)
            if os.write(initial, b"clean\n") != 6:
                raise SystemExit("could not initialize CI host lease")
            os.fsync(initial)
            if injection == "initial_race":
                time.sleep(0.2)
            try:
                os.link(tmp, path)
            except FileExistsError:
                pass
        finally:
            os.close(initial)
            os.unlink(tmp)
        fd = os.open(path, flags)
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink < 1 or stat.S_IMODE(info.st_mode) != 0o666:
        raise SystemExit("unsafe CI host lock file")
    child = None
    interrupted = None

    def stop(signum, _frame):
        nonlocal interrupted
        interrupted = signum
        if child is None:
            return
        try:
            os.killpg(child.pid, signum)
        except ProcessLookupError:
            pass

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, stop)
    # A runner may inherit SIGCHLD=SIG_IGN. This wrapper must own its child.
    signal.signal(signal.SIGCHLD, signal.SIG_DFL)
    try:
        while True:
            if interrupted is not None:
                return 128 + interrupted
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                time.sleep(0.1)
        state = os.pread(fd, 32, 0)
        if state != b"clean\n" and not cleanup_only:
            raise SystemExit("previous CI host lease ended unexpectedly; inspect host processes before resetting the lock")
        was_clean = state == b"clean\n"
        if injection == "partial_dirty":
            os.pwrite(fd, b"d", 0)
            os._exit(86)
        if os.pwrite(fd, b"dirty\n", 0) != 6:
            raise SystemExit("could not mark CI host lease dirty")
        os.fsync(fd)
        if interrupted is not None:
            return 128 + interrupted
        # The child inherits the lock, so a killed wrapper does not immediately
        # release it while the step shell is still alive.
        child = subprocess.Popen(command, start_new_session=True, pass_fds=(fd,))
        if interrupted is not None:
            stop(interrupted, None)
        # Block signals across each nonblocking reap and clearing of child.
        # No handler can target a reused PID after waitpid has reaped it.
        while True:
            old = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM, signal.SIGHUP})
            try:
                waited, status = os.waitpid(child.pid, os.WNOHANG)
                if waited:
                    child.returncode = os.waitstatus_to_exitcode(status)
                    child = None
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, old)
            if waited:
                break
            time.sleep(0.1)
        status = os.waitstatus_to_exitcode(status)
        if injection == "signal_before_clean":
            os.kill(os.getpid(), signal.SIGTERM)
        watched = {signal.SIGINT, signal.SIGTERM, signal.SIGHUP}
        old = signal.pthread_sigmask(signal.SIG_BLOCK, watched)
        try:
            if interrupted is None and not (signal.sigpending() & watched) and status == 0 and (not cleanup_only or was_clean):
                if injection == "before_clean":
                    os._exit(86)
                if injection == "partial_clean":
                    os.pwrite(fd, b"c", 0)
                    os._exit(86)
                if injection == "signal_during_clean":
                    os.kill(os.getpid(), signal.SIGTERM)
                if os.pwrite(fd, b"clean\n", 0) != 6:
                    raise SystemExit("could not mark CI host lease clean")
                os.fsync(fd)
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, old)
        return 128 + interrupted if interrupted is not None else (status if status >= 0 else 128 - status)
    finally:
        if child is not None:
            # The wrapper is the only waiter for this still-unreaped child.
            old = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT, signal.SIGTERM, signal.SIGHUP})
            try:
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                while True:
                    try:
                        _, status = os.waitpid(child.pid, 0)
                        child.returncode = os.waitstatus_to_exitcode(status)
                        child = None
                        break
                    except InterruptedError:
                        continue
            finally:
                signal.pthread_sigmask(signal.SIG_SETMASK, old)
        os.close(fd)


if __name__ == "__main__":
    sys.exit(main())
