#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <stdint.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#if !defined(__aarch64__) && !defined(__x86_64__)
#error "Alloy Linux plugin supervision requires a verified aarch64 or x86_64 signal ABI"
#endif

/* The parent resolves libc's syscall/errno entry points before clone. Every
 * post-clone operation below uses that thin syscall entry point, not libc's
 * cancellation, signal, loader, or stdio wrappers. */
static void clear_bytes(void *buffer, unsigned long length) {
    volatile unsigned char *p = buffer;
    for (unsigned long i = 0; i < length; ++i) p[i] = 0;
}
static _Noreturn void child_exit(int status) {
    syscall(SYS_exit_group, status);
    __builtin_unreachable();
}
static _Noreturn void target_failure(int error) {
    if (!error) error=EIO;
    (void)syscall(SYS_write, 5, &error, sizeof(error));
    child_exit(125);
}
static int child_close(int fd) { return (int)syscall(SYS_close, fd); }
static int child_poll(struct pollfd *pfd, int milliseconds) {
    struct timespec timeout = {milliseconds / 1000, (long)(milliseconds % 1000) * 1000000};
    return (int)syscall(SYS_ppoll, pfd, 1, &timeout, 0, sizeof(uint64_t));
}
static void child_send(int fd, const char *value) {
    (void)syscall(SYS_sendto, fd, value, 1, MSG_NOSIGNAL, 0, 0);
}
struct linux_dirent64 { uint64_t ino; int64_t off; unsigned short reclen; unsigned char type; char name[]; };
/* Linux x86_64 and asm-generic (including aarch64) kernel rt_sigaction ABI.
 * This deliberately is NOT glibc's differently laid out struct sigaction. */
struct kernel_sigaction { uintptr_t handler; unsigned long flags; uintptr_t restorer; uint64_t mask; };
_Static_assert(sizeof(struct kernel_sigaction) == 32, "Linux 64-bit kernel signal ABI");
static long kernel_sigaction(int sig, const struct kernel_sigaction *action, struct kernel_sigaction *old) {
    return syscall(SYS_rt_sigaction, sig, action, old, sizeof(uint64_t));
}
/* Only syscalls and local buffers: supervisor was forked from a multithreaded host. */
static int group_live(pid_t group) {
    int dir = (int)syscall(SYS_openat, AT_FDCWD, "/proc", O_RDONLY | O_DIRECTORY | O_CLOEXEC, 0);
    if (dir < 0) return -errno;
    union { uint64_t alignment; char bytes[8192]; } entries;
    for (;;) {
        long bytes = syscall(SYS_getdents64, dir, entries.bytes, sizeof(entries.bytes));
        if (bytes < 0) { int e = errno; child_close(dir); return -e; }
        if (!bytes) { child_close(dir); return 0; }
        for (long off = 0; off < bytes;) {
            struct linux_dirent64 *d = (struct linux_dirent64 *)(entries.bytes + off);
            if (d->reclen < 20 || off + d->reclen > bytes) { child_close(dir); return -EIO; }
            off += d->reclen;
            unsigned len = 0;
            while (len < 20 && d->name[len] >= '0' && d->name[len] <= '9') ++len;
            if (!len || d->name[len]) continue;
            int pid = 0;
            for (unsigned i = 0; i < len; ++i) {
                if (pid > 214748364) { child_close(dir); return -EIO; }
                pid = pid * 10 + (d->name[i] - '0');
            }
            pid_t pgid = (pid_t)syscall(SYS_getpgid, pid);
            if (pgid < 0) { if (errno == ESRCH) continue; int e = errno; child_close(dir); return -e; }
            if (pgid != group) continue;
            char path[32];
            if (len + 6 >= sizeof(path)) { child_close(dir); return -EIO; }
            /* Volatile stores prevent the compiler introducing a lazy-bound
             * libc memcpy into this post-clone path. */
            volatile char *path_bytes = path;
            for (unsigned i = 0; i < len; ++i) path_bytes[i] = d->name[i];
            path[len++] = '/'; path[len++] = 's'; path[len++] = 't';
            path[len++] = 'a'; path[len++] = 't'; path[len] = 0;
            int fd = (int)syscall(SYS_openat, dir, path, O_RDONLY | O_CLOEXEC, 0);
            if (fd < 0) { if (errno == ENOENT || errno == ESRCH) continue; int e = errno; child_close(dir); return -e; }
            char buf[4096]; ssize_t n = syscall(SYS_read, fd, buf, sizeof(buf)-1); int e = errno; child_close(fd);
            if (n < 0) { if (e == ENOENT || e == ESRCH) continue; child_close(dir); return -e; }
            if (!n) continue;
            buf[n] = 0;
            char *end = 0;
            for (ssize_t i = 0; i < n; ++i) if (buf[i] == ')') end = buf + i;
            if (!end || end + 4 >= buf + n || end[1] != ' ') { child_close(dir); return -EIO; }
            char state = end[2];
            char *v = end + 4;
            while (*v >= '0' && *v <= '9') ++v;
            if (*v++ != ' ') { child_close(dir); return -EIO; }
            int observed_group = 0, digits = 0;
            while (*v >= '0' && *v <= '9') { if (observed_group > 214748364) { child_close(dir); return -EIO; } observed_group = observed_group*10 + (*v++ - '0'); ++digits; }
            if (!digits || *v != ' ') { child_close(dir); return -EIO; }
            if (observed_group == group && state != 'Z' && state != 'X') { child_close(dir); return 1; }
        }
    }
}
static uint64_t monotonic_ms(void) {
    struct timespec t;
    if (syscall(SYS_clock_gettime, CLOCK_MONOTONIC, &t)) return 0;
    return (uint64_t)t.tv_sec * 1000 + (uint64_t)t.tv_nsec / 1000000;
}

/* Runs after fork in a multi-threaded host: only async-signal-safe operations.
 * Returns only in the target; the isolated supervisor always uses _exit. */
int alloy_linux_supervise(int control, int parent_end) {
    (void)parent_end;
    if (control != 3 && syscall(SYS_dup3, control, 3, 0) < 0) child_exit(125);
    if (syscall(SYS_close_range, 4u, ~0u, 0u)) child_exit(125);
    struct kernel_sigaction original_sigchld;
    if (kernel_sigaction(SIGCHLD, 0, &original_sigchld)) child_exit(125);
    struct kernel_sigaction action = {0};
    if (kernel_sigaction(SIGCHLD, &action, 0)) child_exit(125);
    /* The target must never execute unless the host commits launch.  EOF,
     * cancellation, or an abandoned bootstrap exits before creating it. */
    struct pollfd ready = {3, POLLIN | POLLHUP, 0};
    int ready_result;
    do { ready_result = child_poll(&ready, 5000); } while (ready_result < 0 && errno == EINTR);
    char commit = 0;
    if (ready_result <= 0 || syscall(SYS_recvfrom, 3, &commit, 1, 0, 0, 0) != 1 || (commit != 'R' && commit != 'Q')) child_exit(125);
    int exec_pipe[2] = {-1, -1};
    if (commit == 'Q' && syscall(SYS_pipe2, exec_pipe, O_CLOEXEC | O_NONBLOCK)) child_exit(125);
    pid_t target = (pid_t)syscall(SYS_clone, SIGCHLD, 0, 0, 0, 0);
    if (target < 0) child_exit(125);
    if (target == 0) {
        (void)syscall(SYS_close, 3);
        if (commit == 'Q') {
            child_close(exec_pipe[0]);
            if (exec_pipe[1] != 5 && syscall(SYS_dup3, exec_pipe[1], 5, O_CLOEXEC) < 0) child_exit(125);
        }
        if (syscall(SYS_setpgid, 0, 0)) target_failure(errno);
        if (commit == 'Q') {
            int null = syscall(SYS_openat, AT_FDCWD, "/dev/null", O_RDONLY | O_CLOEXEC, 0);
            if (null < 0 || (null != 0 && syscall(SYS_dup3, null, 0, 0) < 0)) target_failure(errno);
            if (null != 0) child_close(null);
        }
        /* exec preserves ignored dispositions. Only the supervisor needs its
         * private SIGCHLD reset to retain target ownership. */
        if (original_sigchld.handler == (uintptr_t)SIG_IGN &&
            kernel_sigaction(SIGCHLD, &original_sigchld, 0)) target_failure(errno);
        return 0;
    }
    if (commit == 'Q') {
        child_close(exec_pipe[1]);
        int exec_error = 0;
        long n;
        for (;;) {
            n = syscall(SYS_read, exec_pipe[0], &exec_error, sizeof(exec_error));
            if (n >= 0 || (errno != EINTR && errno != EAGAIN)) break;
            struct pollfd cancel = {3, POLLIN | POLLHUP, 0};
            if (child_poll(&cancel, 1) > 0) { exec_error = ECANCELED; n = sizeof(exec_error); break; }
        }
        child_close(exec_pipe[0]);
        char response[5];
        response[0] = n == 0 ? 'S' : 'F';
        int value = n == 0 ? target : (n == sizeof(exec_error) ? exec_error : EIO);
        for (unsigned i=0; i<4; ++i) response[i+1] = (unsigned)value >> (i*8);
        (void)syscall(SYS_sendto, 3, response, sizeof(response), MSG_NOSIGNAL, 0, 0);
    }
    /* The child sets its own group before executing configured code. */
    child_close(0); child_close(1); child_close(2);
    for (;;) {
        siginfo_t info; clear_bytes(&info, sizeof(info));
        if (syscall(SYS_waitid, P_PID, (id_t)target, &info, WEXITED | WNOHANG | WNOWAIT, 0) < 0) {
            if (errno == EINTR) continue;
            child_exit(125);
        }
        if (info.si_pid == target) break;
        struct pollfd pfd = {3, POLLIN | POLLHUP, 0};
        int n = child_poll(&pfd, 10);
        if (n > 0) break;
        if (n < 0 && errno != EINTR) break;
    }
    /* An unreaped direct child pins target's PID and any group bearing it. */
    uint64_t until = monotonic_ms() + 2000;
    int cleanup_ok = 0, reported = 0;
    for (;;) {
        int group_error = syscall(SYS_kill, -target, SIGKILL) && errno != ESRCH;
        int child_error = syscall(SYS_kill, target, SIGKILL) && errno != ESRCH;
        int live = group_live(target);
        if (!group_error && !child_error && live == 0) { cleanup_ok = 1; break; }
        if (!reported && monotonic_ms() >= until) {
            child_send(3, "E");
            reported = 1;
        }
        struct timespec delay = {0, reported ? 100000000 : 1000000};
        syscall(SYS_nanosleep, &delay, 0);
    }
    int status;
    pid_t waited;
    until = monotonic_ms() + 2000;
    do {
        waited = syscall(SYS_wait4, target, &status, WNOHANG, 0);
        if (waited == target || (waited < 0 && errno != EINTR)) break;
        if (!reported && monotonic_ms() >= until) { child_send(3, "E"); reported = 1; }
        struct timespec delay = {0, reported ? 100000000 : 1000000};
        syscall(SYS_nanosleep, &delay, 0);
    } while (1);
    if (cleanup_ok && !reported && waited == target) {
        child_send(3, "D");
        if (commit == 'Q') {
            unsigned char bytes[4];
            for (unsigned i=0; i<4; ++i) bytes[i] = (unsigned)status >> (i*8);
            (void)syscall(SYS_sendto, 3, bytes, sizeof(bytes), MSG_NOSIGNAL, 0, 0);
        }
    }
    child_close(3);
    child_exit(0);
}

/* Create the host-visible pidfd atomically with the supervisor.  No numeric
 * wait or signal is ever needed, even if another host reaper wins immediately. */
static int move_fd(int *fd) {
    if (*fd >= 3) return 0;
    int n = fcntl(*fd, F_DUPFD_CLOEXEC, 3);
    if (n < 0) return -1;
    close(*fd); *fd = n; return 0;
}
static int new_pipe(int p[2]) {
    if (pipe2(p, O_CLOEXEC)) return -1;
    if (move_fd(&p[0]) || move_fd(&p[1])) { int e = errno; close(p[0]); close(p[1]); errno = e; return -1; }
    return 0;
}
int alloy_linux_launch(const char *program, char *const argv[], const char *cwd, int result[5]) {
    int in[2] = {-1,-1}, out[2] = {-1,-1}, err[2] = {-1,-1}, ctl[2] = {-1,-1};
    if (new_pipe(in) || new_pipe(out) || new_pipe(err)) goto failure;
    if (socketpair(AF_UNIX, SOCK_STREAM | SOCK_CLOEXEC, 0, ctl)) goto failure;
    if (move_fd(&ctl[0]) || move_fd(&ctl[1])) goto failure;
    sigset_t all, previous;
    sigfillset(&all);
    int mask_error = pthread_sigmask(SIG_SETMASK, &all, &previous);
    if (mask_error) { errno = mask_error; goto failure; }
    /* Resolve the errno accessor in the parent before cloning. */
    int *error_slot = &errno;
    *error_slot = 0;
    int pidfd = -1;
    pid_t pid = (pid_t)syscall(SYS_clone, (unsigned long)(SIGCHLD | CLONE_PIDFD), 0, &pidfd, 0, 0);
    int clone_error = errno;
    if (pid == 0) {
        if (syscall(SYS_dup3, in[0], 0, 0) < 0 || syscall(SYS_dup3, out[1], 1, 0) < 0 || syscall(SYS_dup3, err[1], 2, 0) < 0) child_exit(125);
        if (alloy_linux_supervise(ctl[1], ctl[0]) != 0) child_exit(125);
        /* exec resets caught handlers; do that before unblocking so no
         * application handler can run in the forked, multithreaded image. */
        for (int sig = 1; sig < NSIG; ++sig) {
            struct kernel_sigaction old;
            if (kernel_sigaction(sig, 0, &old) != 0) continue;
            if (old.handler != (uintptr_t)SIG_DFL && old.handler != (uintptr_t)SIG_IGN) {
                struct kernel_sigaction clean = {0};
                if (kernel_sigaction(sig, &clean, 0)) target_failure(errno);
            }
        }
        if (syscall(SYS_rt_sigprocmask, SIG_SETMASK, &previous, 0, sizeof(uint64_t)) ||
            syscall(SYS_chdir, cwd)) { int e = errno; syscall(SYS_write, 5, &e, sizeof(e)); child_exit(125); }
        char *environment[] = {0};
        syscall(SYS_execve, program, argv, environment);
        int exec_error = errno;
        syscall(SYS_write, 5, &exec_error, sizeof(exec_error));
        child_exit(127);
    }
    int restore_error = pthread_sigmask(SIG_SETMASK, &previous, 0);
    if (pid < 0) { errno = clone_error; goto failure; }
    close(in[0]); close(out[1]); close(err[1]); close(ctl[1]);
    result[0] = in[1]; result[1] = out[0]; result[2] = err[0]; result[3] = ctl[0]; result[4] = pidfd;
    /* A restore error cannot safely discard the atomically acquired pidfd.
     * Return descriptors so caller can cancel and reap the exact supervisor. */
    return restore_error ? restore_error : 0;
failure: {
    int e = errno;
    for (int i=0; i<2; ++i) { if (in[i]>=0) close(in[i]); if (out[i]>=0) close(out[i]); if (err[i]>=0) close(err[i]); if (ctl[i]>=0) close(ctl[i]); }
    return e ? e : EIO;
}}
int alloy_linux_wait_pidfd(int fd) {
    struct pollfd pfd = {fd, POLLIN, 0};
    uint64_t now = monotonic_ms();
    if (!now) return EIO;
    uint64_t until = now + 2500;
    for (;;) {
        now = monotonic_ms();
        if (!now) return EIO;
        if (now >= until) return ETIMEDOUT;
        int rc = poll(&pfd, 1, (int)(until - now));
        if (rc < 0 && errno == EINTR) continue;
        if (rc < 0) return errno;
        if (!rc) return ETIMEDOUT;
        siginfo_t info = {0};
        rc = waitid((idtype_t)3 /* P_PIDFD */, (id_t)fd, &info, WEXITED | WNOHANG);
        if (rc < 0 && errno == EINTR) continue;
        if (rc < 0) return errno == ECHILD ? 0 : errno;
        if (info.si_pid) return 0;
    }
}

/* Used only by a bounded number of deferred waiters after the synchronous
 * deadline. This blocks in the kernel; it never polls or uses a numeric PID. */
void alloy_linux_reap_pidfd_blocking(int fd) {
    siginfo_t info = {0};
    int rc;
    do { rc = waitid((idtype_t)3 /* P_PIDFD */, (id_t)fd, &info, WEXITED); } while (rc < 0 && errno == EINTR);
}
