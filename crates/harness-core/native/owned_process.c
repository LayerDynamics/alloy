/* macOS child ownership: an isolated supervisor exclusively reaps its target.
 * Only that supervisor signals numeric target/group IDs while the child pins
 * them. Daemon fallback signals use kernel-validated stable process identity.
 * See Apple's xnu bsd/sys/proc_info_private.h and bsd/kern/proc_info.c. */
#include "owned_process.h"
#include <errno.h>
#include <fcntl.h>
#include <libproc.h>
#include <mach/message.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/proc.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

/* Stable Darwin libproc ABI, not exported by the public SDK header. */
struct alloy_unique_info {
    uint8_t uuid[16];
    uint64_t uniqueid, parent_uniqueid;
    int32_t idversion, original_parent_version;
    uint64_t reserved[2];
};
_Static_assert(sizeof(struct alloy_unique_info) == 56, "Darwin unique identity ABI");
_Static_assert(ATOMIC_LLONG_LOCK_FREE == 2, "identity atomics must be lock-free");
_Static_assert(ATOMIC_INT_LOCK_FREE == 2, "shared state must be lock-free");
enum { INITIAL, READY, CREATE, TARGET_READY, RUN, STOP, DONE };
typedef struct {
    _Atomic unsigned state;
    _Atomic int pid, target_pid, parent_pid, status, error;
    _Atomic uint64_t identity, target_identity, parent_identity;
} launch_state;
typedef struct {
    launch_state *state;
    int owner_pid, reap_owner;
    uint64_t owner_identity;
} owned_control;
static int identity_info(int pid, struct alloy_unique_info *info) {
    int size = proc_pidinfo(pid, 17 /* PROC_PIDUNIQIDENTIFIERINFO */, 0, info, sizeof(*info));
    return size == sizeof(*info) ? 0 : (errno ? errno : EIO);
}
int alloy_owned_alive(int pid, uint64_t identity) {
    struct alloy_unique_info info;
    int error = identity_info(pid, &info);
    if (error == ESRCH || (!error && info.uniqueid != identity)) return 0;
    return error ? -error : 1;
}
static int identity_signal(int pid, uint64_t identity, int signal) {
    if (pid <= 0 || identity == 0) return EINVAL;
    for (int attempt = 0; attempt < 1024; ++attempt) {
        struct alloy_unique_info info;
        int error = identity_info(pid, &info);
        if (error == ESRCH || (!error && info.uniqueid != identity)) return 0;
        if (error) return error;
        audit_token_t token = {{0}};
        token.val[5] = (unsigned)pid;
        token.val[7] = (unsigned)info.idversion;
        /* libproc returns an errno value, not -1/errno. The kernel compares
         * PID+idversion while holding a process reference before signaling. */
        error = proc_signal_with_audittoken(&token, signal);
        if (!error) return 0;
        if (error != ESRCH) return error;
        /* exec may invalidate idversion; re-read only while uniqueid matches. */
    }
    return EBUSY;
}
int alloy_owned_signal(int pid, uint64_t identity) {
    return identity_signal(pid, identity, SIGKILL);
}
static uint64_t milliseconds(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t)) return 0;
    return (uint64_t)t.tv_sec * 1000 + (uint64_t)t.tv_nsec / 1000000;
}
static void pause_briefly(void) {
    struct timespec t = {0, 1000000};
    nanosleep(&t, NULL);
}

/* Called while the exclusive owner still holds the group leader unreaped.
 * libproc includes zombies; they cannot execute or retain pipe endpoints. */
static int group_live(int group) {
    pid_t members[4096];
    errno = 0;
    int count = proc_listpgrppids(group, members, sizeof(members));
    if (count < 0 || (!count && errno)) return -(errno ? errno : EIO);
    if (count >= (int)(sizeof(members) / sizeof(members[0]))) return -E2BIG;
    for (int i = 0; i < count; ++i) {
        struct proc_bsdinfo info;
        errno = 0;
        int size = proc_pidinfo(members[i], PROC_PIDTBSDINFO, 0, &info, sizeof(info));
        if (size != sizeof(info)) {
            if (errno == ESRCH) continue;
            return -(errno ? errno : EIO);
        }
        if (info.pbi_pgid == (uint32_t)group && info.pbi_status != SZOMB) return 1;
    }
    return 0;
}
#ifdef ALLOY_OWNED_PROCESS_TEST
int alloy_test_group_live(int group) { return group_live(group); }
#endif
static int cleanup_group(pid_t target) {
    uint64_t until = milliseconds() + 2000;
    for (;;) {
        int group_error = kill(-target, SIGKILL) ? errno : 0;
        int child_error = kill(target, SIGKILL) ? errno : 0;
        int live = group_live(target);
        if (live == 0) return child_error == ESRCH ? 0 : child_error;
        if (live < 0) return -live;
        if (group_error && group_error != ESRCH) return group_error;
        if (child_error && child_error != ESRCH) return child_error;
        if (milliseconds() >= until) return ETIMEDOUT;
        pause_briefly();
    }
}

/* Only this single-threaded process reaps the target. No configured code runs
 * in the supervisor. It retains an exited child until group cleanup is done. */
int alloy_exec_helper(int argc, char **argv) {
    if (argc < 3 || strcmp(argv[1], "--alloy-owned-exec-v1")) return 125;
    struct stat st;
    if (fstat(3, &st) || st.st_size < (off_t)sizeof(launch_state)) return 125;
    launch_state *state = mmap(NULL, sizeof(*state), PROT_READ | PROT_WRITE, MAP_SHARED, 3, 0);
    if (state == MAP_FAILED) return 125;
    close(3);
    struct alloy_unique_info info;
    if (identity_info(getpid(), &info)) return 125;
    state->pid = getpid();
    state->identity = info.uniqueid;
    unsigned expected = INITIAL;
    if (!atomic_compare_exchange_strong_explicit(&state->state, &expected, READY,
            memory_order_release, memory_order_relaxed)) return 125;
    uint64_t until = milliseconds() + 5000;
    unsigned phase;
    while ((phase = atomic_load_explicit(&state->state, memory_order_acquire)) == READY) {
        if (milliseconds() >= until) return 125;
        pause_briefly();
    }
    if (phase != CREATE) return 125;
    struct sigaction action = {0};
    action.sa_handler = SIG_DFL;
    sigemptyset(&action.sa_mask);
    if (sigaction(SIGCHLD, &action, NULL)) return 125;
    posix_spawnattr_t attr;
    posix_spawn_file_actions_t actions;
    if (posix_spawnattr_init(&attr) || posix_spawn_file_actions_init(&actions)) return 125;
    int error = posix_spawnattr_setflags(&attr, POSIX_SPAWN_START_SUSPENDED |
                                       POSIX_SPAWN_CLOEXEC_DEFAULT | POSIX_SPAWN_SETSID);
    for (int fd = 0; !error && fd < 3; ++fd)
        error = posix_spawn_file_actions_addinherit_np(&actions, fd);
    char *environment[] = {NULL};
    pid_t target = 0;
    if (!error) error = posix_spawn(&target, argv[2], &actions, &attr, &argv[2], environment);
    posix_spawn_file_actions_destroy(&actions);
    posix_spawnattr_destroy(&attr);
    if (error) {
        state->error = error;
        atomic_store_explicit(&state->state, DONE, memory_order_release);
        return 0;
    }
    /* A killed suspended child remains an unreaped zombie here: this process
     * alone owns its wait status, so its PID/group cannot be recycled. */
    error = identity_info(target, &info);
    if (!error) {
        state->target_pid = target;
        state->target_identity = info.uniqueid;
        expected = CREATE;
        if (atomic_compare_exchange_strong_explicit(&state->state, &expected, TARGET_READY,
                memory_order_release, memory_order_relaxed)) {
            until = milliseconds() + 5000;
            while ((phase = atomic_load_explicit(&state->state, memory_order_acquire)) == TARGET_READY) {
                if (milliseconds() >= until || alloy_owned_alive(state->parent_pid, state->parent_identity) != 1) {
                    expected = TARGET_READY;
                    atomic_compare_exchange_strong(&state->state, &expected, STOP);
                    phase = atomic_load_explicit(&state->state, memory_order_acquire);
                    break;
                }
                pause_briefly();
            }
            if (phase == RUN && kill(target, SIGCONT)) error = errno;
        }
    }
    for (;;) {
        if (error || atomic_load_explicit(&state->state, memory_order_acquire) != RUN ||
            alloy_owned_alive(state->parent_pid, state->parent_identity) != 1) break;
        siginfo_t exit_info = {0};
        if (waitid(P_PID, (id_t)target, &exit_info, WEXITED | WNOHANG | WNOWAIT)) {
            if (errno == EINTR) continue;
            error = errno; break;
        }
        if (exit_info.si_pid == target) break;
        pause_briefly();
    }
    /* Exclusive reaping is structural, not an assumption about the daemon.
     * Even an exited target still pins this PID/group until the wait below. */
    int cleanup_error = cleanup_group(target);
    if (!error) error = cleanup_error;
    int status = 0;
    until = milliseconds() + 2000;
    for (;;) {
        pid_t waited = waitpid(target, &status, WNOHANG);
        if (waited == target) break;
        if (waited < 0 && errno != EINTR) { if (!error) error = errno; break; }
        if (milliseconds() >= until) { if (!error) error = ETIMEDOUT; break; }
        pause_briefly();
    }
    state->status = status;
    state->error = error;
    atomic_store_explicit(&state->state, DONE, memory_order_release);
    munmap(state, sizeof(*state));
    return 0;
}

/* Request cleanup without using a PID. Shared state survives parent unmapping. */
int alloy_owned_stop(void *opaque) {
    owned_control *control = opaque;
    unsigned phase = atomic_load_explicit(&control->state->state, memory_order_acquire);
    while (phase != DONE && phase != STOP &&
           !atomic_compare_exchange_weak_explicit(&control->state->state, &phase, STOP,
                  memory_order_acq_rel, memory_order_acquire)) {}
    return 0;
}
int alloy_owned_wait(void *opaque, int *status, int nohang) {
    owned_control *control = opaque;
    uint64_t until = milliseconds() + 2000;
    for (;;) {
        if (atomic_load_explicit(&control->state->state, memory_order_acquire) == DONE) {
            *status = control->state->status;
            return control->state->error;
        }
        if (alloy_owned_alive(control->owner_pid, control->owner_identity) == 0) {
            /* Supervisor death cannot authorize a raw PID/group kill. */
            if (control->state->target_identity)
                (void)alloy_owned_signal(control->state->target_pid, control->state->target_identity);
            return EIO;
        }
        if (nohang) return EAGAIN;
        if (milliseconds() >= until) return ETIMEDOUT;
        pause_briefly();
    }
}
void alloy_owned_close(void *opaque) {
    if (!opaque) return;
    owned_control *control = opaque;
    alloy_owned_stop(control);
    int status;
    /* A delayed owner must retain its pinned leader and finish group cleanup.
     * Killing it on a timeout would abandon descendants. wait reports failure
     * to the caller; STOP remains set even after this mapping is released. */
    (void)alloy_owned_wait(control, &status, 0);
    /* External-reaper mode never waits by PID. In host-owned mode the
     * embedding process promises exclusive supervisor reaping. */
    for (int attempt = 0; control->reap_owner && attempt < 100; ++attempt) {
        pid_t result = waitpid(control->owner_pid, &status, WNOHANG);
        if (result == control->owner_pid || (result < 0 && errno != EINTR)) break;
        pause_briefly();
    }
    munmap(control->state, sizeof(*control->state));
    free(control);
}

int alloy_owned_spawn(const char *launcher, const char *program,
    posix_spawn_file_actions_t *actions, posix_spawnattr_t *attr,
    char *const argv[], char *const envp[], alloy_stop_callback stop, void *context, int reap_owner,
    int *pid, uint64_t *identity, void **out_control) {
    *pid = 0; *identity = 0; *out_control = NULL;
    if (stop(context)) return ECANCELED;
    char name[32];
    snprintf(name, sizeof(name), "/alloy-%016llx", (unsigned long long)(((uint64_t)arc4random() << 32) | arc4random()));
    /* Darwin shm_open atomically sets FD_CLOEXEC; O_CLOEXEC is rejected. */
    int fd = shm_open(name, O_CREAT | O_EXCL | O_RDWR, 0600);
    if (fd < 0) return errno;
    shm_unlink(name);
    int result = EIO;
    launch_state *state = MAP_FAILED;
    char **arguments = NULL;
    owned_control *control = NULL;
    if (fd < 4) {
        int copy = fcntl(fd, F_DUPFD_CLOEXEC, 4);
        int error = errno;
        close(fd); fd = copy;
        if (fd < 0) return error;
    }
    if (ftruncate(fd, sizeof(launch_state))) { result = errno; goto done; }
    state = mmap(NULL, sizeof(*state), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (state == MAP_FAILED) { result = errno; goto done; }
    memset(state, 0, sizeof(*state));
    atomic_init(&state->state, INITIAL);
    struct alloy_unique_info parent;
    result = identity_info(getpid(), &parent);
    if (result) goto done;
    state->parent_pid = getpid();
    state->parent_identity = parent.uniqueid;
    size_t count = 0;
    while (argv[count]) ++count;
    arguments = calloc(count + 3, sizeof(char *));
    if (!arguments) { result = ENOMEM; goto done; }
    arguments[0] = (char *)launcher;
    arguments[1] = "--alloy-owned-exec-v1";
    arguments[2] = (char *)program;
    for (size_t i = 1; i < count; ++i) arguments[i + 2] = argv[i];
    result = posix_spawn_file_actions_adddup2(actions, fd, 3);
    if (result) goto done;
    if (fd != 3 && (result = posix_spawn_file_actions_addclose(actions, fd))) goto done;
    control = calloc(1, sizeof(*control));
    if (!control) { result = ENOMEM; goto done; }
    result = posix_spawn(pid, launcher, actions, attr, arguments, envp);
    if (result) { *pid = 0; goto done; }
    int owner_pid = *pid;
    control->state = state;
    control->owner_pid = owner_pid;
    control->reap_owner = reap_owner;
    uint64_t until = milliseconds() + 5000;
    for (;;) {
        unsigned phase = atomic_load_explicit(&state->state, memory_order_acquire);
        if (phase == READY) {
            if (state->pid != owner_pid || !state->identity) { result = EIO; break; }
            control->owner_identity = state->identity;
            if (stop(context)) { result = ECANCELED; break; }
            unsigned expected = READY;
            atomic_compare_exchange_strong(&state->state, &expected, CREATE);
        } else if (phase == TARGET_READY) {
            if (stop(context)) { result = ECANCELED; break; }
            *pid = state->target_pid;
            *identity = state->target_identity;
            unsigned expected = TARGET_READY;
            if (!atomic_compare_exchange_strong(&state->state, &expected, RUN)) continue;
            *out_control = control;
            control = NULL;
            state = MAP_FAILED;
            result = 0;
            break;
        } else if (phase == DONE) {
            result = state->error ? state->error : EIO;
            break;
        }
        if (stop(context) || milliseconds() >= until) { result = ECANCELED; break; }
        pause_briefly();
    }
    if (result) {
        unsigned phase = atomic_load_explicit(&state->state, memory_order_acquire);
        if (phase >= READY && state->identity) control->owner_identity = state->identity;
        alloy_owned_close(control);
        control = NULL;
        state = MAP_FAILED;
        *pid = 0; *identity = 0;
    }

done:
    free(control);
    free(arguments);
    if (state != MAP_FAILED) munmap(state, sizeof(*state));
    close(fd);
    return result;
}
