/* Native descendant cleanup, independent of sessions and process groups.
 *
 * A launchd-created resource coalition supplies membership. A revocable
 * Seatbelt extension gates new fork/spawn calls. The cleaner remains outside
 * that sandbox; two trusted sandboxed processes retain the inherited lease so
 * either can explicitly revoke it after the other's death. Merely closing a
 * process does NOT revoke an inherited extension.
 *
 * Private Darwin interfaces are detected at runtime. Missing APIs, unknown
 * membership, a truncated census, and incomplete teardown are errors.
 */
#include "macos_containment.h"
#include "owned_process.h"
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <libproc.h>
#include <sandbox.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/proc.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

typedef char *(*issue_fn)(const char *, uint32_t);
typedef int64_t (*consume_fn)(const char *);
typedef int (*release_fn)(int64_t);
typedef int (*usage_fn)(uint64_t, void *, size_t);
typedef struct {
    launch_state owner;
    _Atomic int setup, revoked, finish, revoker;
    _Atomic uint64_t revoker_identity;
} cohort_state;

int alloy_containment_gate;
static uint64_t now_ms(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t)) return 0;
    return (uint64_t)t.tv_sec * 1000 + (uint64_t)t.tv_nsec / 1000000;
}
static void pause_ms(void) {
    struct timespec t = {0, 1000000};
    nanosleep(&t, NULL);
}
static void erase(void *p, size_t size) {
    volatile unsigned char *bytes = p;
    while (size--) *bytes++ = 0;
}
static int coalition(pid_t pid, uint64_t *id) {
    uint64_t info[5] = {0};
    errno = 0;
    if (proc_pidinfo(pid, 20 /* PROC_PIDCOALITIONINFO */, 0, info, sizeof(info)) != sizeof(info))
        return errno ? errno : EIO;
    if (!info[0]) return EIO;
    *id = info[0];
    return 0;
}
static int count_tasks(usage_fn usage, uint64_t id, uint64_t *count) {
    uint64_t info[64] = {0};
    if (usage(id, info, sizeof(info))) return errno ? errno : EIO;
    if (info[1] > info[0]) return EIO;
    *count = info[0] - info[1];
    return 0;
}
static int stopped(launch_state *state) {
    return atomic_load_explicit(&state->state, memory_order_acquire) == STOP ||
        alloy_owned_alive(state->parent_pid, state->parent_identity) != 1;
}
static void request_stop(launch_state *state) {
    unsigned phase = atomic_load_explicit(&state->state, memory_order_acquire);
    while (phase != DONE && phase != STOP &&
        !atomic_compare_exchange_weak_explicit(&state->state, &phase, STOP,
            memory_order_acq_rel, memory_order_acquire)) {}
}
static int anonymous_state(size_t size, void **mapping) {
    char name[64];
    snprintf(name, sizeof(name), "/alloy-cohort-%016llx",
        (unsigned long long)(((uint64_t)arc4random() << 32) | arc4random()));
    int fd = shm_open(name, O_CREAT | O_EXCL | O_RDWR, 0600);
    if (fd < 0) return -1;
    shm_unlink(name);
    if (ftruncate(fd, (off_t)size)) { int e = errno; close(fd); errno = e; return -1; }
    *mapping = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (*mapping == MAP_FAILED) { int e = errno; close(fd); errno = e; return -1; }
    memset(*mapping, 0, size);
    return fd;
}
static int install_gate(issue_fn issue, consume_fn consume, int64_t *handle) {
    char nonce[128], policy[2048], registry[128];
    uint64_t random[2];
    arc4random_buf(random, sizeof(random));
    snprintf(nonce, sizeof(nonce), "com.layerdynamics.alloy.fork.%016llx%016llx",
        (unsigned long long)random[0], (unsigned long long)random[1]);
    char *token = issue(nonce, 0);
    if (!token) return errno ? errno : EIO;
    snprintf(registry,sizeof(registry),"/private/tmp/alloy-brokers-%u",getuid());
    snprintf(policy, sizeof(policy),
        "(version 1)(allow default)"
        "(deny process-fork (require-not (extension \"%s\")))"
        "(deny generic-issue-extension file-issue-extension mach-issue-extension (extension-class \"%s\"))"
        "(deny signal (require-not (require-any (target self)(target children))))"
        "(deny mach-task* (require-not (target self)))"
        "(deny job-creation appleevent-send)"
        "(deny file-link file-read* file-write* (subpath \"%s\"))", nonce, nonce, registry);
    char *diagnostic = NULL;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    int result = sandbox_init(policy, 0, &diagnostic);
    if (diagnostic) sandbox_free_error(diagnostic);
#pragma clang diagnostic pop
    int error = result ? (errno ? errno : EIO) : 0;
    if (!error && (*handle = consume(token)) < 0) error = errno ? errno : EIO;
    /* No token, class string or policy buffer reaches a workload fork/exec. */
    erase(token, strlen(token)); free(token);
    erase(nonce, sizeof(nonce)); erase(policy, sizeof(policy)); erase(random, sizeof(random));
    return error;
}
static void revoke_value(_Atomic int *field, int value) {
    int old = atomic_load_explicit(field, memory_order_acquire);
    /* Successful release is monotonic: another holder may concurrently learn
     * that this same inherited handle was already released. */
    while (old != 1 && !atomic_compare_exchange_weak_explicit(field, &old, value,
        memory_order_acq_rel, memory_order_acquire)) {}
}
static void publish_revoke_value(cohort_state *shared, launch_state *outer, alloy_cohort_record *record, int value) {
    revoke_value(&shared->revoked, value);
    value = atomic_load_explicit(&shared->revoked, memory_order_acquire);
    revoke_value(&outer->fork_revoked, value);
    if (record) {
        revoke_value(&record->revoked, value);
        (void)msync(record, sizeof(*record), MS_SYNC);
    }
}
static void publish_revoke(cohort_state *shared, launch_state *outer, alloy_cohort_record *record, release_fn release, int64_t handle) {
    int result = release(handle);
    publish_revoke_value(shared, outer, record, result ? -(errno ? errno : EIO) : 1);
}
static void run_gate(cohort_state *shared, launch_state *outer, alloy_cohort_record *record, int statefd, int request,
                      int argc, char **arguments, const int io[3],
                      issue_fn issue, consume_fn consume, release_fn release) {
#ifdef ALLOY_OWNED_PROCESS_TEST
    if(atomic_load(&outer->test_setup_pause)){
        atomic_store(&outer->test_setup_paused,1);
        uint64_t barrier=now_ms()+5000;
        while(atomic_load(&outer->test_setup_pause) && now_ms()<barrier &&
            alloy_owned_alive(shared->owner.parent_pid,shared->owner.parent_identity)==1)pause_ms();
    }
#endif
    /* launchd tears down its dead job leader's process group. The independent
     * revocation authority must survive that cleanup as well as broker loss. */
    if (setsid() < 0) { shared->setup = -errno; publish_revoke_value(shared,outer,record,1); _exit(125); }
    int64_t handle = -1;
    int error = install_gate(issue, consume, &handle);
    if (error) { shared->setup = -error; publish_revoke_value(shared,outer,record,1); _exit(125); }
    pid_t revoker = fork();
    if (revoker < 0) { shared->setup = -errno; publish_revoke(shared, outer, record, release, handle); _exit(125); }
    if (!revoker) {
        close(statefd);
        /* This process owns neither workload I/O nor the completion descriptor. */
        for (int i = 0; i < 3; ++i) if (io[i] > 2) close(io[i]);
        char byte;
        ssize_t n;
        do { n = read(request, &byte, 1); } while (n < 0 && errno == EINTR);
        close(request);
        publish_revoke(shared, outer, record, release, handle);
        while (!atomic_load_explicit(&shared->finish, memory_order_acquire) &&
               alloy_owned_alive(shared->owner.parent_pid, shared->owner.parent_identity) == 1)
            pause_ms();
        _exit(0);
    }
    close(request);
    shared->revoker = revoker;
    shared->revoker_identity = alloy_process_identity(revoker);
    if (!shared->revoker_identity) { publish_revoke(shared, outer, record, release, handle); shared->setup = -EIO; _exit(125); }
    /* Duplicate all inputs before replacing any low descriptor. */
    int copies[4] = {-1,-1,-1,-1};
    for (int i = 0; i < 4; ++i) {
        copies[i] = fcntl(i == 3 ? statefd : io[i], F_DUPFD_CLOEXEC, 10);
        if (copies[i] < 0) { shared->setup = -errno; publish_revoke(shared, outer, record, release, handle); _exit(125); }
    }
    for (int i = 0; i < 4; ++i) { if (dup2(copies[i], i) < 0) _exit(125); close(copies[i]); }
    shared->setup = 1;
    alloy_containment_gate = 1;
    (void)alloy_exec_helper(argc, arguments);
    /* Revoker death must not strand the lease. The other holder can release
     * the same inherited handle after the target owner finishes. */
    while (!atomic_load_explicit(&shared->revoked, memory_order_acquire)) {
        if (alloy_owned_alive(revoker, shared->revoker_identity) == 0 ||
            alloy_owned_alive(shared->owner.parent_pid, shared->owner.parent_identity) != 1) {
            publish_revoke(shared, outer, record, release, handle);
            break;
        }
        pause_ms();
    }
    publish_revoke_value(shared, outer, record,
        atomic_load_explicit(&shared->revoked, memory_order_acquire));
    while (!atomic_load_explicit(&shared->finish, memory_order_acquire) &&
           alloy_owned_alive(shared->owner.parent_pid, shared->owner.parent_identity) == 1)
        pause_ms();
    int status;
    while (waitpid(revoker, &status, WNOHANG) < 0 && errno == EINTR) {}
    _exit(0);
}
static int sweep(uint64_t id, pid_t gate, pid_t revoker) {
    /* libproc returns a PID count here, unlike proc_pidinfo's byte count. */
    pid_t members[16384];
    errno = 0;
    int count = proc_listallpids(members, sizeof(members));
    if (count <= 0) return errno ? errno : EIO;
    if (count >= (int)(sizeof(members) / sizeof(members[0]))) return E2BIG;
    for (int i = 0; i < count; ++i) {
        pid_t pid = members[i];
        if (pid <= 0 || pid == getpid() || pid == gate || pid == revoker) continue;
        uint64_t identity = alloy_process_identity(pid), member = 0;
        if (!identity) continue;
        int error = coalition(pid, &member);
        if (error == ESRCH) continue;
        if (error) return error;
        if (member != id || identity != alloy_process_identity(pid)) continue;
        error = alloy_owned_signal(pid, identity);
        if (error && error != ESRCH) return error;
    }
    return 0;
}

int alloy_containment_run(launch_state *state, int argc, char **arguments, const int io[3], alloy_cohort_record *record) {
    uint64_t id = 0, count = 0;
    int error = coalition(getpid(), &id);
    void *library = dlopen("/usr/lib/system/libsystem_sandbox.dylib", RTLD_LAZY | RTLD_LOCAL);
    void *system = dlopen("/usr/lib/libSystem.B.dylib", RTLD_LAZY | RTLD_LOCAL);
    issue_fn issue = library ? (issue_fn)dlsym(library, "sandbox_extension_issue_generic") : NULL;
    consume_fn consume = library ? (consume_fn)dlsym(library, "sandbox_extension_consume") : NULL;
    release_fn release = library ? (release_fn)dlsym(library, "sandbox_extension_release") : NULL;
    usage_fn usage = system ? (usage_fn)dlsym(system, "coalition_info_resource_usage") : NULL;
    if (!error && (!issue || !consume || !release || !usage)) error = ENOTSUP;
    if (!error) error = count_tasks(usage, id, &count);
    if (!error && count != 1) error = EBUSY;
    if (error) return error;
    state->cohort_identity = id;

    cohort_state *shared = MAP_FAILED;
    int fd = anonymous_state(sizeof(*shared), (void **)&shared);
    if (fd < 0) return errno;
    int request[2];
    if (pipe(request)) { error = errno; close(fd); munmap(shared, sizeof(*shared)); return error; }
    shared->owner.parent_pid = getpid();
    shared->owner.parent_identity = alloy_process_identity(getpid());
    pid_t gate = fork();
    if (!gate) {
        close(request[1]);
        run_gate(shared, state, record, fd, request[0], argc, arguments, io, issue, consume, release);
    }
    close(request[0]); close(fd);
    if (gate < 0) { error = errno; close(request[1]); munmap(shared, sizeof(*shared)); return error; }
    uint64_t gate_identity = alloy_process_identity(gate), until = now_ms() + 5000;
    state->gate_pid = gate; state->gate_identity = gate_identity;
    state->pid = getpid(); state->identity = shared->owner.parent_identity;
    unsigned expected = INITIAL;
    atomic_compare_exchange_strong_explicit(&state->state, &expected, READY,
        memory_order_release, memory_order_relaxed);
    for (;;) {
        unsigned phase = atomic_load_explicit(&state->state, memory_order_acquire);
        unsigned inner = atomic_load_explicit(&shared->owner.state, memory_order_acquire);
        if (shared->setup < 0) { error = -shared->setup; break; }
        if (shared->setup == 1) {
            state->revoker_pid = shared->revoker;
            state->revoker_identity = shared->revoker_identity;
        }
        if (stopped(state)) break;
        if (phase == CREATE && inner == READY) {
            expected = READY;
            atomic_compare_exchange_strong(&shared->owner.state, &expected, CREATE);
        } else if (phase == CREATE && inner == TARGET_READY) {
            state->target_pid = shared->owner.target_pid;
            state->target_identity = shared->owner.target_identity;
            expected = CREATE;
            atomic_compare_exchange_strong_explicit(&state->state, &expected, TARGET_READY,
                memory_order_release, memory_order_relaxed);
        } else if (phase == RUN && inner == TARGET_READY) {
            expected = TARGET_READY;
            atomic_compare_exchange_strong(&shared->owner.state, &expected, RUN);
        }
        if (inner == DONE) { error = shared->owner.error; break; }
        if (!gate_identity || alloy_owned_alive(gate, gate_identity) == 0) { error = EIO; break; }
        if (phase < RUN && now_ms() >= until) { error = ETIMEDOUT; break; }
        pause_ms();
    }
    /* EOF is the revocation request. It cannot target a reused PID. */
    close(request[1]);
    until = now_ms() + 5000;
    while (!atomic_load_explicit(&shared->revoked, memory_order_acquire)) {
        if (shared->revoker_identity && alloy_owned_alive(shared->revoker,shared->revoker_identity)==0)
            request_stop(&shared->owner);
        if (now_ms() >= until) { error = error ? error : ETIMEDOUT; break; }
        pause_ms();
    }
    int revoked = atomic_load_explicit(&shared->revoked, memory_order_acquire);
#ifdef ALLOY_OWNED_PROCESS_TEST
    if (revoked == 1 && atomic_load(&state->test_revoke_pause)) {
        atomic_store(&state->test_revoke_paused, 1);
        uint64_t barrier = now_ms() + 2000;
        while (atomic_load(&state->test_revoke_pause) && now_ms() < barrier &&
            alloy_owned_alive(state->parent_pid,state->parent_identity)==1) pause_ms();
    }
#endif
    request_stop(&shared->owner);
    if (revoked < 0 && !error) error = -revoked;
    int cleanup_error = revoked == 1 ? 0 : EIO;
    if (!cleanup_error) for (;;) {
        int live_gate = alloy_owned_alive(gate, gate_identity);
        int live_revoker = shared->revoker_identity ?
            alloy_owned_alive(shared->revoker, shared->revoker_identity) : 0;
        if (live_gate < 0 || live_revoker < 0) { cleanup_error = EIO; break; }
        cleanup_error = sweep(id, gate, shared->revoker);
        if (cleanup_error) break;
        cleanup_error = count_tasks(usage, id, &count);
        if (cleanup_error) break;
        if (count == (uint64_t)(1 + live_gate + live_revoker)) break;
        if (now_ms() >= until) { cleanup_error = ETIMEDOUT; break; }
        pause_ms();
    }
    atomic_store_explicit(&shared->finish, 1, memory_order_release);
    int status;
    until = now_ms() + 2000;
    for (;;) {
        pid_t reaped = waitpid(gate, &status, WNOHANG);
        if (reaped == gate || (reaped < 0 && errno == ECHILD)) break;
        if (reaped < 0 && errno != EINTR) { cleanup_error = cleanup_error ? cleanup_error : errno; break; }
        if (now_ms() >= until) { cleanup_error = cleanup_error ? cleanup_error : ETIMEDOUT; break; }
        pause_ms();
    }
    if (!cleanup_error) {
        until = now_ms() + 2000;
        do {
            cleanup_error = count_tasks(usage, id, &count);
            if (cleanup_error || count == 1) break;
            pause_ms();
        } while (now_ms() < until);
        if (!cleanup_error && count != 1) cleanup_error = ETIMEDOUT;
    }
    state->status = shared->owner.status;
    munmap(shared, sizeof(*shared));
    if (!error) error = cleanup_error;
    if (!cleanup_error) {
        if(record){atomic_store_explicit(&record->complete,1,memory_order_release);(void)msync(record,sizeof(*record),MS_SYNC);}
        state->error = error;
        atomic_store_explicit(&state->state, DONE, memory_order_release);
    }
    return error;
}

/* The embedding host can finish cleanup after loss of the isolated cleaner.
 * It uses no waitpid on that process, and never signals a numeric PID without
 * an audit-token identity check. The inherited revoker acknowledges release
 * through this mapping after the cleaner's request pipe closes. */
int alloy_containment_recover(launch_state *state) {
    if (atomic_load_explicit(&state->state, memory_order_acquire) == DONE) return 0;
    uint64_t id = state->cohort_identity, own = 0;
    if (!id || coalition(getpid(), &own) || own == id) return EINVAL;
    void *system = dlopen("/usr/lib/libSystem.B.dylib", RTLD_LAZY | RTLD_LOCAL);
    usage_fn usage = system ? (usage_fn)dlsym(system, "coalition_info_resource_usage") : NULL;
    if (!usage) return ENOTSUP;
    uint64_t until = now_ms() + 5000, count = 0;
    int error = 0;
    for (;;) {
        if (atomic_load_explicit(&state->state, memory_order_acquire) == DONE) {
            dlclose(system);
            return 0; /* Preserve a real completion racing the recovery handoff. */
        }
        error = count_tasks(usage, id, &count);
        if (error == ESRCH) { error = 0; break; }
        if (error) break;
        if (!count) break;
        int revoked = atomic_load_explicit(&state->fork_revoked, memory_order_acquire);
        if (revoked < 0) { error = -revoked; break; }
        if (revoked == 1) {
#ifdef ALLOY_OWNED_PROCESS_TEST
            if(atomic_exchange(&state->test_recovery_fail,0)){error=EIO;break;}
#endif
            error = sweep(id, 0, 0);
            if (error) break;
        }
        if (now_ms() >= until) { error = ETIMEDOUT; break; }
        pause_ms();
    }
    if (!error) {
        if(state->cohort_record_path[0])error=alloy_cohort_record_complete(state->cohort_record_path,id);
    }
    if (!error) {
        state->error = EIO; /* Loss interrupted the command; never invent success. */
        atomic_store_explicit(&state->state, DONE, memory_order_release);
    }
    dlclose(system);
    return error;
}

int alloy_containment_current_coalition(uint64_t *id) { return coalition(getpid(),id); }
int alloy_containment_drain(uint64_t id) {
    uint64_t own=0,count=0;
    if(!id || coalition(getpid(),&own) || own==id)return EINVAL;
    void *system=dlopen("/usr/lib/libSystem.B.dylib",RTLD_LAZY|RTLD_LOCAL);
    usage_fn usage=system?(usage_fn)dlsym(system,"coalition_info_resource_usage"):NULL;
    if(!usage)return ENOTSUP;
    uint64_t until=now_ms()+5000;int error=0;
    for(;;){
        error=count_tasks(usage,id,&count);
        /* Coalition IDs are monotonically allocated within a boot. The record
         * reader validates kern.bootsessionuuid before calling this function. */
        if(error==ESRCH){error=0;break;}
        if(error || !count)break;
        error=sweep(id,0,0);if(error)break;
        if(now_ms()>=until){error=ETIMEDOUT;break;}pause_ms();
    }
    dlclose(system);return error;
}
int alloy_containment_empty(uint64_t id) {
    if(!id)return -EINVAL;
    void *system=dlopen("/usr/lib/libSystem.B.dylib",RTLD_LAZY|RTLD_LOCAL);
    usage_fn usage=system?(usage_fn)dlsym(system,"coalition_info_resource_usage"):NULL;
    if(!usage){if(system)dlclose(system);return -ENOTSUP;}
    uint64_t count=0;int error=count_tasks(usage,id,&count);dlclose(system);
    if(error==ESRCH)return 1;
    return error?-error:count==0;
}
