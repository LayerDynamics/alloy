/* macOS child ownership: an isolated supervisor exclusively reaps its target.
 * Only that supervisor signals numeric target/group IDs while the child pins
 * them. Daemon fallback signals use kernel-validated stable process identity.
 * See Apple's xnu bsd/sys/proc_info_private.h and bsd/kern/proc_info.c. */
#include "owned_process.h"
#include "macos_containment.h"
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
#include <pthread.h>
#include <poll.h>
#include <sys/socket.h>
#include <mach/mach.h>
#include <mach/task_special_ports.h>
#include <sys/fileport.h>
#include <servers/bootstrap.h>
#include <bsm/libbsm.h>
#include <Security/Security.h>
#include <ServiceManagement/ServiceManagement.h>
#include <dlfcn.h>

static int shell_broker(int argc, char **argv);
static int cohort_service(int argc, char **argv);
static int verification_service(int argc,char **argv);

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
typedef struct {
    launch_state *state;
    int owner_pid, reap_owner;
    uint64_t owner_identity;
    int shell_slot;
} owned_control;
static int reserve_shell_slot(launch_state *state);
static void retire_shell_slot(int slot);
#ifdef ALLOY_OWNED_PROCESS_TEST

static _Atomic int test_wait_for_owner_exit;
static _Atomic int test_unknown_identity_pid;
static _Atomic int test_watcher_start_failure;
void alloy_test_watcher_start_failure(void){atomic_store(&test_watcher_start_failure,1);}
void alloy_test_identity_unknown(int pid) {atomic_store(&test_unknown_identity_pid,pid);}
void alloy_test_wait_owner_exit(void) {atomic_store(&test_wait_for_owner_exit,1);}
static _Thread_local const char *test_shell_stage="before native shell spawn";
static _Thread_local char test_shell_diagnostic[256];
const char *alloy_test_shell_diagnostic(void) {
    return test_shell_diagnostic[0]?test_shell_diagnostic:test_shell_stage;
}
#define SHELL_STAGE(text) (test_shell_stage=(text))
#else
#define SHELL_STAGE(text) ((void)0)
#endif
static int identity_info(int pid, struct alloy_unique_info *info) {
    errno = 0;
    int size = proc_pidinfo(pid, 17 /* PROC_PIDUNIQIDENTIFIERINFO */, 0, info, sizeof(*info));
    return size == sizeof(*info) ? 0 : (errno ? errno : EIO);
}
uint64_t alloy_process_identity(int pid) {
    struct alloy_unique_info info;
    return identity_info(pid, &info) ? 0 : info.uniqueid;
}
int alloy_broker_registered(const char *name) {
    /* Looking up a Mach endpoint may activate a retiring service, then lose
     * its receive right during the lookup. Inspect launchd's job metadata
     * instead: inactive jobs are still registrations and must be removed. */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    void *library=dlopen("/System/Library/Frameworks/ServiceManagement.framework/ServiceManagement",RTLD_LAZY|RTLD_LOCAL);
    if(!library)return -ENOTSUP;
    __typeof__(&SMJobCopyDictionary) copy=(void *)dlsym(library,"SMJobCopyDictionary");
    __typeof__(&CFStringCreateWithCString) string=(void *)dlsym(library,"CFStringCreateWithCString");
    __typeof__(&CFRelease) release=(void *)dlsym(library,"CFRelease");
    const CFStringRef *domain=dlsym(library,"kSMDomainUserLaunchd");
    if(!copy || !string || !release || !domain){dlclose(library);return -ENOTSUP;}
    CFStringRef label=string(NULL,name,kCFStringEncodingUTF8);
    if(!label){dlclose(library);return -ENOMEM;}
    CFDictionaryRef job=copy(*domain,label);
    int present=job!=NULL;
    if(job)release(job);release(label);dlclose(library);return present;
#pragma clang diagnostic pop
}
/* Public ServiceManagement compatibility API for unbundled CLI/embedding hosts.
 * Unlike invoking launchctl, these calls create no child of the embedding host.
 * Keep framework initialization out of private-bootstrap pipe helpers. */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
int alloy_launchd_job(const char *name,const void *plist,size_t size,char *diagnostic,size_t capacity) {
    if(!name || !diagnostic || capacity<2 || size>65536)return EINVAL;
    diagnostic[0]=0;
    static void *library;
    if(!library)library=dlopen("/System/Library/Frameworks/ServiceManagement.framework/ServiceManagement",RTLD_LAZY|RTLD_LOCAL);
    if(!library){snprintf(diagnostic,capacity,"ServiceManagement is unavailable");return ENOTSUP;}
#define LOAD_SERVICE(name) __typeof__(&name) p_##name=(void *)dlsym(library,#name);if(!p_##name){snprintf(diagnostic,capacity,"ServiceManagement symbol unavailable: %s",#name);return ENOTSUP;}
    LOAD_SERVICE(CFDataCreate);
    LOAD_SERVICE(CFPropertyListCreateWithData);
    LOAD_SERVICE(CFDictionaryGetTypeID);
    LOAD_SERVICE(CFGetTypeID);
    LOAD_SERVICE(CFStringCreateWithCString);
    LOAD_SERVICE(CFErrorCopyDescription);
    LOAD_SERVICE(CFStringGetCString);
    LOAD_SERVICE(CFRelease);
    LOAD_SERVICE(SMJobSubmit);
    LOAD_SERVICE(SMJobRemove);
#undef LOAD_SERVICE
    const CFStringRef *domain=dlsym(library,"kSMDomainUserLaunchd");
    if(!domain){snprintf(diagnostic,capacity,"per-user ServiceManagement domain unavailable");return ENOTSUP;}
    CFDataRef data=NULL;CFPropertyListRef definition=NULL;CFStringRef label=NULL;CFErrorRef error=NULL;
    Boolean success=false;
    if(plist){
        data=p_CFDataCreate(NULL,plist,(CFIndex)size);
        if(data)definition=p_CFPropertyListCreateWithData(NULL,data,0,NULL,&error);
        if(definition && p_CFGetTypeID(definition)==p_CFDictionaryGetTypeID())
            success=p_SMJobSubmit(*domain,(CFDictionaryRef)definition,NULL,&error);
    }else{
        label=p_CFStringCreateWithCString(NULL,name,kCFStringEncodingUTF8);
        if(label)success=p_SMJobRemove(*domain,label,NULL,false,&error);
    }
    if(!success){
        CFStringRef description=error?p_CFErrorCopyDescription(error):NULL;
        if(!description || !p_CFStringGetCString(description,diagnostic,(CFIndex)capacity,kCFStringEncodingUTF8))
            snprintf(diagnostic,capacity,"ServiceManagement %s failed",plist?"submission":"removal");
        if(description)p_CFRelease(description);
    }
    if(error)p_CFRelease(error);if(label)p_CFRelease(label);if(definition)p_CFRelease(definition);if(data)p_CFRelease(data);
    return success?0:EIO;
}
#pragma clang diagnostic pop
int alloy_owned_alive(int pid, uint64_t identity) {
#ifdef ALLOY_OWNED_PROCESS_TEST
    if(pid==atomic_load(&test_unknown_identity_pid))return -EIO;
#endif
    struct alloy_unique_info info;
    int error = identity_info(pid, &info);
    if (error == ESRCH || (!error && info.uniqueid != identity)) return 0;
    if(error)return -error;
    struct proc_bsdinfo state;
    int size=proc_pidinfo(pid,PROC_PIDTBSDINFO,0,&state,sizeof(state));
    if(size==sizeof(state) && state.pbi_status==SZOMB)return 0;
    return 1;
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
    if (argc == 4 && !strcmp(argv[1], "--alloy-verification-service-v1")) return verification_service(argc,argv);
    if (argc == 6 && !strcmp(argv[1], "--alloy-cohort-service-v1")) return cohort_service(argc, argv);
    if (argc == 7 && !strcmp(argv[1], "--alloy-shell-service-v2")) return shell_broker(argc, argv);
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
    int cleanup_error = alloy_containment_gate ?
        (kill(target, SIGKILL) && errno != ESRCH ? errno : 0) : cleanup_group(target);
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
static void *recover_cohort_worker(void *opaque) {
    launch_state *state=opaque;
    /* A watcher is mandatory before RUN, so dropping a control never needs to
     * allocate a thread under resource pressure. Until handoff it only observes. */
    while(!atomic_load_explicit(&state->recovery_requested,memory_order_acquire)) {
        if(atomic_load_explicit(&state->state,memory_order_acquire)==DONE)goto done;
        struct timespec idle={0,10000000};nanosleep(&idle,NULL);
    }
    while(atomic_load(&state->recovery_requested)==2 &&
          atomic_load_explicit(&state->state,memory_order_acquire)!=DONE &&
          alloy_owned_alive(state->pid,state->identity)!=0) {
#ifdef ALLOY_OWNED_PROCESS_TEST
        if(!atomic_exchange(&state->test_cleaner_signal_fail,0))
#endif
            (void)alloy_owned_signal(state->pid,state->identity);
        /* Exact-identity signal errors never discard the cleanup obligation. */
        pause_briefly();
    }
    int error=alloy_containment_recover(state);
    state->recovery_error=error;
    /* A dropped caller cannot abandon the obligation. One bounded-resource
     * watcher stays with this charged slot. After successful revocation it
     * retries bounded attempts with a one-second backoff; before revocation
     * it waits for holder acknowledgement or positive kernel absence. */
    while(error){
        struct timespec delay={1,0};nanosleep(&delay,NULL);
        int revoked=atomic_load_explicit(&state->fork_revoked,memory_order_acquire);
        if(revoked==1 || alloy_containment_empty(state->cohort_identity)==1){
            state->recovery_error=0;
            error=alloy_containment_recover(state);
            state->recovery_error=error;
        }
    }
done:
    /* The retired-slot collector pins this mapping until this final store. */
    atomic_store_explicit(&state->recovery_done,1,memory_order_release);
    return NULL;
}
static int ensure_cohort_watcher(launch_state *state) {
    int expected=0;
    if(!atomic_compare_exchange_strong(&state->recovery_started,&expected,1))return 0;
#ifdef ALLOY_OWNED_PROCESS_TEST
    if(atomic_exchange(&test_watcher_start_failure,0)){
        atomic_store(&state->recovery_started,0);return EAGAIN;
    }
#endif
    pthread_attr_t attributes;
    int error=pthread_attr_init(&attributes);
    if(!error){
        error=pthread_attr_setdetachstate(&attributes,PTHREAD_CREATE_DETACHED);
        if(!error){pthread_t thread;error=pthread_create(&thread,&attributes,recover_cohort_worker,state);}
        pthread_attr_destroy(&attributes);
    }
    if(error)atomic_store(&state->recovery_started,0);
    return error;
}
static int start_cohort_recovery(launch_state *state) {
    int error=ensure_cohort_watcher(state);
    if(!error)atomic_store_explicit(&state->recovery_requested,1,memory_order_release);
    return error;
}
int alloy_owned_wait(void *opaque, int *status, int nohang) {
    owned_control *control = opaque;
    uint64_t until = milliseconds() + 2000;
    for (;;) {
        if (atomic_load_explicit(&control->state->state, memory_order_acquire) == DONE) {
            *status = control->state->status;
            return control->state->error;
        }
#ifdef ALLOY_OWNED_PROCESS_TEST
        if(atomic_exchange(&test_wait_for_owner_exit,0)) {
            uint64_t deadline=milliseconds()+1000;
            while(alloy_owned_alive(control->owner_pid,control->owner_identity)==1 && milliseconds()<deadline)pause_briefly();
        }
#endif
        if (alloy_owned_alive(control->owner_pid, control->owner_identity) == 0) {
            /* DONE may have been published after the first state read but
             * before the exclusive supervisor exited. Preserve its real status. */
            if(atomic_load_explicit(&control->state->state,memory_order_acquire)==DONE) {
                *status=control->state->status;return control->state->error;
            }
            if (control->state->cohort_identity) {
                int error=start_cohort_recovery(control->state);
                if(error)return error;
                if(atomic_load_explicit(&control->state->recovery_done,memory_order_acquire))
                    return control->state->recovery_error?control->state->recovery_error:EIO;
                if(nohang)return EAGAIN;
                if(milliseconds()>=until)return ETIMEDOUT;
                pause_briefly();continue;
            }
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
    if(control->shell_slot && !control->owner_identity) {
        retire_shell_slot(control->shell_slot);free(control);return;
    }
    int status;
    /* Legacy owners retain their pinned group leader across delayed cleanup.
     * Cohort owners additionally have an independent revoker and a kernel
     * membership census, so a stalled cleaner can safely hand off to recovery. */
    (void)alloy_owned_wait(control, &status, 0);
    if(control->state->cohort_identity &&
       atomic_load_explicit(&control->state->state,memory_order_acquire)!=DONE) {
        /* Establish the watcher before relinquishing the last caller. Killing
         * only the exact cleaner closes its revocation channel even if it was
         * paused. The independent holder then revokes forks before the sweep. */
        if(!ensure_cohort_watcher(control->state))
            atomic_store_explicit(&control->state->recovery_requested,2,memory_order_release);
    }
    /* External-reaper mode never waits by PID. In host-owned mode the
     * embedding process promises exclusive supervisor reaping. */
    for (int attempt = 0; control->reap_owner && attempt < 100; ++attempt) {
        pid_t result = waitpid(control->owner_pid, &status, WNOHANG);
        if (result == control->owner_pid || (result < 0 && errno != EINTR)) break;
        pause_briefly();
    }
    if(control->shell_slot)retire_shell_slot(control->shell_slot);
    else munmap(control->state, sizeof(*control->state));
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

/* Shell supervisors are children of one isolated, single-threaded reaper.
 * The host never waits on a broker/supervisor PID. On host death the broker
 * drains its children, then exits under launchd. A broken channel fails closed. */
enum { SHELL_MESSAGE=0x41534f57, SHELL_HELLO=0x41535244 };
typedef struct {mach_msg_header_t header;mach_msg_body_t body;mach_msg_port_descriptor_t ports[4];uint64_t generation[2];uint32_t size,fd_count;char data[131072];} shell_packet;
typedef struct { int pid; uint64_t identity; } broker_identity;
static int new_receiver(mach_port_t *port) {
    if(mach_port_allocate(mach_task_self(),MACH_PORT_RIGHT_RECEIVE,port)!=KERN_SUCCESS)return EIO;
    return 0;
}
static int packet_send_tracked(mach_port_t destination,mach_port_t reply,int kind,const uint64_t generation[2],const void *data,uint32_t size,const int *fds,int count,alloy_stop_callback stop,void *context,int *possible) {
    if(size>131072 || count<0 || count>4)return E2BIG;
    shell_packet *packet=calloc(1,sizeof(*packet));if(!packet)return ENOMEM;
    packet->header.msgh_bits=MACH_MSGH_BITS(MACH_MSG_TYPE_COPY_SEND,reply?MACH_MSG_TYPE_MAKE_SEND:0)|MACH_MSGH_BITS_COMPLEX;
    packet->header.msgh_size=sizeof(*packet);packet->header.msgh_remote_port=destination;packet->header.msgh_local_port=reply;packet->header.msgh_id=kind;
    memcpy(packet->generation,generation,sizeof(packet->generation));
    packet->body.msgh_descriptor_count=4;packet->fd_count=(uint32_t)count;packet->size=size;if(size)memcpy(packet->data,data,size);
    int error=0;
    /* Fixed descriptor layout. Unused slots carry ordinary send rights to a
     * private dummy port; receive validates the declared fd count via capacity. */
    mach_port_t dummy=MACH_PORT_NULL;
    if(count<4){if(new_receiver(&dummy)){free(packet);return EIO;}mach_port_insert_right(mach_task_self(),dummy,dummy,MACH_MSG_TYPE_MAKE_SEND);}
    for(int i=0;i<4;++i){
        fileport_t port=MACH_PORT_NULL;
        if(i<count && fileport_makeport(fds[i],&port)){error=errno;break;}
        packet->ports[i].name=i<count?port:dummy;packet->ports[i].type=MACH_MSG_PORT_DESCRIPTOR;packet->ports[i].disposition=MACH_MSG_TYPE_COPY_SEND;
    }
    if(!error)for(;;){
        if(stop && stop(context)){error=ECANCELED;break;}
        if(possible)*possible=1;
        kern_return_t r=mach_msg(&packet->header,MACH_SEND_MSG|MACH_SEND_TIMEOUT,packet->header.msgh_size,0,MACH_PORT_NULL,25,MACH_PORT_NULL);
        if(r==MACH_MSG_SUCCESS)break;
        if(r!=MACH_SEND_TIMED_OUT && r!=MACH_SEND_INTERRUPTED){error=EPIPE;break;}
        if(!stop){error=EPIPE;break;}
    }
    for(int i=0;i<count;++i)if(packet->ports[i].name)mach_port_deallocate(mach_task_self(),packet->ports[i].name);
    if(dummy){mach_port_deallocate(mach_task_self(),dummy);mach_port_mod_refs(mach_task_self(),dummy,MACH_PORT_RIGHT_RECEIVE,-1);}
    free(packet);return error;
}
static int packet_send(mach_port_t destination,mach_port_t reply,int kind,const uint64_t generation[2],const void *data,uint32_t size,const int *fds,int count,alloy_stop_callback stop,void *context) {
    return packet_send_tracked(destination,reply,kind,generation,data,size,fds,count,stop,context,NULL);
}
static int packet_receive(mach_port_t receiver,mach_port_t *reply,int *kind,uint64_t generation[2],audit_token_t *audit,void *data,uint32_t capacity,uint32_t *size,int *fds,int expected,alloy_stop_callback stop,void *context) {
    size_t room=sizeof(shell_packet)+512;
    shell_packet *packet=calloc(1,room);if(!packet)return ENOMEM;
    kern_return_t r;
    for(;;){
        if(stop && stop(context)){free(packet);return ECANCELED;}
        r=mach_msg(&packet->header,MACH_RCV_MSG|MACH_RCV_TIMEOUT|MACH_RCV_TRAILER_TYPE(MACH_MSG_TRAILER_FORMAT_0)|MACH_RCV_TRAILER_ELEMENTS(MACH_RCV_TRAILER_AUDIT),0,(mach_msg_size_t)room,receiver,25,MACH_PORT_NULL);
        if(r==MACH_MSG_SUCCESS)break;
        if(r!=MACH_RCV_TIMED_OUT && r!=MACH_RCV_INTERRUPTED){free(packet);return EPIPE;}
        if(!stop){free(packet);return EAGAIN;}
    }
    int error=0,received=0;
    if((packet->header.msgh_id!=SHELL_MESSAGE && packet->header.msgh_id!=SHELL_HELLO) || packet->header.msgh_size!=sizeof(*packet) || !(packet->header.msgh_bits&MACH_MSGH_BITS_COMPLEX) || packet->body.msgh_descriptor_count!=4 || packet->size>capacity)error=EPROTO;
    mach_msg_audit_trailer_t *trailer=(void *)((char *)packet+round_msg(packet->header.msgh_size));
    if(!error && (trailer->msgh_trailer_type!=MACH_MSG_TRAILER_FORMAT_0 || trailer->msgh_trailer_size<sizeof(*trailer)))error=EPROTO;
    if(expected<0)expected=packet->header.msgh_id==SHELL_HELLO?0:(expected==-4?(int)packet->fd_count:1);
    if(expected<0 || expected>4 || packet->fd_count!=(uint32_t)expected)error=EPROTO;
    if(!error)for(int i=0;i<4;++i){
        if(packet->ports[i].type!=MACH_MSG_PORT_DESCRIPTOR || packet->ports[i].disposition!=MACH_MSG_TYPE_PORT_SEND){error=EPROTO;break;}
        if(i<expected){
            int fd=fileport_makefd(packet->ports[i].name);if(fd<0){error=errno;break;}
            if(fd<3){int copy=fcntl(fd,F_DUPFD_CLOEXEC,3);int saved=errno;close(fd);if(copy<0){error=saved;break;}fd=copy;}
            fds[received++]=fd;
        }
    }
    if(!error){*kind=packet->header.msgh_id;memcpy(generation,packet->generation,sizeof(packet->generation));if(audit)*audit=trailer->msgh_audit;*size=packet->size;memcpy(data,packet->data,*size);if(reply){*reply=packet->header.msgh_remote_port;packet->header.msgh_remote_port=MACH_PORT_NULL;}}
    if(error)for(int i=0;i<received;++i){close(fds[i]);fds[i]=-1;}
    mach_msg_destroy(&packet->header);free(packet);return error;
}
/* The audit trailer comes from the kernel, not message data. Authenticate the
 * running peer against this host's helper designated code requirement. */
static int authenticate_broker(audit_token_t audit,const char *launcher) {
    if(audit_token_to_euid(audit)!=geteuid())return EPERM;
    /* Pipe helpers run with a private bootstrap reply port. Loading CoreFoundation
     * at process startup deadlocks its launchd lookups before main. Load Security
     * only in a verifier/service with its real bootstrap namespace, and retain
     * one library reference for that process lifetime. The multithreaded host
     * delegates this operation through its private anonymous verifier channel. */
    static void *library;
    if(!library)library=dlopen("/System/Library/Frameworks/Security.framework/Security",RTLD_LAZY|RTLD_LOCAL);
    if(!library)return ENOTSUP;
#define LOAD_SECURITY(name) __typeof__(&name) p_##name=(void *)dlsym(library,#name);if(!p_##name)return ENOTSUP
    LOAD_SECURITY(CFURLCreateFromFileSystemRepresentation);
    LOAD_SECURITY(CFDataCreate);
    LOAD_SECURITY(CFDictionaryCreate);
    LOAD_SECURITY(CFRelease);
    LOAD_SECURITY(SecStaticCodeCreateWithPath);
    LOAD_SECURITY(SecCodeCopyDesignatedRequirement);
    LOAD_SECURITY(SecCodeCopyGuestWithAttributes);
    LOAD_SECURITY(SecCodeCheckValidity);
#undef LOAD_SECURITY
    const CFStringRef *audit_key=dlsym(library,"kSecGuestAttributeAudit");
    const CFDictionaryKeyCallBacks *key_callbacks=dlsym(library,"kCFTypeDictionaryKeyCallBacks");
    const CFDictionaryValueCallBacks *value_callbacks=dlsym(library,"kCFTypeDictionaryValueCallBacks");
    if(!audit_key || !key_callbacks || !value_callbacks)return ENOTSUP;
    CFURLRef url=p_CFURLCreateFromFileSystemRepresentation(NULL,(const UInt8 *)launcher,strlen(launcher),false);
    SecStaticCodeRef expected=NULL;SecRequirementRef requirement=NULL;SecCodeRef peer=NULL;
    CFDataRef token=p_CFDataCreate(NULL,(const UInt8 *)&audit,sizeof(audit));
    const void *keys[]={*audit_key};const void *values[]={token};
    CFDictionaryRef attributes=token?p_CFDictionaryCreate(NULL,keys,values,1,key_callbacks,value_callbacks):NULL;
    OSStatus result=errSecParam;
    if(url && attributes && p_SecStaticCodeCreateWithPath(url,kSecCSDefaultFlags,&expected)==errSecSuccess &&
       p_SecCodeCopyDesignatedRequirement(expected,kSecCSDefaultFlags,&requirement)==errSecSuccess &&
       p_SecCodeCopyGuestWithAttributes(NULL,attributes,kSecCSDefaultFlags,&peer)==errSecSuccess)
        result=p_SecCodeCheckValidity(peer,kSecCSStrictValidate,requirement);
    if(peer)p_CFRelease(peer);if(requirement)p_CFRelease(requirement);if(expected)p_CFRelease(expected);
    if(attributes)p_CFRelease(attributes);if(token)p_CFRelease(token);if(url)p_CFRelease(url);
    return result==errSecSuccess?0:EPERM;
}
static int authenticate_host(audit_token_t audit,int parent,uint64_t identity) {
    struct alloy_unique_info info;
    if(audit_token_to_euid(audit)!=geteuid() || audit_token_to_pid(audit)!=parent)return EPERM;
    if(identity_info(parent,&info) || info.uniqueid!=identity || info.idversion!=audit_token_to_pidversion(audit))return EPERM;
    return 0;
}
#include "macos_cohort_service.inc"
#include "macos_verification_service.inc"
static int shell_broker(int argc, char **argv) {
    (void)argc;
    int parent=(int)strtol(argv[3],NULL,10);
    uint64_t parent_identity=strtoull(argv[4],NULL,10);
    struct sigaction action={0};action.sa_handler=SIG_DFL;sigemptyset(&action.sa_mask);
    if(sigaction(SIGCHLD,&action,NULL)) return 125;
    mach_port_t receiver=MACH_PORT_NULL;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    if(bootstrap_check_in(bootstrap_port,argv[5],&receiver)!=KERN_SUCCESS)return 125;
#pragma clang diagnostic pop
    uint64_t generation[2];arc4random_buf(generation,sizeof(generation));
    for (;;) {
        /* Invocation jobs own their cleanup independently of this broker. */
        if (alloy_owned_alive(parent,parent_identity)!=1) return 0;
        char data[131072];uint32_t size=0;int incoming[4]={-1,-1,-1,-1},kind=0;mach_port_t reply_port=MACH_PORT_NULL;
        uint64_t requested_generation[2];audit_token_t audit;
        int error=packet_receive(receiver,&reply_port,&kind,requested_generation,&audit,data,sizeof(data),&size,incoming,-4,NULL,NULL);
        int statefd=incoming[0];
        if(error==EAGAIN)continue;
        if(error)continue;
        error=authenticate_host(audit,parent,parent_identity);
        if(error){for(int i=0;i<4;++i)if(incoming[i]>=0)close(incoming[i]);if(reply_port)mach_port_deallocate(mach_task_self(),reply_port);continue;}
        if(kind==SHELL_HELLO){
            broker_identity identity={getpid(),alloy_process_identity(getpid())};
            (void)packet_send(reply_port,MACH_PORT_NULL,SHELL_HELLO,generation,&identity,sizeof(identity),NULL,0,NULL,NULL);
            mach_port_deallocate(mach_task_self(),reply_port);continue;
        }
        if(memcmp(generation,requested_generation,sizeof(generation))){
            int rejected[2]={ESTALE,0};int empty[3];for(int i=0;i<3;++i)empty[i]=open("/dev/null",O_RDWR|O_CLOEXEC);
            (void)packet_send(reply_port,MACH_PORT_NULL,SHELL_MESSAGE,generation,rejected,sizeof(rejected),empty,3,NULL,NULL);
            for(int i=0;i<3;++i)if(empty[i]>=0)close(empty[i]);for(int i=0;i<4;++i)if(incoming[i]>=0)close(incoming[i]);
            mach_port_deallocate(mach_task_self(),reply_port);continue;
        }
        if(statefd<10){int copy=fcntl(statefd,F_DUPFD_CLOEXEC,10);close(statefd);statefd=copy;incoming[0]=copy;if(copy<0){for(int i=1;i<4;++i)if(incoming[i]>=0)close(incoming[i]);mach_port_deallocate(mach_task_self(),reply_port);continue;}}
        pid_t child=0;
        int exported[3]={-1,-1,-1};
        error=start_cohort(argv[2],argv[6],data,size,incoming,incoming[3]>=0?4:1,&child,exported);
        for(int i=0;i<4;++i)if(incoming[i]>=0)close(incoming[i]);
        int reply[2]={error,child};
        if(error)for(int i=0;i<3;++i){
            if(exported[i]>=0)close(exported[i]);
            exported[i]=open("/dev/null",O_RDWR|O_CLOEXEC);
        }
        (void)packet_send(reply_port,MACH_PORT_NULL,SHELL_MESSAGE,generation,reply,sizeof(reply),exported,3,NULL,NULL);
        mach_port_deallocate(mach_task_self(),reply_port);
        for(int i=0;i<3;++i)if(exported[i]>=0)close(exported[i]);
    }
}
static pthread_mutex_t shell_broker_lock=PTHREAD_MUTEX_INITIALIZER;
static mach_port_t shell_broker_port=MACH_PORT_NULL;
static uint64_t shell_broker_generation[2];
static broker_identity shell_broker_identity;
static uint64_t reconnect_window;
static unsigned reconnect_attempts;
#ifdef ALLOY_OWNED_PROCESS_TEST
static int broker_fault;
void alloy_test_broker_fault(int stage) {
    pthread_mutex_lock(&shell_broker_lock);broker_fault=stage;pthread_mutex_unlock(&shell_broker_lock);
}
int alloy_test_broker_snapshot(int *pid,uint64_t *identity,uint64_t generation[2]) {
    pthread_mutex_lock(&shell_broker_lock);
    *pid=shell_broker_identity.pid;*identity=shell_broker_identity.identity;
    memcpy(generation,shell_broker_generation,sizeof(shell_broker_generation));
    pthread_mutex_unlock(&shell_broker_lock);return *identity?0:ENOENT;
}
int alloy_test_broker_kill(void) {
    pthread_mutex_lock(&shell_broker_lock);
    int error=alloy_owned_signal(shell_broker_identity.pid,shell_broker_identity.identity);
    pthread_mutex_unlock(&shell_broker_lock);return error;
}
int alloy_test_broker_malformed(void) {
    pthread_mutex_lock(&shell_broker_lock);
    mach_msg_header_t message={0};message.msgh_size=sizeof(message);message.msgh_remote_port=shell_broker_port;
    message.msgh_bits=MACH_MSGH_BITS(MACH_MSG_TYPE_COPY_SEND,0);message.msgh_id=123;
    kern_return_t result=mach_msg(&message,MACH_SEND_MSG|MACH_SEND_TIMEOUT,sizeof(message),0,MACH_PORT_NULL,100,MACH_PORT_NULL);
    pthread_mutex_unlock(&shell_broker_lock);return result==KERN_SUCCESS?0:EIO;
}
void alloy_test_broker_stale_generation(void) {
    pthread_mutex_lock(&shell_broker_lock);shell_broker_generation[0]^=1;pthread_mutex_unlock(&shell_broker_lock);
}
int alloy_test_peer_authentication(const char *expected) {
    audit_token_t audit;mach_msg_type_number_t count=TASK_AUDIT_TOKEN_COUNT;
    if(task_info(mach_task_self(),TASK_AUDIT_TOKEN,(task_info_t)&audit,&count)!=KERN_SUCCESS)return EIO;
    return authenticate_broker(audit,expected);
}
static void broker_checkpoint(int stage) {
    if(broker_fault==stage){broker_fault=0;(void)alloy_owned_signal(shell_broker_identity.pid,shell_broker_identity.identity);}
}
#else
static void broker_checkpoint(int stage) {(void)stage;}
#endif
/* A broker restart cannot reset the host's outstanding ownership budget.
 * Dropped, unconfirmed records retain STOP and a bounded shared mapping until
 * cleanup is established; they are never replaced with a numeric PID kill. */
static pthread_mutex_t shell_slots_lock=PTHREAD_MUTEX_INITIALIZER;
static struct {launch_state *state;int retired;} shell_slots[64];
static void collect_shell_slots(void) {
    for(int i=0;i<64;++i){
        launch_state *state=shell_slots[i].state;
        if(!state || !shell_slots[i].retired)continue;
        unsigned phase=atomic_load_explicit(&state->state,memory_order_acquire);
        int finished=phase==DONE;
        if(!finished && !state->cohort_job[0] && state->pid && state->identity)finished=alloy_owned_alive(state->pid,state->identity)==0;
        if(finished && atomic_load(&state->recovery_started) && !atomic_load_explicit(&state->recovery_done,memory_order_acquire))finished=0;
        /* STOP prevents a late workload, but elapsed time cannot prove that
         * an unscheduled supervisor has exited. Unknown ownership stays
         * charged, including across service generations, until proven done. */
        if(finished && state->cohort_record_path[0] && alloy_cohort_record_retire(state->cohort_record_path))finished=0;
        if(finished){munmap(state,sizeof(*state));memset(&shell_slots[i],0,sizeof(shell_slots[i]));}
    }
}
static int reserve_shell_slot(launch_state *state) {
    pthread_mutex_lock(&shell_slots_lock);collect_shell_slots();
    int slot=0;
    for(int i=0;i<64;++i)if(!shell_slots[i].state){shell_slots[i].state=state;slot=i+1;break;}
    pthread_mutex_unlock(&shell_slots_lock);return slot;
}
static void retire_shell_slot(int slot) {
    pthread_mutex_lock(&shell_slots_lock);
    shell_slots[slot-1].retired=1;
    collect_shell_slots();pthread_mutex_unlock(&shell_slots_lock);
}
#ifdef ALLOY_OWNED_PROCESS_TEST
int alloy_test_cleaner_signal_failure(int target) {
    int error=ESRCH;pthread_mutex_lock(&shell_slots_lock);
    for(int i=0;i<64;++i){launch_state *state=shell_slots[i].state;
        if(state && state->target_pid==target){atomic_store(&state->test_cleaner_signal_fail,1);error=0;break;}}
    pthread_mutex_unlock(&shell_slots_lock);return error;
}
int alloy_test_cohort_fail_recovery(int target) {
    int error=ESRCH;
    pthread_mutex_lock(&shell_slots_lock);
    for(int i=0;i<64;++i){launch_state *state=shell_slots[i].state;
        if(state && state->target_pid==target){atomic_store(&state->test_recovery_fail,1);error=0;break;}}
    pthread_mutex_unlock(&shell_slots_lock);return error;
}
int alloy_test_cohort_startup_stage(int setup) {
    int result=0;
    pthread_mutex_lock(&shell_slots_lock);
    for(int i=0;i<64;++i){launch_state *state=shell_slots[i].state;
        if(!state || !atomic_load(setup?&state->test_setup_paused:&state->test_submit_paused))continue;
        /* The gate publishes its pause from the fork child. Its parent may
         * still be publishing the identities used by the crash injection.
         * Report the checkpoint only when both processes can be addressed
         * by their exact identities, without weakening identity-safe signals. */
        if(setup && (!atomic_load(&state->gate_pid) || !atomic_load(&state->gate_identity) ||
                     !atomic_load(&state->pid) || !atomic_load(&state->identity)))continue;
        result=1;break;}
    pthread_mutex_unlock(&shell_slots_lock);return result;
}
int alloy_test_cohort_recovery(int target,int revoked) {
    int result=-ESRCH;
    pthread_mutex_lock(&shell_slots_lock);
    for(int i=0;i<64;++i){launch_state *state=shell_slots[i].state;
        if(state && state->target_pid==target){result=revoked?atomic_load(&state->fork_revoked):atomic_load(&state->recovery_error);break;}}
    pthread_mutex_unlock(&shell_slots_lock);return result;
}
int alloy_test_uncertain_cohort_record(int target) {
    char directory[1024]={0};
    pthread_mutex_lock(&shell_slots_lock);
    for(int i=0;i<64;++i){launch_state *state=shell_slots[i].state;
        if(state && state->target_pid==target){snprintf(directory,sizeof(directory),"%s",state->cohort_record_path);break;}}
    pthread_mutex_unlock(&shell_slots_lock);
    char *slash=strrchr(directory,'/');if(!slash)return EINVAL;*slash=0;
    /* This fixture never submits a job. Give its record a proven-dead identity
     * so retirement must distinguish an indeterminate RPC from no reservation. */
    pid_t child=fork();if(!child)_exit(0);if(child<0)return errno;
    uint64_t identity=alloy_process_identity(child);int status;
    while(waitpid(child,&status,0)<0){if(errno!=EINTR)return errno;}
    if(!identity)return EIO;
    launch_state state={0};state.parent_pid=child;state.parent_identity=identity;
    uint64_t random[2];arc4random_buf(random,sizeof(random));
    snprintf(state.cohort_job,sizeof(state.cohort_job),"com.layerdynamics.alloy.cohort.%u.%d.%llu.%016llx%016llx",getuid(),child,
        (unsigned long long)identity,(unsigned long long)random[0],(unsigned long long)random[1]);
    int lease=-1,error=alloy_cohort_record_create(directory,&state,&lease);
    if(error)return error;
    error=alloy_cohort_record_submitting(state.cohort_record_path);close(lease);
    if(!error){int retired=alloy_cohort_record_retire(state.cohort_record_path);
        if(retired!=EAGAIN || access(state.cohort_record_path,F_OK))error=EIO;}
    /* The fixture knows it never called SMJobSubmit, so it can publish that
     * no-workload fact and remove its own synthetic record after the assertion. */
    int complete=alloy_cohort_record_complete(state.cohort_record_path,0);
    int retired=complete?complete:alloy_cohort_record_retire(state.cohort_record_path);
    return error?error:retired;
}
static int cohort_holder_pause(int target,int gate,int paused) {
    int error=ESRCH;
    pthread_mutex_lock(&shell_slots_lock);
    for(int i=0;i<64;++i){
        launch_state *state=shell_slots[i].state;
        if(!state || state->target_pid!=target)continue;
        error=identity_signal(gate?state->gate_pid:state->revoker_pid,
            gate?state->gate_identity:state->revoker_identity,paused?SIGSTOP:SIGCONT);break;
    }
    pthread_mutex_unlock(&shell_slots_lock);return error;
}
int alloy_test_cohort_revoker_pause(int target,int paused) {return cohort_holder_pause(target,0,paused);}
int alloy_test_cohort_gate_pause(int target,int paused) {return cohort_holder_pause(target,1,paused);}
int alloy_test_cohort_revoke_barrier(int target,int operation) {
    int result=-ESRCH;
    pthread_mutex_lock(&shell_slots_lock);
    for(int i=0;i<64;++i){
        launch_state *state=shell_slots[i].state;
        if(!state || state->target_pid!=target)continue;
        if(operation==1)atomic_store(&state->test_revoke_pause,1);
        if(operation==0)atomic_store(&state->test_revoke_pause,0);
        result=atomic_load(&state->test_revoke_paused);break;
    }
    pthread_mutex_unlock(&shell_slots_lock);return result;
}
int alloy_test_cohort_record(int target,char *path,char *label,size_t capacity) {
    int error=ESRCH;
    pthread_mutex_lock(&shell_slots_lock);
    for(int i=0;i<64;++i){
        launch_state *state=shell_slots[i].state;
        if(!state || state->target_pid!=target)continue;
        if(strlen(state->cohort_record_path)>=capacity || strlen(state->cohort_job)>=capacity){error=E2BIG;break;}
        snprintf(path,capacity,"%s",state->cohort_record_path);
        snprintf(label,capacity,"%s",state->cohort_job);error=0;break;
    }
    pthread_mutex_unlock(&shell_slots_lock);
    return error;
}
int alloy_test_cohort_kill(int target,int revoker) {
    int error=ESRCH;
    pthread_mutex_lock(&shell_slots_lock);
    for(int i=0;i<64;++i){
        launch_state *state=shell_slots[i].state;
        if(!state || state->target_pid!=target)continue;
        int pid=revoker==2?state->pid:(revoker?state->revoker_pid:state->gate_pid);
        uint64_t identity=revoker==2?state->identity:(revoker?state->revoker_identity:state->gate_identity);
        if(pid>0 && identity)error=alloy_owned_signal(pid,identity);
        break;
    }
    pthread_mutex_unlock(&shell_slots_lock);
    return error;
}
int alloy_test_shell_owner_pause(int target,int paused) {
    int error=ESRCH;
    pthread_mutex_lock(&shell_slots_lock);
    for(int i=0;i<64;++i){
        launch_state *state=shell_slots[i].state;
        if(!state || state->target_pid!=target)continue;
        int pid=state->pid;uint64_t identity=state->identity;
        error=identity_signal(pid,identity,paused?SIGSTOP:SIGCONT);
        uint64_t until=milliseconds()+2000;
        while(!error && paused){
            struct proc_bsdinfo info;
            if(proc_pidinfo(pid,PROC_PIDTBSDINFO,0,&info,sizeof(info))==sizeof(info) && info.pbi_status==SSTOP)break;
            if(milliseconds()>=until){error=ETIMEDOUT;(void)identity_signal(pid,identity,SIGCONT);break;}
            pause_briefly();
        }
        break;
    }
    pthread_mutex_unlock(&shell_slots_lock);return error;
}
int alloy_test_shell_cancellation_pending(int target) {
    int pending=0;
    pthread_mutex_lock(&shell_slots_lock);
    for(int i=0;i<64;++i){
        launch_state *state=shell_slots[i].state;
        if(!state || state->target_pid!=target)continue;
        struct proc_bsdinfo info;
        pending=atomic_load_explicit(&state->state,memory_order_acquire)==STOP &&
            proc_pidinfo(state->pid,PROC_PIDTBSDINFO,0,&info,sizeof(info))==sizeof(info) && info.pbi_status==SSTOP;
        break;
    }
    pthread_mutex_unlock(&shell_slots_lock);return pending;
}
static launch_state *test_stalled_state;
static pid_t test_stalled_pid;
int alloy_test_pid_only_slot(void) {
    launch_state *state=mmap(NULL,sizeof(*state),PROT_READ|PROT_WRITE,MAP_SHARED|MAP_ANON,-1,0);
    if(state==MAP_FAILED)return errno;
    memset(state,0,sizeof(*state));atomic_init(&state->state,STOP);
    /* Legal partial publication: the native owner stores pid before identity.
     * No workload can be created because READY/CREATE were never published. */
    /* The child uses only async-signal-safe calls after fork. It remains a
     * real stopped process until the test's cleanup proves its wait status. */
    pid_t child=fork();
    if(child<0){int error=errno;munmap(state,sizeof(*state));return -error;}
    if(!child){kill(getpid(),SIGSTOP);_exit(0);}
    int status;
    while(waitpid(child,&status,WUNTRACED)<0){if(errno!=EINTR)return -errno;}
    state->pid=child;
    int slot=reserve_shell_slot(state);
    if(!slot){kill(child,SIGKILL);waitpid(child,NULL,0);munmap(state,sizeof(*state));return -EBUSY;}
    test_stalled_state=state;test_stalled_pid=child;
    retire_shell_slot(slot);return child;
}
int alloy_test_finish_stalled_owner(void) {
    if(!test_stalled_pid)return EINVAL;
    if(kill(test_stalled_pid,SIGKILL))return errno;
    while(waitpid(test_stalled_pid,NULL,0)<0){if(errno!=EINTR)return errno;}
    atomic_store_explicit(&test_stalled_state->state,DONE,memory_order_release);
    test_stalled_pid=0;test_stalled_state=NULL;return 0;
}
#endif
typedef struct {
    alloy_stop_callback stop;
    void *context;
    mach_port_t endpoint;
    broker_identity peer;
    uint64_t until;
    int lost, timed_out;
} broker_wait;
static int broker_wait_stopped(void *opaque) {
#ifdef ALLOY_OWNED_PROCESS_TEST
    if(broker_fault==3){broker_fault=0;return 1;}
#endif
    broker_wait *wait=opaque;
    mach_port_type_t type=0;
    if(mach_port_type(mach_task_self(),wait->endpoint,&type)!=KERN_SUCCESS ||
       (type&MACH_PORT_TYPE_DEAD_NAME) ||
       (wait->peer.identity && alloy_owned_alive(wait->peer.pid,wait->peer.identity)==0)) {
        wait->lost=1;return 1;
    }
    if(milliseconds()>=wait->until){wait->timed_out=1;return 1;}
    return wait->stop && wait->stop(wait->context);
}
static void disconnect_shell_broker(void) {
    if(shell_broker_port)mach_port_deallocate(mach_task_self(),shell_broker_port);
    shell_broker_port=MACH_PORT_NULL;
    memset(shell_broker_generation,0,sizeof(shell_broker_generation));
    memset(&shell_broker_identity,0,sizeof(shell_broker_identity));
}
static int connect_shell_broker(const char *launcher,const char *service,alloy_stop_callback stop,void *context) {
    if(shell_broker_port){
        mach_port_type_t type=0;
        if(mach_port_type(mach_task_self(),shell_broker_port,&type)==KERN_SUCCESS &&
           !(type&MACH_PORT_TYPE_DEAD_NAME) && alloy_owned_alive(shell_broker_identity.pid,shell_broker_identity.identity)==1)return 0;
        disconnect_shell_broker();
    }
    uint64_t now=milliseconds();
    if(!reconnect_window || now-reconnect_window>=10000){reconnect_window=now;reconnect_attempts=0;}
    if(reconnect_attempts>=3)return EAGAIN;
    ++reconnect_attempts;
    mach_port_t endpoint=MACH_PORT_NULL,receiver=MACH_PORT_NULL;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    if(bootstrap_look_up(bootstrap_port,service,&endpoint)!=KERN_SUCCESS)return ENOENT;
#pragma clang diagnostic pop
    int error=new_receiver(&receiver);
    broker_wait wait={stop,context,endpoint,{0,0},milliseconds()+5000,0,0};
    uint64_t generation[2]={0,0};
    if(!error)error=packet_send(endpoint,receiver,SHELL_HELLO,generation,NULL,0,NULL,0,broker_wait_stopped,&wait);
    broker_identity identity={0};uint32_t size=0;int kind=0;audit_token_t audit;
    if(!error)error=packet_receive(receiver,NULL,&kind,generation,&audit,&identity,sizeof(identity),&size,NULL,0,broker_wait_stopped,&wait);
    if(error && wait.lost)error=EPIPE;
    else if(error && wait.timed_out)error=ETIMEDOUT;
    if(!error && (kind!=SHELL_HELLO || size!=sizeof(identity) || (!generation[0]&&!generation[1]) || identity.pid!=audit_token_to_pid(audit)))error=EPROTO;
    if(!error)error=authenticate_in_verifier(audit,launcher,service,stop,context);
    if(!error && alloy_owned_alive(identity.pid,identity.identity)!=1)error=EPIPE;
    if(!error){shell_broker_port=endpoint;endpoint=MACH_PORT_NULL;memcpy(shell_broker_generation,generation,sizeof(generation));shell_broker_identity=identity;}
    if(endpoint)mach_port_deallocate(mach_task_self(),endpoint);
    if(receiver)mach_port_mod_refs(mach_task_self(),receiver,MACH_PORT_RIGHT_RECEIVE,-1);
    return error;
}
static int broker_spawn(const char *mode,const char *launcher,const char *service,const char *program,const char *cwd,const int io[3],
    char *const argv[],alloy_stop_callback stop,void *context,int *pid,void **out_control,int result_fds[3]) {
    SHELL_STAGE("broker connection");
#ifdef ALLOY_OWNED_PROCESS_TEST
    test_shell_diagnostic[0]=0;
#endif
    *pid=0;*out_control=NULL;
    for(int i=0;i<3;++i)result_fds[i]=-1;
    while(pthread_mutex_trylock(&shell_broker_lock)!=0) {if(stop(context))return ECANCELED;pause_briefly();}
    int error=connect_shell_broker(launcher,service,stop,context);
    broker_checkpoint(1);
    mach_port_t reply_port=MACH_PORT_NULL;
    char name[32];snprintf(name,sizeof(name),"/alloy-%016llx",(unsigned long long)(((uint64_t)arc4random()<<32)|arc4random()));
    int fd=-1,possible=0;launch_state *state=MAP_FAILED;owned_control *control=NULL;char *data=NULL;
    if(error)goto done;
    SHELL_STAGE("shared launch state allocation");
    fd=shm_open(name,O_CREAT|O_EXCL|O_RDWR,0600);if(fd<0){error=errno;goto done;}shm_unlink(name);
    if(fd<3){int copy=fcntl(fd,F_DUPFD_CLOEXEC,3);int saved=errno;close(fd);fd=copy;if(fd<0){error=saved;goto done;}}
    if(ftruncate(fd,sizeof(*state))){error=errno;goto done;}
    state=mmap(NULL,sizeof(*state),PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
    if(state==MAP_FAILED){error=errno;goto done;}
    memset(state,0,sizeof(*state));atomic_init(&state->state,INITIAL);
    struct alloy_unique_info parent;error=identity_info(getpid(),&parent);if(error)goto done;
    state->parent_pid=getpid();state->parent_identity=parent.uniqueid;
#ifdef ALLOY_OWNED_PROCESS_TEST
    if(broker_fault==4){state->test_submit_pause=1;broker_fault=0;}
    if(broker_fault==5){state->test_setup_pause=1;broker_fault=0;}
#endif
    size_t size=strlen(mode)+1+strlen(cwd)+1;for(size_t i=0;argv[i];++i){size+=strlen(argv[i])+1;if(size>131072){error=E2BIG;goto done;}}
    data=malloc(size);control=calloc(1,sizeof(*control));if(control)control->state=state; if(!data||!control){error=ENOMEM;goto done;}
    control->shell_slot=reserve_shell_slot(state);if(!control->shell_slot){error=EBUSY;goto done;}
    size_t offset=strlen(mode)+1;memcpy(data,mode,offset);
    size_t cwd_size=strlen(cwd)+1;memcpy(data+offset,cwd,cwd_size);offset+=cwd_size;
    for(size_t i=0;argv[i];++i){size_t n=strlen(argv[i])+1;memcpy(data+offset,argv[i],n);offset+=n;}
    (void)program;
    error=new_receiver(&reply_port);
    SHELL_STAGE("launch request send");
    broker_wait wait={stop,context,shell_broker_port,shell_broker_identity,milliseconds()+5000,0,0};
    int request_fds[4]={fd,io?io[0]:-1,io?io[1]:-1,io?io[2]:-1};
    if(!error)error=packet_send_tracked(shell_broker_port,reply_port,SHELL_MESSAGE,shell_broker_generation,data,(uint32_t)size,request_fds,io?4:1,broker_wait_stopped,&wait,&possible);
    if(!error)broker_checkpoint(2);
    SHELL_STAGE("launch reply receive");
    int reply[2]={0};uint32_t reply_size=0;int kind=0;uint64_t generation[2];audit_token_t audit;
    if(!error)error=packet_receive(reply_port,NULL,&kind,generation,&audit,reply,sizeof(reply),&reply_size,result_fds,3,broker_wait_stopped,&wait);
    if(error && wait.lost)error=EPIPE;
    else if(error && wait.timed_out)error=ETIMEDOUT;
    if(!error && (kind!=SHELL_MESSAGE || memcmp(generation,shell_broker_generation,sizeof(generation)) || audit_token_to_pid(audit)!=shell_broker_identity.pid))error=ESTALE;
    if(error)goto done;
    if(reply_size!=sizeof(reply)){error=EPROTO;goto done;}
    if(reply[0]){if(!reply[1])possible=0;error=reply[0];goto done;}
    control->state=state;control->owner_pid=reply[1];control->reap_owner=0;
    SHELL_STAGE("supervisor startup state");
    uint64_t ready_until=milliseconds()+5000;
    for(;;) {
        unsigned phase=atomic_load_explicit(&state->state,memory_order_acquire);
        if(phase==READY) {
            if(state->pid!=reply[1] || !state->identity){error=EIO;break;}
            control->owner_identity=state->identity;
            if(stop(context)){error=ECANCELED;break;}
            unsigned expected=READY;atomic_compare_exchange_strong(&state->state,&expected,CREATE);
        } else if(phase==TARGET_READY) {
            if(stop(context)){error=ECANCELED;break;}
            error=ensure_cohort_watcher(state);
            if(error)break; /* Never execute a workload without its retained watcher. */
            unsigned expected=TARGET_READY;
            if(!atomic_compare_exchange_strong(&state->state,&expected,RUN))continue;
            *pid=state->target_pid;*out_control=control;control=NULL;state=MAP_FAILED;break;
        } else if(phase==DONE){error=state->error?state->error:EIO;break;}
        if(control->owner_identity && alloy_owned_alive(control->owner_pid,control->owner_identity)==0){error=EIO;break;}
        if(milliseconds()>=ready_until){error=ETIMEDOUT;break;}
        if(stop(context)){error=ECANCELED;break;}pause_briefly();
    }
done:
#ifdef ALLOY_OWNED_PROCESS_TEST
    if(error)snprintf(test_shell_diagnostic,sizeof(test_shell_diagnostic),"%s: errno=%d state=%u owner=%d target=%d status=%d supervisor_error=%d",test_shell_stage,error,state==MAP_FAILED?0:state->state,state==MAP_FAILED?0:state->pid,state==MAP_FAILED?0:state->target_pid,state==MAP_FAILED?0:state->status,state==MAP_FAILED?0:state->error);
#endif
    if(error && state!=MAP_FAILED) {
        /* No send was attempted, or an authenticated reply proved no child
         * was created. These slots can be released without guessing a PID. */
        if(!possible)atomic_store_explicit(&state->state,DONE,memory_order_release);
        /* STOP before CREATE/RUN prevents an abandoned request executing later. */
        unsigned phase=atomic_load_explicit(&state->state,memory_order_acquire);
        while(phase!=DONE && phase!=STOP && !atomic_compare_exchange_weak(&state->state,&phase,STOP)){}
    }
    if(control && (control->owner_pid || control->shell_slot)){alloy_owned_close(control);state=MAP_FAILED;}
    else free(control);
    if(state!=MAP_FAILED)munmap(state,sizeof(*state));
    if(fd>=0)close(fd);free(data);
    if(error)for(int i=0;i<3;++i)if(result_fds[i]>=0){close(result_fds[i]);result_fds[i]=-1;}
    if(reply_port)mach_port_mod_refs(mach_task_self(),reply_port,MACH_PORT_RIGHT_RECEIVE,-1);
    if(error==EPIPE || error==ESTALE)disconnect_shell_broker();
    pthread_mutex_unlock(&shell_broker_lock);
    return error;
}

int alloy_shell_spawn(const char *launcher,const char *service,const char *program,const char *cwd,const int io[3],
    char *const argv[],alloy_stop_callback stop,void *context,int *pid,void **out_control,int result_fds[3]) {
    return broker_spawn("shell",launcher,service,program,cwd,io,argv,stop,context,pid,out_control,result_fds);
}
int alloy_trusted_spawn(const char *launcher,const char *service,const char *program,const char *cwd,const int io[3],
    char *const argv[],alloy_stop_callback stop,void *context,int *pid,void **out_control,int result_fds[3]) {
    return broker_spawn("plugin",launcher,service,program,cwd,io,argv,stop,context,pid,out_control,result_fds);
}
