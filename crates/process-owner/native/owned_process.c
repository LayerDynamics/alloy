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
#include <pthread.h>
#include <poll.h>
#include <sys/socket.h>
#include <mach/mach.h>
#include <mach/task_special_ports.h>
#include <sys/fileport.h>

static int shell_broker(int argc, char **argv);

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
    if (argc == 5 && !strcmp(argv[1], "--alloy-shell-broker-v1")) return shell_broker(argc, argv);
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

/* Shell supervisors are children of one isolated, single-threaded reaper.
 * The host never waits on a broker/supervisor PID. On host death the broker
 * drains its children, then exits under launchd. A broken channel fails closed. */
enum { SHELL_MESSAGE=0x41534f57, SHELL_READY=0x41535244 };
typedef struct {mach_msg_header_t header;mach_msg_body_t body;mach_msg_port_descriptor_t ports[3];uint32_t size;char data[131072];} shell_packet;
static int new_receiver(mach_port_t *port) {
    if(mach_port_allocate(mach_task_self(),MACH_PORT_RIGHT_RECEIVE,port)!=KERN_SUCCESS)return EIO;
    return 0;
}
static int packet_send(mach_port_t destination,mach_port_t reply,const void *data,uint32_t size,const int *fds,int count,alloy_stop_callback stop,void *context) {
    if(size>131072 || count>3)return E2BIG;
    shell_packet *packet=calloc(1,sizeof(*packet));if(!packet)return ENOMEM;
    packet->header.msgh_bits=MACH_MSGH_BITS(MACH_MSG_TYPE_COPY_SEND,reply?MACH_MSG_TYPE_MAKE_SEND:0)|MACH_MSGH_BITS_COMPLEX;
    packet->header.msgh_size=sizeof(*packet);packet->header.msgh_remote_port=destination;packet->header.msgh_local_port=reply;packet->header.msgh_id=SHELL_MESSAGE;
    packet->body.msgh_descriptor_count=3;packet->size=size;memcpy(packet->data,data,size);
    int error=0;
    /* Fixed descriptor layout. Unused slots carry ordinary send rights to a
     * private dummy port; receive validates the declared fd count via capacity. */
    mach_port_t dummy=MACH_PORT_NULL;
    if(count<3){if(new_receiver(&dummy)){free(packet);return EIO;}mach_port_insert_right(mach_task_self(),dummy,dummy,MACH_MSG_TYPE_MAKE_SEND);}
    for(int i=0;i<3;++i){
        fileport_t port=MACH_PORT_NULL;
        if(i<count && fileport_makeport(fds[i],&port)){error=errno;break;}
        packet->ports[i].name=i<count?port:dummy;packet->ports[i].type=MACH_MSG_PORT_DESCRIPTOR;packet->ports[i].disposition=MACH_MSG_TYPE_COPY_SEND;
    }
    if(!error)for(;;){
        if(stop && stop(context)){error=ECANCELED;break;}
        kern_return_t r=mach_msg(&packet->header,MACH_SEND_MSG|MACH_SEND_TIMEOUT,packet->header.msgh_size,0,MACH_PORT_NULL,25,MACH_PORT_NULL);
        if(r==MACH_MSG_SUCCESS)break;
        if(r!=MACH_SEND_TIMED_OUT && r!=MACH_SEND_INTERRUPTED){error=EPIPE;break;}
        if(!stop){error=EPIPE;break;}
    }
    for(int i=0;i<count;++i)if(packet->ports[i].name)mach_port_deallocate(mach_task_self(),packet->ports[i].name);
    if(dummy){mach_port_deallocate(mach_task_self(),dummy);mach_port_mod_refs(mach_task_self(),dummy,MACH_PORT_RIGHT_RECEIVE,-1);}
    free(packet);return error;
}
static int packet_receive(mach_port_t receiver,mach_port_t *reply,void *data,uint32_t capacity,uint32_t *size,int *fds,int expected,alloy_stop_callback stop,void *context) {
    size_t room=sizeof(shell_packet)+512;
    shell_packet *packet=calloc(1,room);if(!packet)return ENOMEM;
    kern_return_t r;
    for(;;){
        if(stop && stop(context)){free(packet);return ECANCELED;}
        r=mach_msg(&packet->header,MACH_RCV_MSG|MACH_RCV_TIMEOUT,0,(mach_msg_size_t)room,receiver,25,MACH_PORT_NULL);
        if(r==MACH_MSG_SUCCESS)break;
        if(r!=MACH_RCV_TIMED_OUT && r!=MACH_RCV_INTERRUPTED){free(packet);return EPIPE;}
        if(!stop){free(packet);return EAGAIN;}
    }
    int error=0,received=0;
    if(packet->header.msgh_id!=SHELL_MESSAGE || packet->header.msgh_size!=sizeof(*packet) || !(packet->header.msgh_bits&MACH_MSGH_BITS_COMPLEX) || packet->body.msgh_descriptor_count!=3 || packet->size>capacity)error=EPROTO;
    if(!error)for(int i=0;i<3;++i){
        if(packet->ports[i].type!=MACH_MSG_PORT_DESCRIPTOR || packet->ports[i].disposition!=MACH_MSG_TYPE_PORT_SEND){error=EPROTO;break;}
        if(i<expected){int fd=fileport_makefd(packet->ports[i].name);if(fd<0){error=errno;break;}fds[received++]=fd;}
    }
    if(!error){*size=packet->size;memcpy(data,packet->data,*size);if(reply){*reply=packet->header.msgh_remote_port;packet->header.msgh_remote_port=MACH_PORT_NULL;}}
    if(error)for(int i=0;i<received;++i){close(fds[i]);fds[i]=-1;}
    mach_msg_destroy(&packet->header);free(packet);return error;
}
static int shell_broker(int argc, char **argv) {
    (void)argc;
    int parent=(int)strtol(argv[3],NULL,10);
    uint64_t parent_identity=strtoull(argv[4],NULL,10);
    struct sigaction action={0};action.sa_handler=SIG_DFL;sigemptyset(&action.sa_mask);
    if(sigaction(SIGCHLD,&action,NULL)) return 125;
    mach_port_t host=MACH_PORT_NULL, receiver=MACH_PORT_NULL;
    if(task_get_special_port(mach_task_self(),TASK_BOOTSTRAP_PORT,&host)!=KERN_SUCCESS || new_receiver(&receiver))return 125;
    struct {mach_msg_header_t header;mach_msg_body_t body;mach_msg_port_descriptor_t port;} ready={0};
    ready.header.msgh_bits=MACH_MSGH_BITS(MACH_MSG_TYPE_COPY_SEND,0)|MACH_MSGH_BITS_COMPLEX;ready.header.msgh_size=sizeof(ready);ready.header.msgh_remote_port=host;ready.header.msgh_id=SHELL_READY;
    ready.body.msgh_descriptor_count=1;ready.port.name=receiver;ready.port.type=MACH_MSG_PORT_DESCRIPTOR;ready.port.disposition=MACH_MSG_TYPE_MAKE_SEND;
    if(mach_msg(&ready.header,MACH_SEND_MSG|MACH_SEND_TIMEOUT,sizeof(ready),0,MACH_PORT_NULL,2000,MACH_PORT_NULL)!=KERN_SUCCESS)return 125;
    mach_port_deallocate(mach_task_self(),host);
    task_set_special_port(mach_task_self(),TASK_BOOTSTRAP_PORT,MACH_PORT_NULL);
    int connected=1,children=0;
    for (;;) {
        int status;pid_t reaped;
        while((reaped=waitpid(-1,&status,WNOHANG))>0) --children;
        if (!connected || alloy_owned_alive(parent,parent_identity)!=1) {
            if (!children) return 0;
            connected=0;pause_briefly();continue;
        }
        char data[131072];uint32_t size=0;int statefd=-1;mach_port_t reply_port=MACH_PORT_NULL;
        int error=packet_receive(receiver,&reply_port,data,sizeof(data),&size,&statefd,1,NULL,NULL);
        if(error==EAGAIN)continue;
        if(error){connected=0;continue;}
        if(statefd<10){int copy=fcntl(statefd,F_DUPFD_CLOEXEC,10);close(statefd);statefd=copy;if(copy<0){mach_port_deallocate(mach_task_self(),reply_port);continue;}}
        char *strings[4098];size_t count=0,offset=0;
        while(offset<size && count<4098) {
            char *end=memchr(data+offset,0,size-offset);
            if(!end)break;
            strings[count++]=data+offset;offset=(size_t)(end-data)+1;
        }
        int endpoints[6]={-1,-1,-1,-1,-1,-1};
        pid_t child=0;
        posix_spawnattr_t attr=NULL;posix_spawn_file_actions_t actions=NULL;
        if(offset!=size || count<2 || children>=64) error=EINVAL;
        if(!error && (pipe(endpoints)||pipe(endpoints+2)||pipe(endpoints+4))) error=errno;
        if(!error) error=posix_spawnattr_init(&attr);
        if(!error) error=posix_spawn_file_actions_init(&actions);
        if(!error) error=posix_spawnattr_setflags(&attr,POSIX_SPAWN_CLOEXEC_DEFAULT|POSIX_SPAWN_SETSID);
        if(!error) error=posix_spawn_file_actions_addopen(&actions,0,"/dev/null",O_RDONLY,0);
        if(!error) error=posix_spawn_file_actions_adddup2(&actions,endpoints[3],1);
        if(!error) error=posix_spawn_file_actions_adddup2(&actions,endpoints[5],2);
        if(!error) error=posix_spawn_file_actions_adddup2(&actions,statefd,3);
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        if(!error) error=posix_spawn_file_actions_addchdir_np(&actions,strings[0]);
#pragma clang diagnostic pop
        char *arguments[4100]={argv[2],"--alloy-owned-exec-v1",NULL};
        for(size_t i=1;i<count;++i) arguments[i+1]=strings[i];
        char *environment[]={NULL};
        if(!error) error=posix_spawn(&child,argv[2],&actions,&attr,arguments,environment);
        if(!error) ++children;
        if(actions)posix_spawn_file_actions_destroy(&actions);
        if(attr)posix_spawnattr_destroy(&attr);
        close(statefd);
        int reply[2]={error,child};
        int exported[3]={endpoints[1],endpoints[2],endpoints[4]};
        /* Always supply three valid descriptors; error replies use /dev/null. */
        if(error)for(int i=0;i<3;++i)exported[i]=open("/dev/null",O_RDWR|O_CLOEXEC);
        (void)packet_send(reply_port,MACH_PORT_NULL,reply,sizeof(reply),exported,3,NULL,NULL);
        mach_port_deallocate(mach_task_self(),reply_port);
        if(error)for(int i=0;i<3;++i)if(exported[i]>=0)close(exported[i]);
        for(int i=0;i<6;++i)if(endpoints[i]>=0)close(endpoints[i]);
    }
}
static pthread_mutex_t shell_broker_lock=PTHREAD_MUTEX_INITIALIZER;
static mach_port_t shell_broker_port=MACH_PORT_NULL, shell_pending_receiver=MACH_PORT_NULL;
static uint64_t shell_pending_deadline;
static int shell_broker_error;
static int connect_shell_broker(const char *launcher,alloy_stop_callback stop,void *context) {
    if(shell_broker_port)return 0;
    if(shell_broker_error)return shell_broker_error;
    mach_port_t receiver=shell_pending_receiver;
    int error=0;
    if(!receiver){
    error=new_receiver(&receiver);if(error)return error;
    mach_port_insert_right(mach_task_self(),receiver,receiver,MACH_MSG_TYPE_MAKE_SEND);
    struct alloy_unique_info parent={0};error=identity_info(getpid(),&parent);
    char pidtext[24],identitytext[32];snprintf(pidtext,sizeof(pidtext),"%d",getpid());snprintf(identitytext,sizeof(identitytext),"%llu",(unsigned long long)parent.uniqueid);
    posix_spawnattr_t attr=NULL;posix_spawn_file_actions_t actions=NULL;
    if(!error)error=posix_spawnattr_init(&attr);if(!error)error=posix_spawn_file_actions_init(&actions);
    if(!error)error=posix_spawnattr_setflags(&attr,POSIX_SPAWN_CLOEXEC_DEFAULT|POSIX_SPAWN_SETSID);
    if(!error)error=posix_spawnattr_setspecialport_np(&attr,receiver,TASK_BOOTSTRAP_PORT);
    for(int fd=0;!error && fd<3;++fd)error=posix_spawn_file_actions_addopen(&actions,fd,"/dev/null",fd?O_WRONLY:O_RDONLY,0);
    char *arguments[]={(char *)launcher,"--alloy-shell-broker-v1",(char *)launcher,pidtext,identitytext,NULL};char *environment[]={NULL};pid_t broker=0;
    if(!error)error=posix_spawn(&broker,launcher,&actions,&attr,arguments,environment);
    if(actions)posix_spawn_file_actions_destroy(&actions);if(attr)posix_spawnattr_destroy(&attr);
    if(!error){shell_pending_receiver=receiver;shell_pending_deadline=milliseconds()+5000;}
    }
    if(!error){
        union {mach_msg_header_t header;char bytes[1024];} reply={0};
        uint64_t until=shell_pending_deadline;
        for(;;){
            if(stop(context)){return ECANCELED;}
            kern_return_t result=mach_msg(&reply.header,MACH_RCV_MSG|MACH_RCV_TIMEOUT,0,sizeof(reply),receiver,25,MACH_PORT_NULL);
            if(result==MACH_RCV_TIMED_OUT||result==MACH_RCV_INTERRUPTED){if(milliseconds()>=until){error=ETIMEDOUT;break;}continue;}
            if(result!=KERN_SUCCESS){error=EPIPE;break;}
            struct ready_message {mach_msg_header_t header;mach_msg_body_t body;mach_msg_port_descriptor_t port;} *ready=(void *)&reply;
            if(ready->header.msgh_id!=SHELL_READY||ready->header.msgh_size!=sizeof(*ready)||!(ready->header.msgh_bits&MACH_MSGH_BITS_COMPLEX)||ready->body.msgh_descriptor_count!=1||ready->port.type!=MACH_MSG_PORT_DESCRIPTOR||ready->port.disposition!=MACH_MSG_TYPE_PORT_SEND){mach_msg_destroy(&reply.header);error=EPROTO;break;}
            shell_broker_port=ready->port.name;ready->port.name=MACH_PORT_NULL;mach_msg_destroy(&reply.header);break;
        }
    }
    mach_port_deallocate(mach_task_self(),receiver);
    mach_port_mod_refs(mach_task_self(),receiver,MACH_PORT_RIGHT_RECEIVE,-1);
    shell_pending_receiver=MACH_PORT_NULL;
    if(error)shell_broker_error=error;
    return error;
}
int alloy_shell_spawn(const char *launcher,const char *program,const char *cwd,
    char *const argv[],alloy_stop_callback stop,void *context,int *pid,void **out_control,int result_fds[3]) {
    *pid=0;*out_control=NULL;
    for(int i=0;i<3;++i)result_fds[i]=-1;
    while(pthread_mutex_trylock(&shell_broker_lock)!=0) {if(stop(context))return ECANCELED;pause_briefly();}
    int error=connect_shell_broker(launcher,stop,context);
    mach_port_t reply_port=MACH_PORT_NULL;
    char name[32];snprintf(name,sizeof(name),"/alloy-%016llx",(unsigned long long)(((uint64_t)arc4random()<<32)|arc4random()));
    int fd=-1;launch_state *state=MAP_FAILED;owned_control *control=NULL;char *data=NULL;
    if(error)goto done;
    fd=shm_open(name,O_CREAT|O_EXCL|O_RDWR,0600);if(fd<0){error=errno;goto done;}shm_unlink(name);
    if(ftruncate(fd,sizeof(*state))){error=errno;goto done;}
    state=mmap(NULL,sizeof(*state),PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
    if(state==MAP_FAILED){error=errno;goto done;}
    memset(state,0,sizeof(*state));atomic_init(&state->state,INITIAL);
    struct alloy_unique_info parent;error=identity_info(getpid(),&parent);if(error)goto done;
    state->parent_pid=getpid();state->parent_identity=parent.uniqueid;
    size_t size=strlen(cwd)+1;for(size_t i=0;argv[i];++i){size+=strlen(argv[i])+1;if(size>131072){error=E2BIG;goto done;}}
    data=malloc(size);control=calloc(1,sizeof(*control));if(control)control->state=state; if(!data||!control){error=ENOMEM;goto done;}
    size_t offset=strlen(cwd)+1;memcpy(data,cwd,offset);
    for(size_t i=0;argv[i];++i){size_t n=strlen(argv[i])+1;memcpy(data+offset,argv[i],n);offset+=n;}
    (void)program;
    error=new_receiver(&reply_port);
    if(!error)error=packet_send(shell_broker_port,reply_port,data,(uint32_t)size,&fd,1,stop,context);
    int reply[2]={0};uint32_t reply_size=0;
    if(!error)error=packet_receive(reply_port,NULL,reply,sizeof(reply),&reply_size,result_fds,3,stop,context);
    if(error)goto done;
    if(reply_size!=sizeof(reply)){error=EPROTO;goto done;}
    if(reply[0]){error=reply[0];goto done;}
    control->state=state;control->owner_pid=reply[1];control->reap_owner=0;
    for(;;) {
        unsigned phase=atomic_load_explicit(&state->state,memory_order_acquire);
        if(phase==READY) {
            if(state->pid!=reply[1] || !state->identity){error=EIO;break;}
            control->owner_identity=state->identity;
            if(stop(context)){error=ECANCELED;break;}
            unsigned expected=READY;atomic_compare_exchange_strong(&state->state,&expected,CREATE);
        } else if(phase==TARGET_READY) {
            if(stop(context)){error=ECANCELED;break;}
            unsigned expected=TARGET_READY;
            if(!atomic_compare_exchange_strong(&state->state,&expected,RUN))continue;
            *pid=state->target_pid;*out_control=control;control=NULL;state=MAP_FAILED;break;
        } else if(phase==DONE){error=state->error?state->error:EIO;break;}
        if(stop(context)){error=ECANCELED;break;}pause_briefly();
    }
done:
    if(error && state!=MAP_FAILED) {
        /* STOP before CREATE/RUN prevents an abandoned request executing later. */
        unsigned phase=atomic_load_explicit(&state->state,memory_order_acquire);
        while(phase!=DONE && phase!=STOP && !atomic_compare_exchange_weak(&state->state,&phase,STOP)){}
    }
    if(control && control->owner_pid){alloy_owned_close(control);state=MAP_FAILED;}
    else free(control);
    if(state!=MAP_FAILED)munmap(state,sizeof(*state));
    if(fd>=0)close(fd);free(data);
    if(error)for(int i=0;i<3;++i)if(result_fds[i]>=0){close(result_fds[i]);result_fds[i]=-1;}
    if(reply_port)mach_port_mod_refs(mach_task_self(),reply_port,MACH_PORT_RIGHT_RECEIVE,-1);
    pthread_mutex_unlock(&shell_broker_lock);
    return error;
}
