/* Security.framework can open executable files without atomic CLOEXEC. Keep
 * its full dynamic signature check outside the multithreaded embedding host.
 * One host-lifetime verifier receives only an anonymous inherited mapping.
 * The host never waitpids it. A failed verifier is not replaced: at most one
 * infrastructure child/status can remain charged until host exit, even with
 * arbitrary external reapers. Broker and workload owners remain under launchd. */
enum { VERIFY_INITIAL, VERIFY_READY, VERIFY_REQUEST, VERIFY_REPLY };
typedef struct {
    _Atomic unsigned phase;
    int parent, pid, result;
    uint64_t parent_identity, identity;
    audit_token_t audit;
#ifdef ALLOY_OWNED_PROCESS_TEST
    _Atomic int pause, paused, unknown_parent;
#endif
} verification_state;
static verification_state *host_verifier;
static int verifier_failed;
static int verifier_expected_pid;
static audit_token_t verifier_last_valid_audit;

static void *verification_parent_watch(void *opaque) {
    verification_state *state=opaque;
    for(;;){
#ifdef ALLOY_OWNED_PROCESS_TEST
        if(atomic_load(&state->unknown_parent))atomic_store(&test_unknown_identity_pid,state->parent);
#endif
        /* Exiting our own infrastructure is safe on uncertainty; retaining an
         * unverified dead parent would leak one verifier per historical host. */
        if(alloy_owned_alive(state->parent,state->parent_identity)!=1)_exit(0);
        struct timespec idle={0,10000000};nanosleep(&idle,NULL);
    }
}
static int verification_service(int argc,char **argv) {
    (void)argc;
    if(strcmp(argv[3],"3"))return 125;
    struct stat st;
    if(fstat(3,&st) || st.st_size<(off_t)sizeof(verification_state))return 125;
    verification_state *state=mmap(NULL,sizeof(*state),PROT_READ|PROT_WRITE,MAP_SHARED,3,0);
    close(3);if(state==MAP_FAILED)return 125;
    if(alloy_owned_alive(state->parent,state->parent_identity)!=1)return 125;
    pthread_t watchdog;
    if(pthread_create(&watchdog,NULL,verification_parent_watch,state))return 125;
    state->pid=getpid();state->identity=alloy_process_identity(getpid());
    unsigned expected=VERIFY_INITIAL;
    if(!state->identity || !atomic_compare_exchange_strong_explicit(&state->phase,&expected,VERIFY_READY,memory_order_release,memory_order_relaxed))return 125;
    for(;;){
        if(atomic_load_explicit(&state->phase,memory_order_acquire)==VERIFY_REQUEST){
#ifdef ALLOY_OWNED_PROCESS_TEST
            while(atomic_load(&state->pause)){
                atomic_store(&state->paused,1);pause_briefly();
            }
#endif
            state->result=authenticate_broker(state->audit,argv[2]);
            atomic_store_explicit(&state->phase,VERIFY_REPLY,memory_order_release);
        }
        pause_briefly();
    }
}

/* Caller holds shell_broker_lock. The anonymous mapping is never rebound to a
 * replacement PID; neither PID reuse nor a stale reply can authenticate a peer. */
static int authenticate_in_verifier(audit_token_t audit,const char *launcher,const char *service,
                                    alloy_stop_callback stop,void *context) {
    (void)service;
    if(verifier_failed)return EIO;
    if(!host_verifier){
        char name[32];snprintf(name,sizeof(name),"/av-%016llx%08x",
            (unsigned long long)(((uint64_t)arc4random()<<32)|arc4random()),
            arc4random());
        int fd=shm_open(name,O_CREAT|O_EXCL|O_RDWR,0600);
        if(fd<0)return errno;
        /* shm_open is atomically CLOEXEC. Unlink before any helper can run. */
        if(shm_unlink(name)){int error=errno;close(fd);return error;}
        if(fd<4){int copy=fcntl(fd,F_DUPFD_CLOEXEC,4);int error=errno;close(fd);fd=copy;if(fd<0)return error;}
        if(ftruncate(fd,sizeof(verification_state))){int error=errno;close(fd);return error;}
        host_verifier=mmap(NULL,sizeof(*host_verifier),PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
        if(host_verifier==MAP_FAILED){int error=errno;host_verifier=NULL;close(fd);return error;}
        memset(host_verifier,0,sizeof(*host_verifier));
        host_verifier->parent=getpid();host_verifier->parent_identity=alloy_process_identity(getpid());
        posix_spawn_file_actions_t actions;posix_spawnattr_t attr;
        int actions_ready=0,attr_ready=0,pid=0;
        int error=posix_spawn_file_actions_init(&actions);
        if(!error){actions_ready=1;error=posix_spawnattr_init(&attr);}
        if(!error){attr_ready=1;error=posix_spawnattr_setflags(&attr,
            POSIX_SPAWN_CLOEXEC_DEFAULT|POSIX_SPAWN_SETPGROUP|POSIX_SPAWN_SETSIGDEF|POSIX_SPAWN_SETSIGMASK);}
        /* A stopped watchdog cannot observe parent death. Keep the verifier in
         * its own group in the parent's session, with unblocked default HUP:
         * Darwin then tears it down when that stopped group becomes orphaned. */
        sigset_t defaults,mask;sigemptyset(&defaults);sigaddset(&defaults,SIGHUP);sigemptyset(&mask);
        if(!error)error=posix_spawnattr_setpgroup(&attr,0);
        if(!error)error=posix_spawnattr_setsigdefault(&attr,&defaults);
        if(!error)error=posix_spawnattr_setsigmask(&attr,&mask);
        if(!error)error=posix_spawn_file_actions_adddup2(&actions,fd,3);
        for(int i=0;!error && i<3;++i)
            error=posix_spawn_file_actions_addopen(&actions,i,"/dev/null",i?O_WRONLY:O_RDONLY,0);
        char *arguments[]={(char *)launcher,"--alloy-verification-service-v1",(char *)launcher,"3",NULL};
        char *environment[]={NULL};
        if(!error)error=posix_spawn(&pid,launcher,&actions,&attr,arguments,environment);
        if(attr_ready)posix_spawnattr_destroy(&attr);
        if(actions_ready)posix_spawn_file_actions_destroy(&actions);
        close(fd);
        /* A failed startup is retained, never retried with another child. */
        if(error){verifier_failed=1;return error;}
        verifier_expected_pid=pid;
        uint64_t until=milliseconds()+5000;
        while(atomic_load_explicit(&host_verifier->phase,memory_order_acquire)==VERIFY_INITIAL){
            if(stop(context))return ECANCELED;
            if(milliseconds()>=until)return ETIMEDOUT;
            pause_briefly();
        }
        if(host_verifier->pid!=pid || !host_verifier->identity){verifier_failed=1;return EPERM;}
    }
    uint64_t until=milliseconds()+5000;
    while(atomic_load_explicit(&host_verifier->phase,memory_order_acquire)!=VERIFY_READY){
        /* Discard a cancelled previous verification's late verdict. */
        if(atomic_load_explicit(&host_verifier->phase,memory_order_acquire)==VERIFY_REPLY){
            atomic_store(&host_verifier->phase,VERIFY_READY);break;
        }
        if(stop(context))return ECANCELED;
        if(milliseconds()>=until)return ETIMEDOUT;
        pause_briefly();
    }
    if(host_verifier->pid!=verifier_expected_pid || !host_verifier->identity){verifier_failed=1;return EPERM;}
    if(alloy_owned_alive(host_verifier->pid,host_verifier->identity)!=1){verifier_failed=1;return EIO;}
    host_verifier->audit=audit;
    atomic_store_explicit(&host_verifier->phase,VERIFY_REQUEST,memory_order_release);
    while(atomic_load_explicit(&host_verifier->phase,memory_order_acquire)!=VERIFY_REPLY){
        if(stop(context))return ECANCELED;
        if(milliseconds()>=until)return ETIMEDOUT;
        if(alloy_owned_alive(host_verifier->pid,host_verifier->identity)==0){verifier_failed=1;return EIO;}
        pause_briefly();
    }
    int result=host_verifier->result;
    if(!result)verifier_last_valid_audit=audit;
    atomic_store_explicit(&host_verifier->phase,VERIFY_READY,memory_order_release);
    return result;
}

#ifdef ALLOY_OWNED_PROCESS_TEST
int alloy_test_verifier_pid(void){return host_verifier?host_verifier->pid:0;}
int alloy_test_verifier_unknown_parent(void){
    if(!host_verifier)return ENOENT;
    atomic_store(&host_verifier->unknown_parent,1);return 0;
}
int alloy_test_verifier_alive(void){
    return host_verifier?alloy_owned_alive(host_verifier->pid,host_verifier->identity):-ENOENT;
}
void alloy_test_ignore_block_hup(void){
    signal(SIGHUP,SIG_IGN);sigset_t mask;sigemptyset(&mask);sigaddset(&mask,SIGHUP);
    pthread_sigmask(SIG_BLOCK,&mask,NULL);
}
int alloy_test_verifier_pause(void){
    if(!host_verifier)return ENOENT;
    int error=identity_signal(host_verifier->pid,host_verifier->identity,SIGSTOP);
    uint64_t until=milliseconds()+2000;
    while(!error){
        struct proc_bsdinfo state;
        if(proc_pidinfo(host_verifier->pid,PROC_PIDTBSDINFO,0,&state,sizeof(state))==sizeof(state) && state.pbi_status==SSTOP)return 0;
        if(milliseconds()>=until){(void)identity_signal(host_verifier->pid,host_verifier->identity,SIGCONT);return ETIMEDOUT;}
        pause_briefly();
    }
    return error;
}
int alloy_test_verifier_kill(void){
    return host_verifier?alloy_owned_signal(host_verifier->pid,host_verifier->identity):ESRCH;
}
static int verifier_test_running(void *context){(void)context;return 0;}
int alloy_test_verifier_reject_self(void){
    audit_token_t audit;mach_msg_type_number_t count=TASK_AUDIT_TOKEN_COUNT;
    if(task_info(mach_task_self(),TASK_AUDIT_TOKEN,(task_info_t)&audit,&count)!=KERN_SUCCESS)return EIO;
    return host_verifier?authenticate_in_verifier(audit,"","",verifier_test_running,NULL):ENOENT;
}
static int verifier_test_deadline(void *context){return milliseconds()>=*(uint64_t *)context;}
int alloy_test_verifier_late_allow(void){
    if(!host_verifier)return ENOENT;
    audit_token_t good=verifier_last_valid_audit; /* Last authenticated real broker. */
    atomic_store(&host_verifier->pause,1);
    uint64_t until=milliseconds()+100;
    int error=authenticate_in_verifier(good,"","",verifier_test_deadline,&until);
    int paused=atomic_load(&host_verifier->paused);
    atomic_store(&host_verifier->pause,0);
    if(error!=ECANCELED || !paused)return EPROTO;
    /* Its delayed allow must not be returned for this unrelated process. */
    return alloy_test_verifier_reject_self()==EPERM?0:EPROTO;
}
#endif
