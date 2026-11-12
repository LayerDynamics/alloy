/* Included after the authenticated Mach transport; one launchd job per cohort. */
typedef struct { uint64_t until; int peer; uint64_t identity; } cohort_deadline;
static int cohort_interrupted(void *opaque) {
    cohort_deadline *deadline = opaque;
    return milliseconds() >= deadline->until ||
        (deadline->identity && alloy_owned_alive(deadline->peer, deadline->identity) != 1);
}
static char *cohort_xml(const char *text) {
    size_t length = strlen(text);
    if (length > 16384) return NULL;
    char *out = malloc(length * 6 + 1), *at = out;
    if (!out) return NULL;
    for (const unsigned char *p = (const unsigned char *)text; *p; ++p) {
        const char *escape = NULL;
        if (*p < 32) { free(out); return NULL; }
        switch (*p) {
            case '&': escape = "&amp;"; break;
            case '<': escape = "&lt;"; break;
            case '>': escape = "&gt;"; break;
            case '"': escape = "&quot;"; break;
            case '\'': escape = "&apos;"; break;
        }
        if (escape) { size_t n = strlen(escape); memcpy(at, escape, n); at += n; }
        else *at++ = (char)*p;
    }
    *at = 0;
    return out;
}
static int start_cohort(const char *launcher, const char *directory, const void *data, uint32_t size,
                        const int incoming[4], int incoming_count, int *child, int exported[3]) {
    int statefd=incoming[0];
    *child = 0;
    struct stat st;
    /* Darwin rounds POSIX shared-memory objects to its page size. */
    if (fstat(statefd, &st) || st.st_size < (off_t)sizeof(launch_state)) return EINVAL;
    launch_state *state = mmap(NULL, sizeof(*state), PROT_READ | PROT_WRITE, MAP_SHARED, statefd, 0);
    if (state == MAP_FAILED) return errno;
    char *path = cohort_xml(launcher), *definition = NULL;
    if (!path) { munmap(state, sizeof(*state)); return EINVAL; }
    uint64_t random[2]; arc4random_buf(random, sizeof(random));
    snprintf(state->cohort_job, sizeof(state->cohort_job),
        "com.layerdynamics.alloy.cohort.%u.%d.%llu.%016llx%016llx", getuid(),
        state->parent_pid, (unsigned long long)state->parent_identity,
        (unsigned long long)random[0], (unsigned long long)random[1]);
    int registry_lease=-1;
    int record_error=alloy_cohort_record_create(directory,state,&registry_lease);
    if(record_error){free(path);munmap(state,sizeof(*state));return record_error;}
    char label[sizeof(state->cohort_job)]; memcpy(label, state->cohort_job, sizeof(label));
    int n = asprintf(&definition,
        "<?xml version=\"1.0\"?><plist version=\"1.0\"><dict>"
        "<key>Label</key><string>%s</string><key>ProgramArguments</key><array>"
        "<string>/usr/bin/env</string><string>-i</string><string>%s</string>"
        "<string>--alloy-cohort-service-v1</string><string>%d</string><string>%llu</string><string>%s</string><string>%s</string>"
        "</array><key>MachServices</key><dict><key>%s</key><true/></dict>"
        "<key>LimitLoadToSessionType</key><array><string>Aqua</string><string>Background</string></array>"
        "<key>KeepAlive</key><false/><key>RunAtLoad</key><true/>"
        "<key>ProcessType</key><string>Standard</string></dict></plist>",
        label, path, getpid(), (unsigned long long)alloy_process_identity(getpid()), label, state->cohort_record_path, label);
    free(path);
    char diagnostic[2048];
    int error=n<0?ENOMEM:0;
#ifdef ALLOY_OWNED_PROCESS_TEST
    if(atomic_load(&state->test_submit_pause)){
        atomic_store(&state->test_submit_paused,1);
        uint64_t barrier=milliseconds()+5000;
        while(atomic_load(&state->test_submit_pause) && milliseconds()<barrier &&
            alloy_owned_alive(state->parent_pid,state->parent_identity)==1)pause_briefly();
    }
#endif
    if(!error && (atomic_load(&state->state)==STOP || alloy_owned_alive(state->parent_pid,state->parent_identity)!=1))error=ECANCELED;
    if(!error)error=alloy_cohort_record_submitting(state->cohort_record_path);
    if(!error)error=alloy_launchd_job(label, definition, (size_t)n, diagnostic, sizeof(diagnostic));
    else (void)alloy_cohort_record_complete(state->cohort_record_path,0);
    close(registry_lease);
    free(definition);
    mach_port_t endpoint = MACH_PORT_NULL, receiver = MACH_PORT_NULL;
    cohort_deadline deadline = { milliseconds() + 5000, 0, 0 };
    if (!error) error = new_receiver(&receiver);
    if (!error) {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
        if (bootstrap_look_up(bootstrap_port, label, &endpoint) != KERN_SUCCESS) error = ENOENT;
#pragma clang diagnostic pop
    }
    uint64_t generation[2]; arc4random_buf(generation, sizeof(generation));
    if (!error) error = packet_send(endpoint, receiver, SHELL_MESSAGE, generation,
        data, size, incoming, incoming_count, cohort_interrupted, &deadline);
    int reply[2] = {0}, kind = 0; uint32_t reply_size = 0;
    uint64_t received_generation[2]; audit_token_t audit;
    if (!error) error = packet_receive(receiver, NULL, &kind, received_generation, &audit,
        reply, sizeof(reply), &reply_size, exported, 3, cohort_interrupted, &deadline);
    if (!error && (kind != SHELL_MESSAGE || reply_size != sizeof(reply) ||
        memcmp(generation, received_generation, sizeof(generation)))) error = EPROTO;
    if (!error) error = authenticate_broker(audit, launcher);
    if (!error && (!reply[1] || reply[1] != audit_token_to_pid(audit))) error = EPROTO;
    if (!error) { *child = reply[1]; error = reply[0]; }
    if (endpoint) mach_port_deallocate(mach_task_self(), endpoint);
    if (receiver) mach_port_mod_refs(mach_task_self(), receiver, MACH_PORT_RIGHT_RECEIVE, -1);
    /* Do not remove a job that may have accepted its state: it owns cleanup
     * independently of this broker. A never-connected job times out itself. */
    munmap(state, sizeof(*state));
    return error;
}
static int cohort_service(int argc, char **argv) {
    (void)argc;
    int parent = (int)strtol(argv[2], NULL, 10);
    uint64_t identity = strtoull(argv[3], NULL, 10);
    const char *label = argv[4];
    if (parent <= 0 || !identity || strlen(label) >= 256 ||
        strncmp(label, "com.layerdynamics.alloy.cohort.", sizeof("com.layerdynamics.alloy.cohort.") - 1)) return 125;
    struct sigaction action = {0}; action.sa_handler = SIG_DFL; sigemptyset(&action.sa_mask);
    if (sigaction(SIGCHLD, &action, NULL)) return 125;
    mach_port_t receiver = MACH_PORT_NULL, reply_port = MACH_PORT_NULL;
    int incoming[4]={-1,-1,-1,-1};
    int error = 0, statefd = -1, kind = 0, endpoints[6] = {-1,-1,-1,-1,-1,-1};
    alloy_cohort_record *record=NULL;
    launch_state *state = MAP_FAILED; char **strings = NULL, **arguments = NULL;
    char data[131072]; uint32_t size = 0; uint64_t generation[2]; audit_token_t audit;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
    if (bootstrap_check_in(bootstrap_port, label, &receiver) != KERN_SUCCESS) error = EIO;
#pragma clang diagnostic pop
    cohort_deadline deadline = {milliseconds() + 5000, parent, identity};
    if (!error) error = packet_receive(receiver, &reply_port, &kind, generation, &audit,
        data, sizeof(data), &size, incoming, -4, cohort_interrupted, &deadline);
    statefd=incoming[0];
    if (!error) error = authenticate_host(audit, parent, identity);
    if (!error && kind != SHELL_MESSAGE) error = EPROTO;
    struct stat st;
    if (!error && (fstat(statefd, &st) || st.st_size < (off_t)sizeof(*state))) error = EINVAL;
    if (!error) {
        state = mmap(NULL, sizeof(*state), PROT_READ | PROT_WRITE, MAP_SHARED, statefd, 0);
        if (state == MAP_FAILED) error = errno;
    }
    if (!error && (strcmp(state->cohort_job, label)||strcmp(state->cohort_record_path,argv[5]))) error = EPERM;
    if (!error) error=alloy_cohort_record_attach(state,&record);
    size_t count = 0, offset = 0;
    if (!error && !(strings = calloc((size_t)size + 1, sizeof(char *)))) error = ENOMEM;
    while (!error && offset < size) {
        char *end = memchr(data + offset, 0, size - offset);
        if (!end) { error = EINVAL; break; }
        strings[count++] = data + offset; offset = (size_t)(end - data) + 1;
    }
    if (!error && (count < 3 || (strcmp(strings[0], "shell") && strcmp(strings[0], "plugin")))) error = EINVAL;
    if (!error && chdir(strings[1])) error = errno;
    int plugin=!error&&!strcmp(strings[0],"plugin");
    if (!error && plugin) {
        if(incoming[1]<0 || incoming[2]<0 || incoming[3]<0)error=EINVAL;
        else {endpoints[0]=incoming[1];endpoints[3]=incoming[2];endpoints[5]=incoming[3];}
    } else if(!error) {
        if(incoming[1]>=0)error=EINVAL;
        else if(pipe(endpoints)||pipe(endpoints+2)||pipe(endpoints+4))error=errno;
    }
    if (!error && !(arguments = calloc(count + 1, sizeof(char *)))) error = ENOMEM;
    if (!error) {
        arguments[0] = argv[0]; arguments[1] = "--alloy-owned-exec-v1";
        for (size_t i = 2; i < count; ++i) arguments[i] = strings[i];
    }
    int reply[2] = {error, getpid()}, exported[3] = {endpoints[1], endpoints[2], endpoints[4]};
    if (reply_port) {
        if (error || plugin) for (int i = 0; i < 3; ++i) exported[i] = open("/dev/null", O_RDWR | O_CLOEXEC);
        int sent = packet_send(reply_port, MACH_PORT_NULL, SHELL_MESSAGE, generation,
            reply, sizeof(reply), exported, 3, NULL, NULL);
        if (error || plugin) for (int i = 0; i < 3; ++i) if (exported[i] >= 0) close(exported[i]);
        if (!error) error = sent;
        mach_port_deallocate(mach_task_self(), reply_port);
    }
    if (!error) {
        if(endpoints[1]>=0)close(endpoints[1]); endpoints[1] = -1;
        if(endpoints[2]>=0)close(endpoints[2]); endpoints[2] = -1;
        if(endpoints[4]>=0)close(endpoints[4]); endpoints[4] = -1;
        int input = !strcmp(strings[0], "shell") ? open("/dev/null", O_RDONLY | O_CLOEXEC) : endpoints[0];
        int io[3] = {input, endpoints[3], endpoints[5]};
        error = input < 0 ? errno : alloy_containment_run(state, (int)count, arguments, io, record);
        if (input != endpoints[0] && input >= 0) close(input);
    }
    /* Startup failure cannot have run a target. A failed cleanup keeps its
     * non-DONE state; callers retain the ownership slot and report failure. */
    if (state != MAP_FAILED && error && !state->pid) {
        if(record){atomic_store_explicit(&record->complete,1,memory_order_release);(void)msync(record,sizeof(*record),MS_SYNC);}
        state->error = error;
        atomic_store_explicit(&state->state, DONE, memory_order_release);
    }
    for(int i=1;i<4;++i)if(incoming[i]>=0 && incoming[i]!=endpoints[0] && incoming[i]!=endpoints[3] && incoming[i]!=endpoints[5])close(incoming[i]);
    for (int i = 0; i < 6; ++i) if (endpoints[i] >= 0) close(endpoints[i]);
    if(record)munmap(record,sizeof(*record));
    if (statefd >= 0) close(statefd);
    if (state != MAP_FAILED) munmap(state, sizeof(*state));
    free(strings); free(arguments);
    if (receiver) mach_port_mod_refs(mach_task_self(), receiver, MACH_PORT_RIGHT_RECEIVE, -1);
    char diagnostic[2048];
    if(state==MAP_FAILED)(void)alloy_cohort_record_complete(argv[5],0);
    (void)alloy_cohort_record_retire(argv[5]);
    (void)alloy_launchd_job(label, NULL, 0, diagnostic, sizeof(diagnostic));
    return error ? 125 : 0;
}
