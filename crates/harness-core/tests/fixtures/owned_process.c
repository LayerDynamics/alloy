/* Native integration checks for the production identity-safe signal path. */
#include "owned_process.h"
#include <assert.h>
#include <errno.h>
#include <signal.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <libproc.h>
#include <sys/proc.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>
int alloy_test_group_live(int group);
static int never_stop(void *unused) { (void)unused; return 0; }
static void launch(const char *launcher, const char *program, char **args, int *pid, uint64_t *identity, void **control) {
    posix_spawnattr_t attr;
    posix_spawn_file_actions_t actions;
    assert(!posix_spawnattr_init(&attr));
    assert(!posix_spawn_file_actions_init(&actions));
    assert(!posix_spawnattr_setflags(&attr, POSIX_SPAWN_CLOEXEC_DEFAULT | POSIX_SPAWN_SETSID));
    for (int fd = 0; fd < 3; ++fd) assert(!posix_spawn_file_actions_addinherit_np(&actions, fd));
    char *environment[] = {NULL};
    int error = alloy_owned_spawn(launcher, program, &actions, &attr, args, environment,
                                 never_stop, NULL, 0, pid, identity, control);
    if (error) fprintf(stderr, "owned spawn: %s\n", strerror(error));
    assert(!error);
    assert(*pid > 0 && *identity > 0);
    posix_spawn_file_actions_destroy(&actions);
    posix_spawnattr_destroy(&attr);
}
static void gone(int pid, uint64_t identity) {
    for (int attempt = 0; attempt < 2000; ++attempt) {
        if (alloy_owned_alive(pid, identity) == 0) return;
        usleep(1000);
    }
    assert(!"original identity remained alive");
}
static _Atomic int reaping = 1;
static void *reaper(void *unused) {
    (void)unused;
    while (atomic_load(&reaping)) {
        while (waitpid(-1, NULL, WNOHANG) > 0) {}
        usleep(100);
    }
    return NULL;
}
static void process_stopped(int pid) {
    for (int i = 0; i < 2000; ++i) {
        struct proc_bsdinfo info;
        int size = proc_pidinfo(pid, PROC_PIDTBSDINFO, 0, &info, sizeof(info));
        if ((size == 0 && errno == ESRCH) || (size == sizeof(info) && info.pbi_status == SZOMB)) return;
        usleep(1000);
    }
    assert(!"trusted subprocess survived group cleanup");
}
int main(int argc, char **argv) {
    if (argc == 4 && !strcmp(argv[1], "--tree")) {
        int descendant = fork();
        assert(descendant >= 0);
        if (descendant == 0) { for (;;) pause(); }
        FILE *out = fopen(argv[2], "w");
        assert(out);
        fprintf(out, "%d\n", descendant);
        fclose(out);
        if (!strcmp(argv[3], "exit")) return 0;
        for (;;) pause();
    }
    assert(argc == 2);
    /* Verify libproc count handling with a zombie leader and one live member,
     * before starting the competing reaper. The zombie pins this test group. */
    int fds[2];
    assert(!pipe(fds));
    pid_t leader = fork();
    assert(leader >= 0);
    if (!leader) {
        close(fds[0]);
        assert(setsid() == getpid());
        pid_t member = fork();
        assert(member >= 0);
        if (!member) { close(fds[1]); for (;;) pause(); }
        assert(write(fds[1], &member, sizeof(member)) == sizeof(member));
        _exit(0);
    }
    close(fds[1]);
    pid_t member;
    assert(read(fds[0], &member, sizeof(member)) == sizeof(member));
    close(fds[0]);
    siginfo_t exit_info = {0};
    assert(!waitid(P_PID, leader, &exit_info, WEXITED | WNOWAIT));
    assert(alloy_test_group_live(leader) == 1);
    assert(!kill(-leader, SIGKILL));
    process_stopped(member);
    assert(alloy_test_group_live(leader) == 0);
    assert(waitpid(leader, NULL, 0) == leader);
    pthread_t thread;
    assert(!pthread_create(&thread, NULL, reaper, NULL));
    const char *launcher = argv[1];
    for (int iteration = 0; iteration < 32; ++iteration) {
        int pid;
        uint64_t identity;
        void *control;
        char *args[] = {"/bin/sh", "-c", "exec /bin/sleep 30", NULL};
        launch(launcher, args[0], args, &pid, &identity, &control);
        /* Stable identity survives sh -> sleep exec. */
        usleep(10000);
        assert(alloy_owned_alive(pid, identity) == 1);
        assert(alloy_test_group_live(pid) == 1);
        assert(!alloy_owned_signal(pid, identity + 1));
        assert(alloy_owned_alive(pid, identity) == 1);
        assert(!alloy_owned_signal(pid, identity));
        int status;
        int wait_error = alloy_owned_wait(control, &status, 0);
        if (wait_error) fprintf(stderr, "owned wait: %s (%d)\n", strerror(wait_error), wait_error);
        assert(!wait_error);
        alloy_owned_close(control);
        assert(WIFSIGNALED(status) && WTERMSIG(status) == SIGKILL);
        gone(pid, identity);
        /* Deterministic PID-reuse model: a live different identity at the stale
         * numeric lookup key must survive. This does not claim actual PID wrap. */
        int sentinel;
        uint64_t sentinel_identity;
        void *sentinel_control;
        char *sleep_args[] = {"/bin/sleep", "30", NULL};
        launch(launcher, sleep_args[0], sleep_args, &sentinel, &sentinel_identity, &sentinel_control);
        assert(identity != sentinel_identity);
        assert(!alloy_owned_signal(sentinel, identity));
        assert(alloy_owned_alive(sentinel, sentinel_identity) == 1);
        /* A competing reaper already consumed the original. */
        assert(!alloy_owned_signal(pid, identity));
        assert(!alloy_owned_signal(sentinel, sentinel_identity));
        assert(!alloy_owned_wait(sentinel_control, &status, 0));
        alloy_owned_close(sentinel_control);
        gone(sentinel, sentinel_identity);
    }
    /* Group cleanup must also work after the trusted leader exits naturally,
     * while the daemon concurrently reaps every direct child it can find. */
    for (int mode = 0; mode < 2; ++mode) {
        char marker[] = "/tmp/alloy-owned-tree-XXXXXX";
        int fd = mkstemp(marker);
        assert(fd >= 0);
        close(fd);
        int pid, descendant = 0, status;
        uint64_t identity;
        void *control;
        char *args[] = {argv[0], "--tree", marker, mode ? "exit" : "wait", NULL};
        launch(launcher, args[0], args, &pid, &identity, &control);
        for (int i = 0; i < 2000 && !descendant; ++i) {
            FILE *in = fopen(marker, "r");
            assert(in);
            (void)fscanf(in, "%d", &descendant);
            fclose(in);
            usleep(1000);
        }
        unlink(marker);
        assert(descendant > 0);
        if (!mode) assert(!alloy_owned_stop(control));
        assert(!alloy_owned_wait(control, &status, 0));
        alloy_owned_close(control);
        process_stopped(descendant);
        gone(pid, identity);
    }
    atomic_store(&reaping, 0);
    assert(!pthread_join(thread, NULL));
    struct sigaction action = {0};
    action.sa_handler = SIG_IGN;
    sigemptyset(&action.sa_mask);
    assert(!sigaction(SIGCHLD, &action, NULL));
    for (int iteration = 0; iteration < 32; ++iteration) {
        int pid;
        uint64_t identity;
        void *control;
        char *args[] = {"/bin/sleep", "30", NULL};
        launch(launcher, args[0], args, &pid, &identity, &control);
        assert(!alloy_owned_signal(pid, identity));
        gone(pid, identity);
        alloy_owned_close(control);
        assert(!alloy_owned_signal(pid, identity));
    }
    puts("OWNED_IDENTITY_EXEC_REAP_AUTOREAP_PASSED");
    return 0;
}
