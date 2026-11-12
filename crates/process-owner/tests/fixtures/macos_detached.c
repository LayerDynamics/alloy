/* Bounded adversarial workload for the real native containment backend. */
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
#include <spawn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>
extern char **environ;
static int records;
static char *executable, *record_path;
static void record(void) {
    int32_t item[3] = {getpid(), getpgrp(), getsid(0)};
    if (write(records, item, sizeof(item)) != sizeof(item)) _exit(81);
}
static void hold(void) { alarm(10); record(); for (;;) pause(); }
static int spawn_leaf(short flags) {
    posix_spawnattr_t attr;
    if (posix_spawnattr_init(&attr)) return 82;
    posix_spawnattr_setflags(&attr, flags);
    if (flags & POSIX_SPAWN_SETPGROUP) posix_spawnattr_setpgroup(&attr, 0);
    char *args[] = {executable, "leaf", record_path, NULL};
    pid_t child;
    int error = posix_spawn(&child, executable, NULL, &attr, args, environ);
    posix_spawnattr_destroy(&attr);
    return error;
}
static void *create_during_cancel(void *opaque) {
    intptr_t kind = (intptr_t)opaque;
    /* Per-thread fixed resource bounds, including failed attempts. */
    for (int i = 0; i < 32; ++i) {
        int error = 0;
        if (kind) error = spawn_leaf(kind == 1 ? POSIX_SPAWN_SETPGROUP : 0x0400);
        else {
            pid_t child = fork();
            if (!child) { setsid(); hold(); }
            if (child < 0) error = errno;
        }
        if (error == EPERM) {
            int32_t denied[3] = {-getpid(), EPERM, (int32_t)kind};
            if (write(records, denied, sizeof(denied)) != sizeof(denied)) _exit(83);
            return (void *)1;
        }
        if (error) return (void *)2;
        usleep(1000);
    }
    return NULL;
}
int main(int argc, char **argv) {
    if (argc != 3) return 2;
    alarm(10); executable = argv[0]; record_path = argv[2];
    records = open(record_path, O_WRONLY | O_APPEND | O_CREAT, 0600);
    if (records < 0) return 3;
    if (!strcmp(argv[1], "leaf")) hold();
    if (!strcmp(argv[1], "protect")) {
        char path_file[4096], path[1024], alias[4096];
        snprintf(path_file,sizeof(path_file),"%s.protected-path",record_path);
        snprintf(alias,sizeof(alias),"%s.hardlink",record_path);
        int input=-1;
        for(int i=0;i<4000;i++){input=open(path_file,O_RDONLY);if(input>=0)break;usleep(1000);}
        if(input<0)return 20;
        ssize_t n=read(input,path,sizeof(path)-1);close(input);if(n<=0)return 21;path[n]=0;
        errno=0;int writable=open(path,O_WRONLY);int write_error=writable<0?errno:0;if(writable>=0)close(writable);
        errno=0;int linked=link(path,alias);int link_error=linked<0?errno:0;if(!linked)unlink(alias);
        int32_t result[3]={-getpid(),write_error,7};if(write(records,result,sizeof(result))!=sizeof(result))return 22;
        result[1]=link_error;result[2]=8;if(write(records,result,sizeof(result))!=sizeof(result))return 23;
        errno=0;linked=link(record_path,alias);int ordinary_error=linked<0?errno:0;if(!linked)unlink(alias);
        result[1]=ordinary_error;result[2]=9;if(write(records,result,sizeof(result))!=sizeof(result))return 24;
    }
    int ready[2]; if (pipe(ready)) return 4;
    pid_t child = fork();
    if (!child) {
        close(ready[0]); if (setsid() < 0) _exit(5);
        pid_t detached = fork(); if (detached < 0) _exit(6); if (detached) _exit(0);
        alarm(15); record(); char byte = 1; if (write(ready[1], &byte, 1) != 1) _exit(7);
        close(ready[1]); for (;;) pause();
    }
    if (child < 0) return 8;
    close(ready[1]); char byte; if (read(ready[0], &byte, 1) != 1) return 9;
    close(ready[0]); int status; while (waitpid(child, &status, 0) < 0 && errno == EINTR) {}
    child = fork();
    if (!child) { if (setpgid(0, 0)) _exit(10); hold(); }
    if (child < 0) return 11;
    if (!strcmp(argv[1], "detach-exit")) return 23;
    if (!strcmp(argv[1], "storm")) {
        pthread_t workers[3];
        for (intptr_t i = 0; i < 3; ++i) if (pthread_create(&workers[i], NULL, create_during_cancel, (void *)i)) return 12;
        for (int i = 0; i < 3; ++i) { void *value; pthread_join(workers[i], &value); }
    }
    for (;;) pause();
}
