#ifndef ALLOY_OWNED_PROCESS_H
#define ALLOY_OWNED_PROCESS_H
#include <spawn.h>
#include <stdint.h>
#include <stddef.h>
typedef int (*alloy_stop_callback)(void *);
int alloy_owned_spawn(const char *launcher, const char *program,
    posix_spawn_file_actions_t *actions, posix_spawnattr_t *attr,
    char *const argv[], char *const envp[], alloy_stop_callback stop, void *context, int reap_owner,
    int *pid, uint64_t *identity, void **control);
int alloy_owned_signal(int pid, uint64_t identity);
int alloy_owned_alive(int pid, uint64_t identity);
uint64_t alloy_process_identity(int pid);
int alloy_broker_registered(const char *name);
int alloy_launchd_job(const char *name, const void *plist, size_t size, char *diagnostic, size_t capacity);
int alloy_owned_stop(void *control);
int alloy_owned_wait(void *control, int *status, int nohang);
void alloy_owned_close(void *control);
int alloy_exec_helper(int argc, char **argv);
int alloy_shell_spawn(const char *launcher, const char *service, const char *program, const char *cwd, const int io[3],
    char *const argv[], alloy_stop_callback stop, void *context,
    int *pid, void **control, int result_fds[3]);

#endif
