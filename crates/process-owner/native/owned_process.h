#ifndef ALLOY_OWNED_PROCESS_H
#define ALLOY_OWNED_PROCESS_H
#include <spawn.h>
#include <stdint.h>
typedef int (*alloy_stop_callback)(void *);
int alloy_owned_spawn(const char *launcher, const char *program,
    posix_spawn_file_actions_t *actions, posix_spawnattr_t *attr,
    char *const argv[], char *const envp[], alloy_stop_callback stop, void *context, int reap_owner,
    int *pid, uint64_t *identity, void **control);
int alloy_owned_signal(int pid, uint64_t identity);
int alloy_owned_alive(int pid, uint64_t identity);
int alloy_owned_stop(void *control);
int alloy_owned_wait(void *control, int *status, int nohang);
void alloy_owned_close(void *control);
int alloy_exec_helper(int argc, char **argv);
#endif
