#define _GNU_SOURCE
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
int alloy_linux_launch(const char *, char *const [], const char *, int[5]);
int alloy_linux_wait_pidfd(int);
int main(int argc, char **argv) {
    if (argc != 2) return 10;
    char *target[] = {"/usr/bin/touch", argv[1], NULL};
    int fds[5] = {-1,-1,-1,-1,-1};
    if (alloy_linux_launch(target[0], target, "/", fds)) return 11;
    close(fds[0]); close(fds[1]); close(fds[2]);
    close(fds[3]); /* Cancel after identity publication, before RUN. */
    int result = alloy_linux_wait_pidfd(fds[4]); close(fds[4]);
    if (result || access(argv[1], F_OK) == 0) return 14;
    puts("LINUX_PRECOMMIT_CANCEL_ZERO_EFFECT");
    return 0;
}
