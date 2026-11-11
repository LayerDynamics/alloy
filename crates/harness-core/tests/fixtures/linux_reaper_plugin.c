#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
int main(int argc, char **argv) {
    if (argc != 3) return 2;
    struct sigaction action;
    if (sigaction(SIGCHLD, NULL, &action)) return 7;
    int ignored = action.sa_handler == SIG_IGN;
    if (sigaction(SIGABRT, NULL, &action)) return 8;
    int abrt_ignored = action.sa_handler == SIG_IGN;
    pid_t child = fork();
    if (child < 0) return 3;
    if (!child) { execl("/bin/sleep", "sleep", "120", (char *)0); _exit(4); }
    FILE *f = fopen(argv[1], "w"); if (!f) return 5;
    fprintf(f, "%ld %ld %d %d\n", (long)getpid(), (long)child, ignored, abrt_ignored); fclose(f);
    char line[32768];
    while (fgets(line, sizeof(line), stdin)) {
        char *p = line, *last = NULL;
        while ((p = strstr(p, "\"id\":"))) { last = p; p += 5; }
        if (!last) return 6;
        long id = strtol(last + 5, NULL, 10);
        printf("{\"jsonrpc\":\"2.0\",\"id\":%ld,\"result\":{\"protocol_version\":1,\"plugin_id\":\"linux\",\"plugin_version\":\"1.0.0\",\"hooks\":[]}}\n", id);
        fflush(stdout);
        if (!strcmp(argv[2], "exit")) return 0;
    }
    return 0;
}
