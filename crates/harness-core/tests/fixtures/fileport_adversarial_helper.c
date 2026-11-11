#include <fcntl.h>
#include <mach/mach.h>
#include <mach/task_special_ports.h>
#include <stdio.h>
#include <string.h>
#include <sys/fileport.h>
#include <sys/wait.h>
#include <unistd.h>

enum { COUNT = 6, MESSAGE_ID = 0x41504c59 };
typedef struct {
    mach_msg_header_t header;
    mach_msg_body_t body;
    mach_msg_port_descriptor_t ports[COUNT];
} message_t;

static int send_message(void) {
    mach_port_t destination = MACH_PORT_NULL;
    if (task_get_special_port(mach_task_self(), TASK_BOOTSTRAP_PORT, &destination) != KERN_SUCCESS)
        return 1;
    message_t message;
    memset(&message, 0, sizeof(message));
    message.header.msgh_bits = MACH_MSGH_BITS(MACH_MSG_TYPE_COPY_SEND, 0) | MACH_MSGH_BITS_COMPLEX;
    message.header.msgh_size = sizeof(message);
    message.header.msgh_remote_port = destination;
    message.header.msgh_id = MESSAGE_ID;
    message.body.msgh_descriptor_count = COUNT;
    int endpoints[COUNT];
    for (int i = 0; i < COUNT; i += 2)
        if (pipe(&endpoints[i]) != 0) return 2;
#if TEST_MODE == 2
    /* The fourth port imports successfully but is not a pipe. */
    close(endpoints[3]);
    endpoints[3] = open("/dev/null", O_WRONLY);
    if (endpoints[3] < 0) return 3;
#endif
    for (int i = 0; i < COUNT; ++i) {
        fileport_t port = MACH_PORT_NULL;
        if (fileport_makeport(endpoints[i], &port) != 0) return 4;
        message.ports[i].name = port;
        message.ports[i].disposition = MACH_MSG_TYPE_COPY_SEND;
        message.ports[i].type = MACH_MSG_PORT_DESCRIPTOR;
    }
#if TEST_MODE == 3
    /* A complex message with one fewer descriptor than the protocol requires. */
    message.body.msgh_descriptor_count = COUNT - 1;
#endif
    return mach_msg(&message.header, MACH_SEND_MSG | MACH_SEND_TIMEOUT,
                    sizeof(message), 0, MACH_PORT_NULL, 1000, MACH_PORT_NULL) == MACH_MSG_SUCCESS ? 0 : 5;
}

static int record_pid(const char *path, const char *suffix) {
    char name[4096];
    if (snprintf(name, sizeof(name), "%s.%s.pid", path, suffix) >= (int)sizeof(name))
        return 1;
    FILE *file = fopen(name, "w");
    if (!file) return 1;
    int failed = fprintf(file, "%ld\n", (long)getpid()) < 0;
    return fclose(file) != 0 || failed;
}

int main(int argc, char **argv) {
    if (argc < 1 || record_pid(argv[0], "helper")) return 9;
#if TEST_MODE == 1
    /* The message is valid but its kernel audit token identifies this child,
       not the helper PID the daemon spawned. The helper attempts to reap it. */
    pid_t child = fork();
    if (child < 0) return 6;
    if (child == 0) {
        if (record_pid(argv[0], "sender")) _exit(10);
        _exit(send_message());
    }
    int status;
    if (waitpid(child, &status, 0) != child || !WIFEXITED(status) || WEXITSTATUS(status)) return 7;
#else
    if (send_message()) return 8;
#endif
    for (;;) pause();
}
