#include "owned_process.h"
/* Private, bounded transfer of anonymous pipe ends from a single-threaded
 * child. No descriptors are created in the multithreaded daemon until
 * fileport_makefd atomically returns them with FD_CLOEXEC set. */
#include <bsm/libbsm.h>
#include <errno.h>
#include <fcntl.h>
#include <mach/mach.h>
#include <mach/task_special_ports.h>
#include <signal.h>
#include <spawn.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/fileport.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

enum { ALLOY_PIPE_COUNT = 6, ALLOY_MESSAGE_ID = 0x41504c59 };
typedef struct {
    mach_msg_header_t header;
    mach_msg_body_t body;
    mach_msg_port_descriptor_t ports[ALLOY_PIPE_COUNT];
} alloy_message;

/* Called only by the dedicated helper process. */
int alloy_fileport_helper(void) {
    mach_port_t destination = MACH_PORT_NULL;
    if (task_get_special_port(mach_task_self(), TASK_BOOTSTRAP_PORT, &destination) != KERN_SUCCESS)
        return 10;
    alloy_message message;
    memset(&message, 0, sizeof(message));
    message.header.msgh_bits = MACH_MSGH_BITS(MACH_MSG_TYPE_COPY_SEND, 0) | MACH_MSGH_BITS_COMPLEX;
    message.header.msgh_size = sizeof(message);
    message.header.msgh_remote_port = destination;
    message.header.msgh_id = ALLOY_MESSAGE_ID;
    message.body.msgh_descriptor_count = ALLOY_PIPE_COUNT;
    int endpoints[ALLOY_PIPE_COUNT];
    for (int i = 0; i < ALLOY_PIPE_COUNT; i += 2) {
        if (pipe(&endpoints[i]) != 0)
            return 11;
    }
    for (int i = 0; i < ALLOY_PIPE_COUNT; ++i) {
        fileport_t port = MACH_PORT_NULL;
        if (fileport_makeport(endpoints[i], &port) != 0)
            return 12;
        message.ports[i].name = port;
        message.ports[i].disposition = MACH_MSG_TYPE_COPY_SEND;
        message.ports[i].type = MACH_MSG_PORT_DESCRIPTOR;
    }
    kern_return_t result = mach_msg(&message.header, MACH_SEND_MSG | MACH_SEND_TIMEOUT,
                                    message.header.msgh_size, 0, MACH_PORT_NULL,
                                    2000, MACH_PORT_NULL);
    /* Process termination releases all local descriptors and rights. */
    return result == MACH_MSG_SUCCESS ? 0 : 13;
}

/* Cancellation callback is invoked only synchronously on the caller thread. */


/* Return 0 on success; on failure close every imported descriptor, destroy the
 * receive right and kill/reap the exact helper. The plugin is not spawned here. */
int alloy_fileport_pipes(const char *launcher, const char *helper, alloy_stop_callback stop, void *context, int reap_owner, int out[6]) {
    mach_port_t receiver = MACH_PORT_NULL;
    posix_spawnattr_t attr = NULL;
    posix_spawn_file_actions_t actions = NULL;
    int attr_ready = 0, actions_ready = 0, received = 0, result = EIO;
    pid_t pid = 0;
    uint64_t identity = 0;
    void *control = NULL;
    int endpoints[ALLOY_PIPE_COUNT] = {-1,-1,-1,-1,-1,-1};
    union { alloy_message message; unsigned char bytes[4096]; } buffer;
    memset(&buffer, 0, sizeof(buffer));
    if (stop(context)) return ECANCELED;
    if (mach_port_allocate(mach_task_self(), MACH_PORT_RIGHT_RECEIVE, &receiver) != KERN_SUCCESS)
        goto done;
    if (mach_port_insert_right(mach_task_self(), receiver, receiver, MACH_MSG_TYPE_MAKE_SEND) != KERN_SUCCESS)
        goto done;
    int error = posix_spawnattr_init(&attr);
    if (error) { result = error; goto done; }
    attr_ready = 1;
    if ((error = posix_spawnattr_setflags(&attr, POSIX_SPAWN_CLOEXEC_DEFAULT)) != 0 ||
        (error = posix_spawnattr_setspecialport_np(&attr, receiver, TASK_BOOTSTRAP_PORT)) != 0) {
        result = error; goto done;
    }
    error = posix_spawn_file_actions_init(&actions);
    if (error) { result = error; goto done; }
    actions_ready = 1;
    for (int fd = 0; fd < 3; ++fd) {
        error = posix_spawn_file_actions_addopen(&actions, fd, "/dev/null",
                                                  fd == 0 ? O_RDONLY : O_WRONLY, 0);
        if (error) { result = error; goto done; }
    }
    char *argv[] = {(char *)helper, "--alloy-pipe-helper-v1", NULL};
    char *envp[] = {NULL};
    error = alloy_owned_spawn(launcher, helper, &actions, &attr, argv, envp, stop, context, reap_owner, &pid, &identity, &control);
    if (error) { pid = 0; result = error; goto done; }
    mach_msg_option_t options = MACH_RCV_MSG | MACH_RCV_TIMEOUT |
        MACH_RCV_TRAILER_TYPE(MACH_MSG_TRAILER_FORMAT_0) |
        MACH_RCV_TRAILER_ELEMENTS(MACH_RCV_TRAILER_AUDIT);
    kern_return_t code;
    do {
        if (stop(context)) { result = ECANCELED; goto done; }
        code = mach_msg(&buffer.message.header, options, 0, sizeof(buffer),
                        receiver, 25, MACH_PORT_NULL);
    } while (code == MACH_RCV_TIMED_OUT || code == MACH_RCV_INTERRUPTED);
    if (code != MACH_MSG_SUCCESS) goto done;
    received = 1;
    mach_msg_header_t *header = &buffer.message.header;
    size_t offset = (header->msgh_size + 3u) & ~(size_t)3u;
    if (header->msgh_size != sizeof(alloy_message) ||
        !(header->msgh_bits & MACH_MSGH_BITS_COMPLEX) ||
        header->msgh_id != ALLOY_MESSAGE_ID ||
        buffer.message.body.msgh_descriptor_count != ALLOY_PIPE_COUNT ||
        offset + sizeof(mach_msg_audit_trailer_t) > sizeof(buffer))
        goto done;
    mach_msg_audit_trailer_t *trailer = (void *)(buffer.bytes + offset);
    if (trailer->msgh_trailer_type != MACH_MSG_TRAILER_FORMAT_0 ||
        trailer->msgh_trailer_size < sizeof(*trailer) ||
        audit_token_to_pid(trailer->msgh_audit) != pid)
        goto done;
    for (int i = 0; i < ALLOY_PIPE_COUNT; ++i) {
        mach_msg_port_descriptor_t *port = &buffer.message.ports[i];
        if (port->type != MACH_MSG_PORT_DESCRIPTOR ||
            port->disposition != MACH_MSG_TYPE_PORT_SEND ||
            port->name == MACH_PORT_NULL)
            goto done;
        int fd = fileport_makefd(port->name);
        if (fd < 0) { result = errno; goto done; }
        if (fd < 3) {
            int copy = fcntl(fd, F_DUPFD_CLOEXEC, 3);
            int saved = errno;
            close(fd);
            if (copy < 0) { result = saved; goto done; }
            fd = copy;
        }
        endpoints[i] = fd;
        struct stat info;
        int flags = fcntl(fd, F_GETFD);
        int access = fcntl(fd, F_GETFL);
        if (flags < 0 || !(flags & FD_CLOEXEC) || access < 0 ||
            fstat(fd, &info) != 0 || !S_ISFIFO(info.st_mode) ||
            (access & O_ACCMODE) != (i % 2 ? O_WRONLY : O_RDONLY))
            goto done;
    }
    for (;;) {
        int status = 0;
        int waited = alloy_owned_wait(control, &status, 1);
        if (waited == 0) {
            if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) goto done;
            break;
        }
        if (waited != EAGAIN) { result = waited; goto done; }
        if (stop(context)) { result = ECANCELED; goto done; }
        struct timespec pause = {0, 10000000};
        (void)nanosleep(&pause, NULL);
    }
    if (stop(context)) { result = ECANCELED; goto done; }
    for (int i = 0; i < ALLOY_PIPE_COUNT; ++i) {
        out[i] = endpoints[i]; endpoints[i] = -1;
    }
    result = 0;
done:
    for (int i = 0; i < ALLOY_PIPE_COUNT; ++i) if (endpoints[i] >= 0) close(endpoints[i]);
    if (received) mach_msg_destroy(&buffer.message.header);
    if (control) alloy_owned_close(control);
    if (actions_ready) posix_spawn_file_actions_destroy(&actions);
    if (attr_ready) posix_spawnattr_destroy(&attr);
    if (receiver != MACH_PORT_NULL) {
        (void)mach_port_deallocate(mach_task_self(), receiver);
        (void)mach_port_mod_refs(mach_task_self(), receiver, MACH_PORT_RIGHT_RECEIVE, -1);
    }
    return result;
}
