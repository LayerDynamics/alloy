#include <mach/mach.h>
#include <mach/task_special_ports.h>
#include <string.h>
#include <unistd.h>

int main(void) {
    mach_port_t destination = MACH_PORT_NULL;
    if (task_get_special_port(mach_task_self(), TASK_BOOTSTRAP_PORT, &destination) != KERN_SUCCESS)
        return 1;
    mach_msg_header_t message;
    memset(&message, 0, sizeof(message));
    message.msgh_bits = MACH_MSGH_BITS(MACH_MSG_TYPE_COPY_SEND, 0);
    message.msgh_size = sizeof(message);
    message.msgh_remote_port = destination;
    message.msgh_id = 123;
    if (mach_msg(&message, MACH_SEND_MSG | MACH_SEND_TIMEOUT, sizeof(message), 0,
                 MACH_PORT_NULL, 1000, MACH_PORT_NULL) != MACH_MSG_SUCCESS)
        return 2;
    /* The parent must reject the envelope and terminate this exact helper. */
    for (;;) pause();
}
