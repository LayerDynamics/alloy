/* Standalone development helpers, built from the same native owner sources. */
#include "owned_process.h"
#include <string.h>
extern int alloy_fileport_helper(void);
int main(int argc, char **argv) {
#ifdef ALLOY_PIPE_HELPER
    if (argc != 2 || strcmp(argv[1], "--alloy-pipe-helper-v1")) return 2;
    return alloy_fileport_helper();
#else
    return alloy_exec_helper(argc, argv);
#endif
}
