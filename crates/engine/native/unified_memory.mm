// Unified-memory platform observations from Apple's native APIs.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <sys/sysctl.h>
#include <cerrno>
#include <cstdint>

extern "C" int alloy_unified_memory_probe(
    uint64_t* physical_bytes,
    uint64_t* recommended_bytes,
    int* os_error
) {
    if (!physical_bytes || !recommended_bytes || !os_error) {
        return 1;
    }

    *physical_bytes = 0;
    *recommended_bytes = 0;
    *os_error = 0;

    @autoreleasepool {
        @try {
            uint64_t physical = 0;
            size_t size = sizeof(physical);

            if (sysctlbyname(
                    "hw.memsize",
                    &physical,
                    &size,
                    nullptr,
                    0) != 0) {
                *os_error = errno;
                return 2;
            }

            if (size != sizeof(physical) || physical == 0) {
                return 3;
            }

            // This source is compiled with ARC. The create-owned device
            // is released automatically on every return path.
            id<MTLDevice> device = MTLCreateSystemDefaultDevice();
            if (device == nil) {
                return 4;
            }

            if (![device respondsToSelector:@selector(hasUnifiedMemory)] ||
                ![device respondsToSelector:
                    @selector(recommendedMaxWorkingSetSize)]) {
                return 5;
            }

            if (![device hasUnifiedMemory]) {
                return 6;
            }

            uint64_t recommended = [device recommendedMaxWorkingSetSize];
            if (recommended == 0) {
                return 7;
            }

            *physical_bytes = physical;
            *recommended_bytes = recommended;
            return 0;
        }
        @catch (NSException* exception) {
            (void)exception;
            return 8;
        }
    }
}
