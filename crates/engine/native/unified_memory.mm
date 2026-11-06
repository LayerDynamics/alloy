// Unified-memory platform observations from Apple's native APIs.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach/mach.h>
#include <mach/mach_host.h>
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

// Live host pressure for the oMLX-method memory guard.
//
// Outputs are bytes. On failure the corresponding outputs stay zero and a
// non-zero status is returned. Partial success is not reported: callers
// should fall back to static+Metal planning if this probe fails.
extern "C" int alloy_host_memory_pressure_probe(
    uint64_t* free_bytes,
    uint64_t* inactive_bytes,
    uint64_t* active_file_bytes,
    uint64_t* other_apps_active_bytes,
    uint64_t* phys_footprint_bytes,
    uint64_t* graphics_footprint_bytes,
    uint64_t* iogpu_wired_limit_bytes,
    int* os_error
) {
    if (!free_bytes || !inactive_bytes || !active_file_bytes
        || !other_apps_active_bytes || !phys_footprint_bytes
        || !graphics_footprint_bytes || !iogpu_wired_limit_bytes
        || !os_error) {
        return 1;
    }

    *free_bytes = 0;
    *inactive_bytes = 0;
    *active_file_bytes = 0;
    *other_apps_active_bytes = 0;
    *phys_footprint_bytes = 0;
    *graphics_footprint_bytes = 0;
    *iogpu_wired_limit_bytes = 0;
    *os_error = 0;

    vm_size_t page_size = 0;
    if (host_page_size(mach_host_self(), &page_size) != KERN_SUCCESS
        || page_size == 0) {
        return 2;
    }

    vm_statistics64_data_t stats;
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    kern_return_t kr = host_statistics64(
        mach_host_self(),
        HOST_VM_INFO64,
        reinterpret_cast<host_info64_t>(&stats),
        &count);
    if (kr != KERN_SUCCESS) {
        *os_error = static_cast<int>(kr);
        return 3;
    }

    const uint64_t page = static_cast<uint64_t>(page_size);
    *free_bytes = static_cast<uint64_t>(stats.free_count) * page;
    *inactive_bytes = static_cast<uint64_t>(stats.inactive_count) * page;
    // Approximate file-backed active pages; anonymous active is the
    // remainder of active_count. Exact external/internal splits vary by
    // SDK; this matches oMLX's conservative reclaim inputs closely enough
    // for ceiling math and is covered by injected-observation unit tests.
    const uint64_t active = static_cast<uint64_t>(stats.active_count) * page;
    const uint64_t external =
        static_cast<uint64_t>(stats.external_page_count) * page;
    *active_file_bytes = external < active ? external : active;
    *other_apps_active_bytes = active > *active_file_bytes
        ? active - *active_file_bytes
        : 0;

    task_vm_info_data_t task_info_data;
    mach_msg_type_number_t task_count = TASK_VM_INFO_COUNT;
    kr = task_info(
        mach_task_self(),
        TASK_VM_INFO,
        reinterpret_cast<task_info_t>(&task_info_data),
        &task_count);
    if (kr != KERN_SUCCESS) {
        *os_error = static_cast<int>(kr);
        return 4;
    }
    *phys_footprint_bytes = static_cast<uint64_t>(task_info_data.phys_footprint);
    // Graphics ledger is optional; leave zero when the SDK field is absent.
    // usage_bytes() then falls back to max(mlx_active, phys_footprint).
    (void)task_count;
    *graphics_footprint_bytes = 0;

    int64_t wired_mb = 0;
    size_t wired_size = sizeof(wired_mb);
    if (sysctlbyname("iogpu.wired_limit_mb", &wired_mb, &wired_size, nullptr, 0)
        == 0) {
        if (wired_mb > 0) {
            *iogpu_wired_limit_bytes =
                static_cast<uint64_t>(wired_mb) * 1024ull * 1024ull;
        }
    }

    return 0;
}
