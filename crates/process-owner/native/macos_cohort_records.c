/* Crash-safe registration records. The existing per-user registry bounds the
 * number of host directories to 64; each directory has at most 64 cohort
 * records, including uncertain submissions. Reservations use the same
 * cross-host registry flock as Rust's host registration. Proven-complete
 * deletion is idempotent and can only reduce the charged record count. */
#include "macos_containment.h"
#include "owned_process.h"
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <time.h>
#include <unistd.h>

#define COHORT_RECORD_MAGIC UINT64_C(0x414c4c4f59435231)
static int boot_id(char boot[37]) {
    size_t size = 37;
    if (sysctlbyname("kern.bootsessionuuid", boot, &size, NULL, 0)) return errno;
    return size == 37 && boot[36] == 0 ? 0 : EIO;
}
static int valid_path(const char *path) {
    char prefix[96];
    snprintf(prefix, sizeof(prefix), "/private/tmp/alloy-brokers-%u/com.layerdynamics.alloy.broker.", getuid());
    if (strncmp(path, prefix, strlen(prefix)) || strlen(path) >= 1024) return 0;
    const char *file = strrchr(path, '/');
    if (!file || strlen(file + 1) != 45 || strncmp(file + 1, "cohort.", 7)) return 0;
    for (const char *p = path + strlen(prefix); p < file; ++p)
        if (!((*p >= '0' && *p <= '9') || (*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z') || *p == '.')) return 0;
    file += 8;
    for (int i = 0; i < 32; ++i) if (!((file[i] >= '0' && file[i] <= '9') || (file[i] >= 'a' && file[i] <= 'f'))) return 0;
    return !strcmp(file + 32, ".state");
}
static int read_record(const char *path, alloy_cohort_record *record, int *descriptor) {
    if (!valid_path(path)) return EINVAL;
    int fd = open(path, O_RDWR | O_NOFOLLOW | O_CLOEXEC);
    if (fd < 0) return errno;
    struct stat info;
    int error = fstat(fd, &info) ? errno : 0;
    if (!error && (!S_ISREG(info.st_mode) || info.st_uid != getuid() || (info.st_mode & 077) || info.st_size != sizeof(*record))) error = EPERM;
    if (!error && pread(fd, record, sizeof(*record), 0) != sizeof(*record)) error = EIO;
    if (!error && (record->magic != COHORT_RECORD_MAGIC || !memchr(record->label, 0, sizeof(record->label)) ||
        strncmp(record->label, "com.layerdynamics.alloy.cohort.", sizeof("com.layerdynamics.alloy.cohort.") - 1) ||
        record->host <= 0 || !record->host_identity)) error = EINVAL;
    if (error) close(fd);
    else *descriptor = fd;
    return error;
}
static int directory_sync(const char *path) {
    char parent[1024]; snprintf(parent, sizeof(parent), "%s", path);
    char *slash = strrchr(parent, '/'); if (!slash) return EINVAL; *slash = 0;
    int fd = open(parent, O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    if (fd < 0) return errno;
    int result = fsync(fd) ? errno : 0; close(fd); return result;
}
static int registry_lock(const char *directory) {
    char path[1024]; snprintf(path, sizeof(path), "%s", directory);
    char *slash = strrchr(path, '/'); if (!slash) { errno = EINVAL; return -1; }
    snprintf(slash + 1, (size_t)(path + sizeof(path) - slash - 1), "registry.lock");
    int fd = open(path, O_RDWR | O_NOFOLLOW | O_CLOEXEC);
    if (fd < 0) return -1;
    struct stat info;
    if (fstat(fd, &info) || !S_ISREG(info.st_mode) || info.st_uid != getuid() || (info.st_mode & 077)) { close(fd); errno = EPERM; return -1; }
    struct timespec delay = {0, 1000000};
    for (int i = 0; i < 5000; ++i) {
        if (!flock(fd, LOCK_EX | LOCK_NB)) return fd;
        if (errno != EWOULDBLOCK && errno != EINTR) { int e = errno; close(fd); errno = e; return -1; }
        nanosleep(&delay, NULL);
    }
    close(fd); errno = ETIMEDOUT; return -1;
}
int alloy_cohort_record_create(const char *directory, launch_state *state, int *lease) {
    *lease = -1;
    int lock = registry_lock(directory); if (lock < 0) return errno;
    DIR *dir = opendir(directory); int error = dir ? 0 : errno, count = 0;
    if (dir) {
        struct dirent *entry;
        while ((entry = readdir(dir))) if (!strncmp(entry->d_name, "cohort.", 7) && strstr(entry->d_name, ".state")) ++count;
        closedir(dir);
    }
    if (!error && count >= 64) error = EBUSY;
    alloy_cohort_record record = {0}; record.magic = COHORT_RECORD_MAGIC;
    if (!error) error = boot_id(record.boot);
    record.host = state->parent_pid; record.host_identity = state->parent_identity;
    snprintf(record.label, sizeof(record.label), "%s", state->cohort_job);
    const char *nonce = strrchr(state->cohort_job, '.');
    if (!error && (!nonce || strlen(nonce + 1) != 32)) error = EINVAL;
    if (!error && snprintf(state->cohort_record_path, sizeof(state->cohort_record_path), "%s/cohort.%s.state", directory, nonce + 1) >= (int)sizeof(state->cohort_record_path)) error = ENAMETOOLONG;
    int fd = -1;
    if (!error && (fd = open(state->cohort_record_path, O_CREAT | O_EXCL | O_RDWR | O_NOFOLLOW | O_CLOEXEC, 0600)) < 0) error = errno;
    if (!error && write(fd, &record, sizeof(record)) != sizeof(record)) error = EIO;
    if (!error && fsync(fd)) error = errno;
    if (fd >= 0) close(fd);
    if (!error) error = directory_sync(state->cohort_record_path);
    if (error) close(lock);
    else *lease = lock; /* submitter retains the lock through SMJobSubmit */
    return error;
}
int alloy_cohort_record_attach(launch_state *state, alloy_cohort_record **mapping) {
    char directory[1024]; snprintf(directory,sizeof(directory),"%s",state->cohort_record_path);
    char *slash=strrchr(directory,'/');if(!slash)return EINVAL;*slash=0;
    int lock=registry_lock(directory);if(lock<0)return errno;
    alloy_cohort_record record; int fd = -1;
    int error = read_record(state->cohort_record_path, &record, &fd);
    if (error) {close(lock);return error;}
    char boot[37]; error = boot_id(boot);
    if (!error && (memcmp(boot, record.boot, sizeof(boot)) || strcmp(record.label, state->cohort_job) ||
        record.host != state->parent_pid || record.host_identity != state->parent_identity || record.coalition)) error = EPERM;
    close(fd); if (error) {close(lock);return error;}
    record.cleaner = getpid(); record.cleaner_identity = alloy_process_identity(getpid());
    error = alloy_containment_current_coalition(&record.coalition);
    if (error || !record.cleaner_identity) {close(lock);return error ? error : EIO;}
    char staged[1056]; snprintf(staged, sizeof(staged), "%s.stage", state->cohort_record_path);
    fd = open(staged, O_CREAT | O_EXCL | O_RDWR | O_NOFOLLOW | O_CLOEXEC, 0600);
    if (fd < 0) {error=errno;close(lock);return error;}
    if (write(fd, &record, sizeof(record)) != sizeof(record)) error = EIO;
    if (!error && fsync(fd)) error = errno;
    if (!error && rename(staged, state->cohort_record_path)) error = errno;
    if (!error) error = directory_sync(state->cohort_record_path);
    if (!error) {
        *mapping = mmap(NULL, sizeof(record), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (*mapping == MAP_FAILED) error = errno;
    }
    close(fd); close(lock); return error;
}
int alloy_cohort_record_complete(const char *path, uint64_t id) {
    alloy_cohort_record record; int fd = -1; int error = read_record(path, &record, &fd);
    if (error) return error;
    char boot[37]; error = boot_id(boot);
    if (!error && (memcmp(boot, record.boot, sizeof(boot)) || record.coalition != id)) error = EPERM;
    if (!error) {
        alloy_cohort_record *mapped = mmap(NULL, sizeof(record), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (mapped == MAP_FAILED) error = errno;
        else { atomic_store_explicit(&mapped->complete, 1, memory_order_release); if (msync(mapped, sizeof(*mapped), MS_SYNC)) error = errno; munmap(mapped, sizeof(*mapped)); }
    }
    close(fd); return error;
}
int alloy_cohort_record_submitting(const char *path) {
    alloy_cohort_record record;int fd=-1;
    int error=read_record(path,&record,&fd);if(error)return error;
    alloy_cohort_record *mapped=mmap(NULL,sizeof(record),PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
    if(mapped==MAP_FAILED)error=errno;
    else {atomic_store_explicit(&mapped->submission,1,memory_order_release);
        if(msync(mapped,sizeof(*mapped),MS_SYNC))error=errno;
        munmap(mapped,sizeof(*mapped));}
    close(fd);return error;
}
int alloy_cohort_record_retire(const char *path) {
    alloy_cohort_record record; int fd = -1; int error = read_record(path, &record, &fd);
    if (error == ENOENT) return 0;
    if (error) return error;
    char boot[37]; error = boot_id(boot);
    if (error) { close(fd); return error; }
    int same_boot = !memcmp(boot, record.boot, sizeof(boot));
    int host_alive = same_boot ? alloy_owned_alive(record.host, record.host_identity) : 0;
    if (same_boot && !record.complete && host_alive != 0) { close(fd); return EBUSY; }
    if(same_boot && !record.complete && !record.coalition && record.submission){
        int registered=alloy_broker_registered(record.label);
        if(registered<=0){close(fd);return registered<0?ENOTSUP:EAGAIN;}
        /* A submitter may have died inside the RPC. A momentarily absent job
         * cannot prove that no delayed registration will commit. Retain the
         * charged record until a live service proves no workload or a known
         * registration can be explicitly removed under this registry lock. */
    }
    if (same_boot && record.coalition && !record.complete) {
        int cleaner_alive = record.cleaner_identity ? alloy_owned_alive(record.cleaner, record.cleaner_identity) : -1;
        if (cleaner_alive != 0) { close(fd); return EBUSY; }
        /* Revocation is published by the independent trusted holder after
         * cleaner EOF. A crashed holder leaves this record charged. */
        int empty=alloy_containment_empty(record.coalition);
        if(empty<0){close(fd);return -empty;}
        alloy_cohort_record *mapped = mmap(NULL, sizeof(record), PROT_READ, MAP_SHARED, fd, 0);
        if (mapped == MAP_FAILED) { error = errno; close(fd); return error; }
        struct timespec delay = {0, 1000000};
        int revoked = 0;
        for (int i = 0; !empty && i < 2000; ++i) {
            revoked = atomic_load_explicit(&mapped->revoked, memory_order_acquire);
            if (revoked) break;
            empty=alloy_containment_empty(record.coalition);
            if(empty<0){error=-empty;break;}
            if(!empty)nanosleep(&delay, NULL);
        }
        munmap(mapped, sizeof(*mapped));
        if (error || (!empty && revoked != 1)) { close(fd); return error?error:EIO; }
        error = empty ? 0 : alloy_containment_drain(record.coalition);
        if (error) { close(fd); return error; }
    }
    close(fd);
    /* Old-boot jobs cannot exist; do not address a potentially reused label
     * or CID in the new kernel session. Their random record name is local. */
    if (same_boot && alloy_broker_registered(record.label) != 0) {
        char diagnostic[2048]; error = alloy_launchd_job(record.label, NULL, 0, diagnostic, sizeof(diagnostic));
        if (error && alloy_broker_registered(record.label) != 0) return error;
    }
    if (same_boot && alloy_broker_registered(record.label) != 0) return EBUSY;
    if (unlink(path) && errno != ENOENT) return errno;
    char staged[1056]; snprintf(staged, sizeof(staged), "%s.stage", path);
    if (unlink(staged) && errno != ENOENT) return errno;
    return directory_sync(path);
}
int alloy_cohort_records_retire(const char *directory) {
    /* Caller holds the per-user registry lock and verified the old host dead. */
    DIR *dir = opendir(directory); if (!dir) return errno;
    int error = 0; struct dirent *entry;
    while ((entry = readdir(dir))) {
        if (strncmp(entry->d_name, "cohort.", 7)) continue;
        size_t length = strlen(entry->d_name);
        if (length < 6 || strcmp(entry->d_name + length - 6, ".state")) continue;
        char path[1024];
        if (snprintf(path, sizeof(path), "%s/%s", directory, entry->d_name) >= (int)sizeof(path)) { error = ENAMETOOLONG; break; }
        error = alloy_cohort_record_retire(path);
        if (error) break;
    }
    closedir(dir); return error;
}
