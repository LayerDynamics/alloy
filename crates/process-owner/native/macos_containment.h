#ifndef ALLOY_MACOS_CONTAINMENT_H
#define ALLOY_MACOS_CONTAINMENT_H
#include <stdatomic.h>
#include <stdint.h>

/* Private shared-memory protocol. Workload execs never inherit its descriptor. */
enum { INITIAL, READY, CREATE, TARGET_READY, RUN, STOP, DONE };
typedef struct {
    _Atomic unsigned state;
    _Atomic int pid, target_pid, parent_pid, status, error;
    _Atomic uint64_t identity, target_identity, parent_identity;
    char cohort_job[256];
    _Atomic int gate_pid, revoker_pid;
    _Atomic uint64_t gate_identity, revoker_identity;
    _Atomic uint64_t cohort_identity;
    _Atomic int fork_revoked;
    char cohort_record_path[1024];
    _Atomic int test_revoke_pause, test_revoke_paused;
    _Atomic int recovery_started, recovery_done, recovery_error, recovery_requested;
    _Atomic int test_submit_pause, test_submit_paused, test_setup_pause, test_setup_paused;
    _Atomic int test_recovery_fail;
    _Atomic int test_cleaner_signal_fail;
} launch_state;

typedef struct {
    uint64_t magic;
    char boot[37], label[256];
    int32_t host, cleaner;
    uint64_t host_identity, cleaner_identity, coalition;
    _Atomic int revoked, complete, submission;
} alloy_cohort_record;

/* A cohort service must start alone in a fresh launchd resource coalition.
 * Returns an error without publishing successful completion if cleanup is
 * inconclusive. arguments use the --alloy-owned-exec-v1 helper protocol. */
int alloy_containment_run(launch_state *state, int argc, char **arguments,
                          const int io[3], alloy_cohort_record *record);
int alloy_containment_recover(launch_state *state);
int alloy_containment_current_coalition(uint64_t *id);
int alloy_containment_drain(uint64_t id);
int alloy_containment_empty(uint64_t id);
int alloy_cohort_record_create(const char *directory, launch_state *state, int *lease);
int alloy_cohort_record_attach(launch_state *state, alloy_cohort_record **record);
int alloy_cohort_record_submitting(const char *path);
int alloy_cohort_record_complete(const char *path, uint64_t id);
int alloy_cohort_record_retire(const char *path);
int alloy_cohort_records_retire(const char *directory);
extern int alloy_containment_gate;
#endif
