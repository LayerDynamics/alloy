//! Apple-Silicon unified-memory discovery and shared budget planning.
//!
//! This module derives ONE engine-managed budget for memory shared by CPU
//! and Metal. It does not add a GPU memory allowance to installed RAM.
//!
//! Native discovery:
//!
//! - Reads hw.memsize.
//! - Creates the system-default Metal device.
//! - Requires hasUnifiedMemory.
//! - Reads recommendedMaxWorkingSetSize.
//!
//! Metal's recommendation is a performance guideline, not a hard memory
//! ceiling or a measurement of currently available memory.
//!
//! Policy supplies explicit OS/application, control-plane, and allocator
//! headroom. Budget overrides cannot exceed the resulting shared ceiling.
//!
//! Allocator configuration is explicit and process-wide. Discovery and
//! budget derivation never change MLX settings.
//!
//! This module does not discover current OS pressure or a system wired
//! limit, guarantee OS-OOM prevention, reserve model resources, or
//! automatically evict runtime state.

use crate::{admission::AdmissionController, error::EngineError};
use mlx_ffi::memory;
use std::fmt;

/// Platform discovery or shared-budget planning failure.
#[derive(Debug)]
pub enum PlatformError {
    /// Native unified-memory discovery is implemented for Apple-Silicon
    /// macOS, not this target.
    UnsupportedPlatform,

    /// No system-default Metal device was available.
    NoMetalDevice,

    /// The selected Metal device does not share its memory with the CPU.
    NotUnifiedMemory,

    /// The native platform probe failed.
    NativeProbe {
        /// Status from the small native bridge.
        status: i32,
        /// errno reported by a failed system call, when applicable.
        os_error: i32,
    },

    /// Observations or policy values are invalid.
    InvalidPolicy(&'static str),

    /// Checked budget arithmetic overflowed.
    ArithmeticOverflow,

    /// A configured byte count cannot fit the native size_t API.
    NativeSizeOverflow,

    /// An MLX memory operation failed.
    Mlx(memory::MemoryError),

    /// Configuring the memory guideline failed after changing the cache
    /// limit. A compensating cache-limit restore was attempted.
    ConfigurationFailed {
        /// Failure setting the requested memory guideline.
        cause: memory::MemoryError,
        /// Failure restoring the old cache limit, if restoration failed.
        rollback_error: Option<memory::MemoryError>,
    },
}

impl fmt::Display for PlatformError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedPlatform => write!(
                formatter,
                "unified-memory discovery requires Apple-Silicon macOS"
            ),
            Self::NoMetalDevice => {
                write!(formatter, "no system-default Metal device is available")
            }
            Self::NotUnifiedMemory => write!(
                formatter,
                "selected Metal device does not report unified CPU/GPU memory"
            ),
            Self::NativeProbe { status, os_error } => write!(
                formatter,
                "unified-memory probe failed: status={status}, errno={os_error}"
            ),
            Self::InvalidPolicy(message) => {
                write!(formatter, "invalid unified-memory policy: {message}")
            }
            Self::ArithmeticOverflow => {
                write!(formatter, "unified-memory budget arithmetic overflow")
            }
            Self::NativeSizeOverflow => {
                write!(formatter, "unified-memory setting exceeds native size_t")
            }
            Self::Mlx(error) => write!(formatter, "{error}"),
            Self::ConfigurationFailed {
                cause,
                rollback_error,
            } => {
                write!(formatter, "MLX allocator configuration failed: {cause}")?;
                if let Some(error) = rollback_error {
                    write!(formatter, "; cache-limit restoration also failed: {error}")?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for PlatformError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Mlx(error) => Some(error),
            Self::ConfigurationFailed { cause, .. } => Some(cause),
            _ => None,
        }
    }
}

impl From<memory::MemoryError> for PlatformError {
    fn from(error: memory::MemoryError) -> Self {
        Self::Mlx(error)
    }
}

/// Result of platform discovery or shared-memory planning.
pub type PlatformResult<T> = Result<T, PlatformError>;

/// Observations for one shared CPU/Metal memory pool.
///
/// Use discover() for actual native observations. from_observations() is
/// also available for injected platform providers and deterministic tests;
/// it does not independently authenticate caller-supplied measurements.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnifiedMemoryInfo {
    physical_memory_bytes: u64,
    recommended_working_set_bytes: u64,
}

impl UnifiedMemoryInfo {
    /// Validate observations from a platform provider.
    ///
    /// A non-unified device is rejected instead of being assigned a
    /// separate GPU memory budget.
    pub fn from_observations(
        physical_memory_bytes: u64,
        recommended_working_set_bytes: u64,
        has_unified_memory: bool,
    ) -> PlatformResult<Self> {
        if !has_unified_memory {
            return Err(PlatformError::NotUnifiedMemory);
        }
        if physical_memory_bytes == 0 {
            return Err(PlatformError::InvalidPolicy(
                "physical memory observation must be positive",
            ));
        }
        if recommended_working_set_bytes == 0 {
            return Err(PlatformError::InvalidPolicy(
                "Metal working-set recommendation must be positive",
            ));
        }

        Ok(Self {
            physical_memory_bytes,
            recommended_working_set_bytes,
        })
    }

    /// Installed physical memory in bytes.
    pub fn physical_memory_bytes(&self) -> u64 {
        self.physical_memory_bytes
    }

    /// Metal's recommended working-set size in bytes.
    ///
    /// This refers to the same shared pool, not additional GPU memory.
    pub fn recommended_working_set_bytes(&self) -> u64 {
        self.recommended_working_set_bytes
    }

    /// Derive a shared engine/allocator plan without changing MLX state.
    pub fn plan(&self, policy: UnifiedMemoryPolicy) -> PlatformResult<UnifiedBudgetPlan> {
        if policy.os_application_headroom_bytes == 0
            || policy.control_plane_allowance_bytes == 0
            || policy.allocator_headroom_bytes == 0
        {
            return Err(PlatformError::InvalidPolicy(
                "OS/application, control-plane, and allocator headroom must be explicit and positive",
            ));
        }

        if policy.cache_limit_bytes > policy.allocator_headroom_bytes {
            return Err(PlatformError::InvalidPolicy(
                "free-cache allowance must fit inside allocator headroom",
            ));
        }

        let after_os = self
            .physical_memory_bytes
            .checked_sub(policy.os_application_headroom_bytes)
            .ok_or(PlatformError::InvalidPolicy(
                "OS/application headroom exceeds physical memory",
            ))?;

        // These are two ceilings for ONE shared memory pool.
        // They must not be added together.
        let shared_ceiling = after_os.min(self.recommended_working_set_bytes);

        let excluded = policy
            .control_plane_allowance_bytes
            .checked_add(policy.allocator_headroom_bytes)
            .ok_or(PlatformError::ArithmeticOverflow)?;

        let maximum_engine_budget = shared_ceiling
            .checked_sub(excluded)
            .filter(|value| *value > 0)
            .ok_or(PlatformError::InvalidPolicy(
                "headroom leaves no positive engine-managed budget",
            ))?;

        let engine_budget = match policy.engine_budget_override_bytes {
            Some(0) => {
                return Err(PlatformError::InvalidPolicy(
                    "engine budget override must be positive",
                ));
            }
            Some(value) if value > maximum_engine_budget => {
                return Err(PlatformError::InvalidPolicy(
                    "engine budget override exceeds the shared-memory policy ceiling",
                ));
            }
            Some(value) => value,
            None => maximum_engine_budget,
        };

        // MLX allocator overhead shares the same physical pool. This
        // guideline is not a second allowance added to installed memory.
        let mlx_memory_guideline = engine_budget
            .checked_add(policy.allocator_headroom_bytes)
            .ok_or(PlatformError::ArithmeticOverflow)?;

        usize::try_from(mlx_memory_guideline).map_err(|_| PlatformError::NativeSizeOverflow)?;
        usize::try_from(policy.cache_limit_bytes).map_err(|_| PlatformError::NativeSizeOverflow)?;

        Ok(UnifiedBudgetPlan {
            platform: *self,
            policy,
            shared_ceiling_bytes: shared_ceiling,
            maximum_engine_budget_bytes: maximum_engine_budget,
            engine_budget_bytes: engine_budget,
            mlx_memory_guideline_bytes: mlx_memory_guideline,
        })
    }
}

/// Explicit policy for the single unified-memory pool.
///
/// There is deliberately no Default selecting a percentage, RAM tier,
/// cache size, or wired-memory policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnifiedMemoryPolicy {
    /// Memory excluded for the OS and other applications.
    ///
    /// This is policy, not a measurement of their current usage.
    pub os_application_headroom_bytes: u64,

    /// Allowance excluded for tokenizer, registry, and other control-plane
    /// storage not represented by engine reservations.
    pub control_plane_allowance_bytes: u64,

    /// Allowance excluded for allocator overhead and untracked native
    /// workspace/cached buffers.
    pub allocator_headroom_bytes: u64,

    /// MLX free-cache guideline within allocator_headroom_bytes.
    pub cache_limit_bytes: u64,

    /// Optional smaller engine-managed budget.
    ///
    /// Values above the derived ceiling are rejected, not silently clamped.
    pub engine_budget_override_bytes: Option<u64>,
}

/// Validated plan for one unified CPU/Metal memory pool.
///
/// Constructing this value does not acquire reservations or configure
/// process-wide MLX settings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnifiedBudgetPlan {
    platform: UnifiedMemoryInfo,
    policy: UnifiedMemoryPolicy,
    shared_ceiling_bytes: u64,
    maximum_engine_budget_bytes: u64,
    engine_budget_bytes: u64,
    mlx_memory_guideline_bytes: u64,
}

/// Previous process-wide settings returned after explicit configuration.
///
/// Restoration must be coordinated by the application. There is no
/// automatic Drop restoration that could overwrite another owner's policy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PreviousAllocatorSettings {
    /// Memory guideline before configuration.
    pub memory_limit_bytes: usize,
    /// Free-cache guideline before configuration.
    pub cache_limit_bytes: usize,
}

impl UnifiedBudgetPlan {
    /// Platform observations used by this plan.
    pub fn platform(&self) -> UnifiedMemoryInfo {
        self.platform
    }

    /// Explicit policy used by this plan.
    pub fn policy(&self) -> UnifiedMemoryPolicy {
        self.policy
    }

    /// Lower of the post-OS physical ceiling and Metal recommendation.
    pub fn shared_ceiling_bytes(&self) -> u64 {
        self.shared_ceiling_bytes
    }

    /// Largest engine budget permitted by this plan's headroom policy.
    pub fn maximum_engine_budget_bytes(&self) -> u64 {
        self.maximum_engine_budget_bytes
    }

    /// Budget to pass to the shared AdmissionController.
    pub fn engine_budget_bytes(&self) -> u64 {
        self.engine_budget_bytes
    }

    /// Requested MLX graph-evaluation memory guideline.
    ///
    /// Not a hard allocator ceiling or process RSS limit.
    pub fn mlx_memory_guideline_bytes(&self) -> u64 {
        self.mlx_memory_guideline_bytes
    }

    /// Requested free-cache guideline within allocator headroom.
    pub fn mlx_cache_limit_bytes(&self) -> u64 {
        self.policy.cache_limit_bytes
    }

    /// Create one shared admission ledger from the derived engine budget.
    ///
    /// This does not reserve model resources or alter MLX settings.
    pub fn admission_controller(&self) -> Result<AdmissionController, EngineError> {
        AdmissionController::new(self.engine_budget_bytes)
    }

    /// Explicitly configure process-wide MLX memory/cache guidelines.
    ///
    /// Call only from centrally coordinated setup, before starting model
    /// execution. Other runtimes/libraries must not concurrently modify
    /// policy. This is not an atomic native multi-setting transaction.
    ///
    /// The old cache limit is restored on memory-guideline failure where
    /// possible, and a restoration failure is reported explicitly.
    ///
    /// This operation does not set a wired limit, modify sysctls, discover
    /// OS pressure, or alter existing admission reservations.
    pub fn configure_mlx_allocator(&self) -> PlatformResult<PreviousAllocatorSettings> {
        let memory_limit = usize::try_from(self.mlx_memory_guideline_bytes)
            .map_err(|_| PlatformError::NativeSizeOverflow)?;
        let cache_limit = usize::try_from(self.policy.cache_limit_bytes)
            .map_err(|_| PlatformError::NativeSizeOverflow)?;

        let observed = memory::snapshot()?;
        if observed.active_bytes > memory_limit {
            return Err(PlatformError::InvalidPolicy(
                "requested guideline is below already-active MLX allocations",
            ));
        }

        let previous_cache = memory::set_cache_limit_bytes(cache_limit)?;

        match memory::set_memory_limit_bytes(memory_limit) {
            Ok(previous_memory) => Ok(PreviousAllocatorSettings {
                memory_limit_bytes: previous_memory,
                cache_limit_bytes: previous_cache,
            }),
            Err(cause) => {
                let rollback_error = memory::set_cache_limit_bytes(previous_cache).err();

                Err(PlatformError::ConfigurationFailed {
                    cause,
                    rollback_error,
                })
            }
        }
    }
}

/// Discover the actual default-device unified-memory platform.
///
/// Supported on Apple-Silicon macOS. Unsupported targets return an error;
/// Linux tests can exercise pure planning through injected observations.
pub fn discover() -> PlatformResult<UnifiedMemoryInfo> {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        let mut physical = 0_u64;
        let mut recommended = 0_u64;
        let mut os_error = 0_i32;

        // SAFETY: all output pointers refer to correctly typed writable
        // storage. The linked bridge retains/releases Objective-C objects
        // under ARC and catches Objective-C exceptions at its boundary.
        let status =
            unsafe { alloy_unified_memory_probe(&mut physical, &mut recommended, &mut os_error) };

        match status {
            0 => UnifiedMemoryInfo::from_observations(physical, recommended, true),
            4 => Err(PlatformError::NoMetalDevice),
            6 => Err(PlatformError::NotUnifiedMemory),
            status => Err(PlatformError::NativeProbe { status, os_error }),
        }
    }

    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        Err(PlatformError::UnsupportedPlatform)
    }
}

#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
unsafe extern "C" {
    fn alloy_unified_memory_probe(
        physical_bytes: *mut u64,
        recommended_bytes: *mut u64,
        os_error: *mut i32,
    ) -> i32;

    fn alloy_host_memory_pressure_probe(
        free_bytes: *mut u64,
        inactive_bytes: *mut u64,
        active_file_bytes: *mut u64,
        other_apps_active_bytes: *mut u64,
        phys_footprint_bytes: *mut u64,
        graphics_footprint_bytes: *mut u64,
        iogpu_wired_limit_bytes: *mut u64,
        os_error: *mut i32,
    ) -> i32;
}

/// Discover a full [`crate::memory_guard::HostMemorySnapshot`] on Apple Silicon.
///
/// Combines the unified-memory probe with live vm_stat / phys_footprint /
/// `iogpu.wired_limit_mb` observations for the oMLX-method memory guard.
pub fn discover_host_memory_snapshot() -> PlatformResult<crate::memory_guard::HostMemorySnapshot> {
    let info = discover()?;

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        let mut free = 0_u64;
        let mut inactive = 0_u64;
        let mut active_file = 0_u64;
        let mut other_apps = 0_u64;
        let mut phys = 0_u64;
        let mut graphics = 0_u64;
        let mut wired = 0_u64;
        let mut os_error = 0_i32;

        // SAFETY: all pointers refer to local writable u64/i32 storage.
        let status = unsafe {
            alloy_host_memory_pressure_probe(
                &mut free,
                &mut inactive,
                &mut active_file,
                &mut other_apps,
                &mut phys,
                &mut graphics,
                &mut wired,
                &mut os_error,
            )
        };
        if status != 0 {
            return Err(PlatformError::NativeProbe { status, os_error });
        }

        Ok(crate::memory_guard::HostMemorySnapshot {
            physical_memory_bytes: info.physical_memory_bytes(),
            free_bytes: free,
            inactive_bytes: inactive,
            active_file_bytes: active_file,
            other_apps_active_bytes: other_apps,
            phys_footprint_bytes: phys,
            graphics_footprint_bytes: graphics,
            iogpu_wired_limit_bytes: wired,
            recommended_working_set_bytes: info.recommended_working_set_bytes(),
            release_credit_bytes: 0,
        })
    }

    #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
    {
        let _ = info;
        Err(PlatformError::UnsupportedPlatform)
    }
}

#[cfg(test)]
#[path = "../tests/unit/platform_tests.rs"]
mod tests;
