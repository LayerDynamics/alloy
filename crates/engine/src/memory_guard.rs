//! oMLX-method live memory guard (tiers, ceiling, watermarks).
//!
//! This module computes the hard ceiling and soft/hard watermarks the same
//! way oMLX's `ProcessMemoryEnforcer` does. Numbers are pinned in
//! `docs/designs/omlx-memory-method-parity.md`.
//!
//! It does not allocate model tensors, own reservations, or call Metal.
//! Callers supply host/MLX observations and apply reclaim actions.

use std::fmt;

/// Memory left free for the OS and other apps.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum MemoryGuardTier {
    /// Keep ~20% of RAM (6–16 GiB) free.
    Safe,
    /// Keep ~8% of RAM (3–8 GiB) free. Default.
    Balanced,
    /// Keep ~2% of RAM (1.5–4 GiB) free.
    Aggressive,
    /// User-pinned ceiling in GiB (2 GiB static reserve).
    Custom,
}

impl MemoryGuardTier {
    /// Parse a CLI / settings tier name. Unknown values are rejected.
    pub fn parse(name: &str) -> Result<Self, MemoryGuardError> {
        match name.trim().to_ascii_lowercase().as_str() {
            "safe" => Ok(Self::Safe),
            "balanced" => Ok(Self::Balanced),
            "aggressive" => Ok(Self::Aggressive),
            "custom" => Ok(Self::Custom),
            _ => Err(MemoryGuardError::InvalidTier),
        }
    }

    /// Stable CLI name.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Safe => "safe",
            Self::Balanced => "balanced",
            Self::Aggressive => "aggressive",
            Self::Custom => "custom",
        }
    }
}

impl fmt::Display for MemoryGuardTier {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Which component produced `hard_limit` after `min(...)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CeilingBinding {
    /// `total_ram − tier_reserve`.
    Static,
    /// Live reclaimable pool (or custom pinned GB).
    Dynamic,
    /// Metal / wired cap plus CPU footprint.
    MetalCap,
}

impl CeilingBinding {
    /// Stable name for errors and status.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Static => "static",
            Self::Dynamic => "dynamic",
            Self::MetalCap => "metal_cap",
        }
    }
}

impl fmt::Display for CeilingBinding {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Soft / hard / ok pressure relative to watermarks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PressureLevel {
    /// Below the soft watermark.
    Ok,
    /// Soft ≤ usage < hard.
    Soft,
    /// Usage ≥ hard watermark.
    Hard,
}

impl PressureLevel {
    /// Stable name for logs and status.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::Soft => "soft",
            Self::Hard => "hard",
        }
    }
}

/// Explicit guard configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MemoryGuardConfig {
    /// Tier selecting reserves and watermark fractions.
    pub tier: MemoryGuardTier,
    /// Custom ceiling in bytes. Used only when `tier == Custom`.
    pub custom_ceiling_bytes: u64,
    /// When false, hard ceiling is reported as 0 (“no limit”).
    pub enabled: bool,
}

impl MemoryGuardConfig {
    /// Default: balanced, enabled, no custom ceiling.
    pub fn balanced() -> Self {
        Self {
            tier: MemoryGuardTier::Balanced,
            custom_ceiling_bytes: 0,
            enabled: true,
        }
    }

    /// Custom pinned ceiling.
    pub fn custom(ceiling_bytes: u64) -> Result<Self, MemoryGuardError> {
        if ceiling_bytes == 0 {
            return Err(MemoryGuardError::InvalidCustomCeiling);
        }
        Ok(Self {
            tier: MemoryGuardTier::Custom,
            custom_ceiling_bytes: ceiling_bytes,
            enabled: true,
        })
    }
}

/// Host and process observations required for one ceiling computation.
///
/// Fields may be zero when a probe is unavailable; the ceiling math skips
/// zero metal caps and falls back carefully for missing vm stats.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostMemorySnapshot {
    /// `hw.memsize`.
    pub physical_memory_bytes: u64,
    /// Free pages from `vm_statistics64` (bytes).
    pub free_bytes: u64,
    /// Inactive pages (bytes).
    pub inactive_bytes: u64,
    /// File-backed active pages reclaimable without compression (bytes).
    pub active_file_bytes: u64,
    /// Other apps' anonymous active pages (bytes).
    pub other_apps_active_bytes: u64,
    /// Process physical footprint (bytes).
    pub phys_footprint_bytes: u64,
    /// Process graphics / Metal ledger (bytes). Zero if unavailable.
    pub graphics_footprint_bytes: u64,
    /// `iogpu.wired_limit_mb` in bytes. Zero means unset.
    pub iogpu_wired_limit_bytes: u64,
    /// MTL `recommendedMaxWorkingSetSize`.
    pub recommended_working_set_bytes: u64,
    /// Bytes of just-released memory not yet visible in vm_stat.
    pub release_credit_bytes: u64,
}

impl HostMemorySnapshot {
    /// CPU-side footprint outside Metal buffers.
    pub fn cpu_footprint_bytes(self) -> u64 {
        if self.graphics_footprint_bytes == 0 {
            return 0;
        }
        self.phys_footprint_bytes
            .saturating_sub(self.graphics_footprint_bytes)
    }

    /// Effective Metal allocation cap (sysctl if set, else recommended).
    pub fn metal_cap_raw_bytes(self) -> u64 {
        if self.iogpu_wired_limit_bytes > 0 {
            self.iogpu_wired_limit_bytes
        } else {
            self.recommended_working_set_bytes
        }
    }
}

/// Breakdown of the hard ceiling and which component binds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CeilingBreakdown {
    /// Tier reserve subtracted from physical RAM.
    pub reserve_bytes: u64,
    /// `physical − reserve`.
    pub static_bytes: u64,
    /// Live reclaimable / custom ceiling.
    pub dynamic_bytes: u64,
    /// Metal cap + CPU footprint.
    pub metal_cap_bytes: u64,
    /// `min` of positive candidates (0 when guard disabled).
    pub hard_limit_bytes: u64,
    /// Binding component when `hard_limit_bytes > 0`.
    pub binding: Option<CeilingBinding>,
}

/// Watermark and admit-line bytes for one ceiling snapshot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WatermarkBytes {
    /// Soft watermark.
    pub soft_bytes: u64,
    /// Hard watermark.
    pub hard_bytes: u64,
    /// Prefill admission line.
    pub admit_line_bytes: u64,
    /// Stable abort limit (`min(static, metal)`).
    pub abort_limit_bytes: u64,
}

/// Memory-guard failures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MemoryGuardError {
    /// Unknown tier name.
    InvalidTier,
    /// Custom ceiling must be positive.
    InvalidCustomCeiling,
    /// Physical memory observation must be positive.
    InvalidPhysicalMemory,
    /// Checked arithmetic overflowed.
    ArithmeticOverflow,
}

impl fmt::Display for MemoryGuardError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidTier => write!(
                formatter,
                "memory guard tier must be safe|balanced|aggressive|custom"
            ),
            Self::InvalidCustomCeiling => {
                write!(formatter, "custom memory-guard ceiling must be positive")
            }
            Self::InvalidPhysicalMemory => {
                write!(formatter, "physical memory observation must be positive")
            }
            Self::ArithmeticOverflow => write!(formatter, "memory guard arithmetic overflow"),
        }
    }
}

impl std::error::Error for MemoryGuardError {}

/// Result alias for this module.
pub type MemoryGuardResult<T> = Result<T, MemoryGuardError>;

const GIB: u64 = 1024 * 1024 * 1024;

/// Tier reserve: memory left for OS and other apps.
pub fn tier_reserve_bytes(tier: MemoryGuardTier, total_ram: u64) -> MemoryGuardResult<u64> {
    if total_ram == 0 {
        return Err(MemoryGuardError::InvalidPhysicalMemory);
    }
    match tier {
        MemoryGuardTier::Custom => Ok(2 * GIB),
        MemoryGuardTier::Safe => clamp_share(total_ram, 0.20, 6 * GIB, 16 * GIB),
        MemoryGuardTier::Balanced => clamp_share(total_ram, 0.08, 3 * GIB, 8 * GIB),
        MemoryGuardTier::Aggressive => {
            clamp_share(total_ram, 0.02, (1.5 * GIB as f64) as u64, 4 * GIB)
        }
    }
}

fn clamp_share(total: u64, share: f64, low: u64, high: u64) -> MemoryGuardResult<u64> {
    let raw = (total as f64 * share).round() as u64;
    Ok(raw.clamp(low, high).min(total))
}

/// Other-apps active reclaim ratio for the dynamic ceiling.
pub fn other_app_reclaim_ratio(tier: MemoryGuardTier) -> f64 {
    match tier {
        MemoryGuardTier::Safe => 0.0,
        MemoryGuardTier::Balanced => 0.25,
        MemoryGuardTier::Aggressive => 0.5,
        MemoryGuardTier::Custom => 0.0,
    }
}

/// Soft watermark fraction of the hard ceiling.
pub fn soft_threshold(tier: MemoryGuardTier) -> f64 {
    match tier {
        MemoryGuardTier::Safe => 0.85,
        MemoryGuardTier::Balanced => 0.90,
        MemoryGuardTier::Aggressive => 0.95,
        MemoryGuardTier::Custom => 0.90,
    }
}

/// Hard watermark fraction of the hard ceiling.
pub fn hard_threshold(tier: MemoryGuardTier) -> f64 {
    match tier {
        MemoryGuardTier::Safe => 0.95,
        MemoryGuardTier::Balanced => 0.95,
        MemoryGuardTier::Aggressive => 0.98,
        MemoryGuardTier::Custom => 0.97,
    }
}

/// Prefill headroom fraction of the hard ceiling.
pub fn prefill_headroom(tier: MemoryGuardTier) -> f64 {
    match tier {
        MemoryGuardTier::Safe => 0.90,
        MemoryGuardTier::Balanced => 0.92,
        MemoryGuardTier::Aggressive => 0.97,
        MemoryGuardTier::Custom => 0.95,
    }
}

/// Prefill abort prediction margin fraction of the hard ceiling.
pub fn prefill_abort_margin(tier: MemoryGuardTier) -> f64 {
    match tier {
        MemoryGuardTier::Safe => 0.90,
        MemoryGuardTier::Balanced => 0.93,
        MemoryGuardTier::Aggressive => 0.97,
        MemoryGuardTier::Custom => 0.95,
    }
}

/// Compute the hard ceiling and binding component.
pub fn ceiling_breakdown(
    config: MemoryGuardConfig,
    host: HostMemorySnapshot,
    alloy_usage_bytes: u64,
) -> MemoryGuardResult<CeilingBreakdown> {
    if !config.enabled {
        return Ok(CeilingBreakdown {
            reserve_bytes: 0,
            static_bytes: 0,
            dynamic_bytes: 0,
            metal_cap_bytes: 0,
            hard_limit_bytes: 0,
            binding: None,
        });
    }

    let reserve = tier_reserve_bytes(config.tier, host.physical_memory_bytes)?;
    let static_bytes = host.physical_memory_bytes.saturating_sub(reserve);

    let dynamic_bytes = match config.tier {
        MemoryGuardTier::Custom => config.custom_ceiling_bytes,
        _ => {
            let reclaim = (host.other_apps_active_bytes as f64
                * other_app_reclaim_ratio(config.tier))
            .round() as u64;
            alloy_usage_bytes
                .saturating_add(host.free_bytes)
                .saturating_add(host.inactive_bytes)
                .saturating_add(host.active_file_bytes)
                .saturating_add(reclaim)
                .saturating_add(host.release_credit_bytes)
                .saturating_sub(reserve)
        }
    };

    let metal_raw = host.metal_cap_raw_bytes();
    let metal_cap_bytes = if metal_raw > 0 {
        metal_raw.saturating_add(host.cpu_footprint_bytes())
    } else {
        0
    };

    let mut hard = static_bytes;
    let mut binding = CeilingBinding::Static;
    if dynamic_bytes > 0 && dynamic_bytes < hard {
        hard = dynamic_bytes;
        binding = CeilingBinding::Dynamic;
    }
    if metal_cap_bytes > 0 && metal_cap_bytes < hard {
        hard = metal_cap_bytes;
        binding = CeilingBinding::MetalCap;
    }

    Ok(CeilingBreakdown {
        reserve_bytes: reserve,
        static_bytes,
        dynamic_bytes,
        metal_cap_bytes,
        hard_limit_bytes: hard,
        binding: if hard > 0 { Some(binding) } else { None },
    })
}

/// Soft / hard / admit / abort bytes for a ceiling.
pub fn watermark_bytes(tier: MemoryGuardTier, breakdown: CeilingBreakdown) -> WatermarkBytes {
    let hard_limit = breakdown.hard_limit_bytes;
    let soft = scale(hard_limit, soft_threshold(tier));
    let hard = scale(hard_limit, hard_threshold(tier));
    let admit = scale(hard_limit, prefill_headroom(tier));
    let admit_line = hard_limit.min(hard).min(admit);
    let abort = if breakdown.metal_cap_bytes > 0 {
        breakdown.static_bytes.min(breakdown.metal_cap_bytes)
    } else {
        breakdown.static_bytes
    };
    WatermarkBytes {
        soft_bytes: soft,
        hard_bytes: hard,
        admit_line_bytes: admit_line,
        abort_limit_bytes: abort,
    }
}

/// Classify usage against soft/hard watermarks.
pub fn pressure_level(usage_bytes: u64, watermarks: WatermarkBytes) -> PressureLevel {
    if usage_bytes < watermarks.soft_bytes {
        PressureLevel::Ok
    } else if usage_bytes < watermarks.hard_bytes {
        PressureLevel::Soft
    } else {
        PressureLevel::Hard
    }
}

/// Process usage floor matching oMLX's jetsam-oriented signal.
///
/// `max(mlx_active, phys_footprint − graphics_footprint)` when the graphics
/// ledger is present; otherwise `max(mlx_active, phys_footprint)`. External
/// child-process bytes are added by the caller.
pub fn usage_bytes(mlx_active_bytes: u64, host: HostMemorySnapshot) -> u64 {
    let phys_term = if host.graphics_footprint_bytes == 0 {
        host.phys_footprint_bytes
    } else {
        host.phys_footprint_bytes
            .saturating_sub(host.graphics_footprint_bytes)
    };
    mlx_active_bytes.max(phys_term)
}

/// Wired-limit suggestion: floor to whole MiB, leave ≥5% of RAM for the OS.
pub fn wired_limit_suggestion_bytes(desired_bytes: u64, total_ram: u64) -> u64 {
    if total_ram == 0 {
        return desired_bytes;
    }
    let capped = desired_bytes.min(total_ram.saturating_sub(total_ram / 20));
    let mib = 1024 * 1024;
    (capped / mib) * mib
}

/// Whether the process should call `mlx_set_wired_limit` (sysctl explicitly set).
pub fn should_apply_wired_limit(host: HostMemorySnapshot) -> bool {
    host.iogpu_wired_limit_bytes > 0
}

fn scale(value: u64, fraction: f64) -> u64 {
    ((value as f64) * fraction).floor() as u64
}

#[cfg(test)]
#[path = "../tests/unit/memory_guard_tests.rs"]
mod tests;
