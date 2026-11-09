//! Fixed-layout, pre-admitted parser storage.
//!
//! Counts the exact requested slab Layout, including alignment padding,
//! not allocator rounding, RSS, or OS OOM prevention.
//!
//! Owner structs, existing admission-controller metadata, and any later
//! Arc control block are separate layouts.
//!
//! No String, Vec, Map, or Value is created by this module.
//!
//! JSON grammar and duplicate-key validation are the parser's job.
//! This module supplies bounded storage for that work. Freezing storage
//! does not validate the protocol or authorize execution.

use crate::{
    admission::{MemoryReservation, ReservationGrant},
    error::EngineError,
};
use std::{
    alloc::{Layout, alloc_zeroed, dealloc},
    fmt,
    ops::Range,
    ptr::NonNull,
    slice, str,
};

/// Explicit fixed capacities.
///
/// No buffer grows after allocation.
#[derive(Clone, Copy, Debug)]
pub struct ParserAllocationLimits {
    /// Partial delimiter/UTF-8 carry bytes.
    pub carry_bytes: usize,

    /// Pending call-body bytes, excluding carry.
    pub call_bytes: usize,

    /// Decoded JSON-string/key scratch bytes.
    pub decoded_bytes: usize,

    /// JSON nesting-stack entries.
    pub depth: usize,

    /// Duplicate-key bookkeeping entries.
    pub keys: usize,

    /// JSON node descriptors.
    pub nodes: usize,

    /// Total immutable payload bytes for all committed proposals.
    pub proposal_bytes: usize,

    /// Maximum committed proposal descriptors.
    pub calls: usize,
}

/// Zero-valid parser-stack record.
///
/// Integer tags are interpreted by the parser.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct JsonFrame {
    /// Parser-defined container tag.
    pub kind: usize,

    /// Parser-defined grammar state.
    pub state: usize,

    /// Stable object identity for duplicate-key comparison.
    pub object_id: usize,

    /// First key belonging to this object.
    pub first_key: usize,
}

/// Key text is decoded into the separate fixed decoded-byte region.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct JsonKey {
    /// Object owning this key.
    pub object_id: usize,

    /// Decoded-byte start.
    pub start: usize,

    /// Decoded-byte length.
    pub len: usize,
}

/// Allocation-free JSON descriptor.
///
/// There are no owning child containers.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct JsonNode {
    /// Parser-defined JSON node tag.
    pub kind: usize,

    /// Parser-defined parent-node index or sentinel.
    pub parent: usize,

    /// Start offset interpreted according to kind.
    pub start: usize,

    /// Byte length interpreted according to kind.
    pub len: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
struct ProposalSlot {
    name_start: usize,
    name_len: usize,
    args_start: usize,
    args_len: usize,
}

/// Storage failures reported without constructing a formatted message.
#[derive(Debug)]
pub enum ParserAllocationError {
    /// Every fixed capacity must be positive.
    InvalidLimits,

    /// Layout/count/addressability arithmetic failed.
    LayoutOverflow,

    /// Internal region/padding invariants did not match the checked layout.
    InvalidLayout,

    /// A reservation must cover this slab exactly.
    ///
    /// It must not represent another shared allocation.
    ReservationMismatch {
        /// Exact requested backing-layout bytes.
        expected: u64,

        /// Bytes held by the supplied exclusive reservation.
        actual: u64,
    },

    /// The global allocator returned null for this admitted layout.
    AllocationFailed {
        /// Size of the refused allocator request.
        requested_bytes: usize,
    },

    /// A fixed region cannot accommodate the requested operation.
    Capacity {
        /// Fixed region that could not accept the operation.
        region: &'static str,

        /// Required occupied bytes or descriptor count.
        required: usize,

        /// Configured capacity in the same units.
        capacity: usize,
    },

    /// A range is outside its currently populated source region.
    InvalidRange,

    /// Proposal source is not UTF-8.
    InvalidUtf8,

    /// Tool name is empty, padded, or contains control characters.
    InvalidName,

    /// Startup-grant partitioning failed.
    Admission(EngineError),
}

impl fmt::Display for ParserAllocationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for ParserAllocationError {}

impl From<EngineError> for ParserAllocationError {
    fn from(error: EngineError) -> Self {
        Self::Admission(error)
    }
}

#[derive(Clone, Copy)]
struct Region {
    offset: usize,
    count: usize,
}

#[derive(Clone, Copy)]
struct Plan {
    layout: Layout,
    carry: Region,
    call: Region,
    decoded: Region,
    stack: Region,
    keys: Region,
    nodes: Region,
    payload: Region,
    proposals: Region,
}

impl Plan {
    fn new(limits: ParserAllocationLimits) -> Result<Self, ParserAllocationError> {
        if [
            limits.carry_bytes,
            limits.call_bytes,
            limits.decoded_bytes,
            limits.depth,
            limits.keys,
            limits.nodes,
            limits.proposal_bytes,
            limits.calls,
        ]
        .contains(&0)
        {
            return Err(ParserAllocationError::InvalidLimits);
        }

        fn add<T>(layout: &mut Layout, count: usize) -> Result<Region, ParserAllocationError> {
            let array =
                Layout::array::<T>(count).map_err(|_| ParserAllocationError::LayoutOverflow)?;

            let (next, offset) = layout
                .extend(array)
                .map_err(|_| ParserAllocationError::LayoutOverflow)?;

            *layout = next;

            Ok(Region { offset, count })
        }

        let mut layout = Layout::new::<()>();

        let carry = add::<u8>(&mut layout, limits.carry_bytes)?;
        let call = add::<u8>(&mut layout, limits.call_bytes)?;
        let decoded = add::<u8>(&mut layout, limits.decoded_bytes)?;
        let stack = add::<JsonFrame>(&mut layout, limits.depth)?;
        let keys = add::<JsonKey>(&mut layout, limits.keys)?;
        let nodes = add::<JsonNode>(&mut layout, limits.nodes)?;
        let payload = add::<u8>(&mut layout, limits.proposal_bytes)?;
        let proposals = add::<ProposalSlot>(&mut layout, limits.calls)?;

        let plan = Self {
            layout: layout.pad_to_align(),
            carry,
            call,
            decoded,
            stack,
            keys,
            nodes,
            payload,
            proposals,
        };
        plan.validate_layout()?;
        Ok(plan)
    }
}

/// Mutable, disjoint parser workspaces.
///
/// Every slice has its full fixed capacity. The parser maintains used
/// lengths and must reject capacity overflow.
pub struct ParserWorkspaces<'a> {
    /// Partial delimiter/UTF-8 carry.
    pub carry: &'a mut [u8],

    /// Pending call body.
    pub call: &'a mut [u8],

    /// Decoded strings/keys.
    pub decoded: &'a mut [u8],

    /// Nesting stack; use checked indexing.
    pub stack: &'a mut [JsonFrame],

    /// Duplicate-key entries; use checked indexing.
    pub keys: &'a mut [JsonKey],

    /// Flat JSON descriptors; use checked indexing.
    pub nodes: &'a mut [JsonNode],
}

/// Exclusive ownership of one fixed parser/proposal slab and its reservation.
///
/// This type is not Clone.
///
/// Scratch and retained payload coexist in the same admitted slab.
/// Drop deallocates the slab before the reservation field is released.
pub struct ParserAllocation {
    base: NonNull<u8>,
    plan: Plan,
    payload_used: usize,
    proposal_count: usize,
    reservation: MemoryReservation,
}

// SAFETY: The slab is exclusively owned. All mutation requires &mut self.
// References returned by workspaces borrow that exclusive lifetime.
// Moving the owner does not move or invalidate its allocation.
// MemoryReservation supports thread transfer.
unsafe impl Send for ParserAllocation {}

impl ParserAllocation {
    /// Exact requested backing allocation.
    ///
    /// Includes inter-region and trailing alignment padding.
    pub fn layout(limits: ParserAllocationLimits) -> Result<Layout, ParserAllocationError> {
        Ok(Plan::new(limits)?.layout)
    }

    /// Number of backing bytes startup must admit before allocating the slab.
    pub fn required_bytes(limits: ParserAllocationLimits) -> Result<u64, ParserAllocationError> {
        u64::try_from(Self::layout(limits)?.size())
            .map_err(|_| ParserAllocationError::LayoutOverflow)
    }

    /// Transfer an exact partition from an already admitted startup grant.
    ///
    /// No second reserve/reacquire operation occurs here.
    pub fn from_grant(
        grant: &mut ReservationGrant,
        limits: ParserAllocationLimits,
    ) -> Result<Self, ParserAllocationError> {
        let bytes = Self::required_bytes(limits)?;

        let reservation = grant
            .take("Qwen parser/proposal slab", bytes)?
            .ok_or(ParserAllocationError::InvalidLimits)?;

        Self::from_reservation(reservation, limits)
    }

    /// Consume exclusive accounting ownership before allocating.
    ///
    /// Sharing an `Arc<MemoryReservation>` is intentionally not accepted.
    pub fn from_reservation(
        reservation: MemoryReservation,
        limits: ParserAllocationLimits,
    ) -> Result<Self, ParserAllocationError> {
        Self::allocate_with(reservation, limits, alloc_zeroed)
    }

    fn allocate_with(
        reservation: MemoryReservation,
        limits: ParserAllocationLimits,
        allocate: unsafe fn(Layout) -> *mut u8,
    ) -> Result<Self, ParserAllocationError> {
        let plan = Plan::new(limits)?;

        let expected =
            u64::try_from(plan.layout.size()).map_err(|_| ParserAllocationError::LayoutOverflow)?;

        if reservation.bytes() != expected {
            return Err(ParserAllocationError::ReservationMismatch {
                expected,
                actual: reservation.bytes(),
            });
        }

        // SAFETY: Plan validates a nonzero, addressable Layout.
        // Production passes alloc_zeroed, which returns zeroed storage
        // with this alignment or null. The test alternative returns
        // null without allocating.
        let raw = unsafe { allocate(plan.layout) };

        let base = NonNull::new(raw).ok_or(ParserAllocationError::AllocationFailed {
            requested_bytes: plan.layout.size(),
        })?;

        Ok(Self {
            base,
            plan,
            payload_used: 0,
            proposal_count: 0,
            reservation,
        })
    }

    /// Borrow all mutable scratch regions without allocating or growing storage.
    ///
    /// The immutable proposal region is deliberately not exposed here.
    pub fn workspaces(&mut self) -> ParserWorkspaces<'_> {
        let plan = self.plan;
        let base = self.base.as_ptr();

        // SAFETY: Each region is disjoint, and Layout::extend supplies
        // proper alignment for its element type. alloc_zeroed initialized
        // every byte. All represented types contain only usize fields or
        // u8, for which every initialized bit pattern is valid.
        //
        // These slices share only this exclusive &mut self lifetime.
        // No region contains owning or Drop types.
        unsafe {
            ParserWorkspaces {
                carry: slice::from_raw_parts_mut(base.add(plan.carry.offset), plan.carry.count),
                call: slice::from_raw_parts_mut(base.add(plan.call.offset), plan.call.count),
                decoded: slice::from_raw_parts_mut(
                    base.add(plan.decoded.offset),
                    plan.decoded.count,
                ),
                stack: slice::from_raw_parts_mut(
                    base.add(plan.stack.offset).cast::<JsonFrame>(),
                    plan.stack.count,
                ),
                keys: slice::from_raw_parts_mut(
                    base.add(plan.keys.offset).cast::<JsonKey>(),
                    plan.keys.count,
                ),
                nodes: slice::from_raw_parts_mut(
                    base.add(plan.nodes.offset).cast::<JsonNode>(),
                    plan.nodes.count,
                ),
            }
        }
    }

    /// Commit a call after grammar, duplicate-key, and envelope validation.
    ///
    /// Name references populated decoded scratch. Arguments reference
    /// populated pending-call bytes. Ranges are checked against used lengths,
    /// not merely against region capacity.
    ///
    /// All fallible checks complete before retained bytes, descriptors,
    /// or counters change.
    ///
    /// Source/destination overlap in lifetime is already represented by
    /// the single slab layout. Their byte ranges do not overlap.
    ///
    /// This method checks UTF-8 and name structure. It does not validate
    /// JSON grammar or establish that the arguments are an object.
    pub fn commit_validated_call(
        &mut self,
        decoded_used: usize,
        name: Range<usize>,
        call_used: usize,
        arguments: Range<usize>,
    ) -> Result<usize, ParserAllocationError> {
        if decoded_used > self.plan.decoded.count
            || call_used > self.plan.call.count
            || name.start > name.end
            || name.end > decoded_used
            || arguments.start > arguments.end
            || arguments.end > call_used
        {
            return Err(ParserAllocationError::InvalidRange);
        }

        if self.proposal_count >= self.plan.proposals.count {
            return Err(ParserAllocationError::Capacity {
                region: "proposal descriptors",
                required: self.proposal_count.saturating_add(1),
                capacity: self.plan.proposals.count,
            });
        }

        let name_len = name.end - name.start;
        let args_len = arguments.end - arguments.start;

        let args_start = self
            .payload_used
            .checked_add(name_len)
            .ok_or(ParserAllocationError::LayoutOverflow)?;

        let end = args_start
            .checked_add(args_len)
            .ok_or(ParserAllocationError::LayoutOverflow)?;

        if end > self.plan.payload.count {
            return Err(ParserAllocationError::Capacity {
                region: "proposal payload",
                required: end,
                capacity: self.plan.payload.count,
            });
        }

        let base = self.base.as_ptr();

        // SAFETY: Checked source ranges are within their populated,
        // disjoint regions. These reads borrow the source storage only
        // for validation before the destination is mutated.
        let (name_bytes, args_bytes) = unsafe {
            (
                slice::from_raw_parts(base.add(self.plan.decoded.offset + name.start), name_len),
                slice::from_raw_parts(base.add(self.plan.call.offset + arguments.start), args_len),
            )
        };

        let decoded_name =
            str::from_utf8(name_bytes).map_err(|_| ParserAllocationError::InvalidUtf8)?;

        str::from_utf8(args_bytes).map_err(|_| ParserAllocationError::InvalidUtf8)?;

        if decoded_name.is_empty()
            || decoded_name.trim() != decoded_name
            || decoded_name.chars().any(char::is_control)
        {
            return Err(ParserAllocationError::InvalidName);
        }

        let index = self.proposal_count;

        let slot = ProposalSlot {
            name_start: self.payload_used,
            name_len,
            args_start,
            args_len,
        };

        // SAFETY: Every pointer refers to this live slab.
        //
        // Source scratch regions do not overlap the payload region.
        // Destination ranges and the descriptor index were checked above.
        // ProposalSlot contains only usize fields and is properly aligned.
        unsafe {
            let payload = base.add(self.plan.payload.offset);

            std::ptr::copy_nonoverlapping(
                name_bytes.as_ptr(),
                payload.add(self.payload_used),
                name_len,
            );

            std::ptr::copy_nonoverlapping(args_bytes.as_ptr(), payload.add(args_start), args_len);

            base.add(self.plan.proposals.offset)
                .cast::<ProposalSlot>()
                .add(index)
                .write(slot);
        }

        self.payload_used = end;
        self.proposal_count += 1;

        Ok(index)
    }

    /// Exact retained slab charge.
    ///
    /// Independent of how much proposal payload is currently populated.
    pub fn allocated_bytes(&self) -> usize {
        self.plan.layout.size()
    }

    /// Bytes owned in the admission ledger for this slab.
    pub fn reserved_bytes(&self) -> u64 {
        self.reservation.bytes()
    }

    /// Occupied immutable proposal-payload bytes.
    pub fn payload_used(&self) -> usize {
        self.payload_used
    }

    /// Number of committed descriptors.
    pub fn proposal_count(&self) -> usize {
        self.proposal_count
    }

    /// End mutation and transfer the same allocation/reservation to delivery.
    ///
    /// Call only after successful parser/generation completion.
    /// No copy or heap allocation occurs here.
    ///
    /// This method is not itself a protocol-completion or authorization gate.
    pub fn freeze(self) -> FrozenParserAllocation {
        FrozenParserAllocation { storage: self }
    }
}

impl Drop for ParserAllocation {
    fn drop(&mut self) {
        // SAFETY: base is the unique live allocation obtained with this
        // exact Layout. No resize/realloc occurs. Borrow checking prevents
        // references outliving this owner.
        //
        // Fields, including reservation, drop after this method returns.
        unsafe {
            dealloc(self.base.as_ptr(), self.plan.layout);
        }
    }
}

/// Immutable ownership of the slab plus its admission reservation.
///
/// Move this owner with delivery, not merely its borrowed proposal slices.
pub struct FrozenParserAllocation {
    storage: ParserAllocation,
}

// SAFETY: The frozen wrapper exposes only immutable slices and counters.
// It has no interior mutation, and ParserAllocation never escapes mutably
// from it. MemoryReservation's shared ledger is synchronized.
unsafe impl Sync for FrozenParserAllocation {}

/// Borrowed, allocation-free proposal data.
///
/// Valid only while its frozen owner remains alive.
pub struct ProposalRef<'a> {
    /// Validated UTF-8 tool name.
    pub name: &'a str,

    /// UTF-8 JSON bytes validated by the parser before commit.
    pub arguments_json: &'a str,
}

impl FrozenParserAllocation {
    /// Number of retained proposals.
    pub fn proposal_count(&self) -> usize {
        self.storage.proposal_count
    }

    /// Retained slab charge.
    ///
    /// Scratch remains allocated until this owner drops.
    pub fn allocated_bytes(&self) -> usize {
        self.storage.allocated_bytes()
    }

    /// Ledger charge retained alongside delivery.
    pub fn reserved_bytes(&self) -> u64 {
        self.storage.reserved_bytes()
    }

    /// Borrow a committed descriptor without materializing Map/Value/String.
    pub fn proposal(&self, index: usize) -> Option<ProposalRef<'_>> {
        if index >= self.storage.proposal_count {
            return None;
        }

        let base = self.storage.base.as_ptr();
        let plan = self.storage.plan;

        // SAFETY: Only committed indexes are read.
        //
        // commit_validated_call wrote this descriptor and checked payload
        // bounds and UTF-8. freeze prevents all subsequent mutation.
        // Returned slices borrow this owner.
        unsafe {
            let slot = base
                .add(plan.proposals.offset)
                .cast::<ProposalSlot>()
                .add(index)
                .read();

            let payload = base.add(plan.payload.offset);

            let name = slice::from_raw_parts(payload.add(slot.name_start), slot.name_len);

            let args = slice::from_raw_parts(payload.add(slot.args_start), slot.args_len);

            Some(ProposalRef {
                name: str::from_utf8_unchecked(name),
                arguments_json: str::from_utf8_unchecked(args),
            })
        }
    }
}

#[cfg(test)]
#[path = "../tests/unit/parser_allocation_tests.rs"]
mod tests;

// W2_LAYOUT_AUDIT_AND_GRANT_TESTS_V1

/// One fixed slab region, including its preceding alignment padding.
#[derive(Clone, Copy, Debug)]
pub struct ParserRegionLayout {
    /// Region name.
    pub name: &'static str,
    /// Byte offset from the allocation base.
    pub offset: usize,
    /// Array element count.
    pub count: usize,
    /// Size of one array element.
    pub element_bytes: usize,
    /// Required element alignment.
    pub alignment: usize,
    /// Payload bytes, excluding preceding padding.
    pub bytes: usize,
    /// Bytes between the preceding region end and this region start.
    pub preceding_padding: usize,
}

/// Exact backing-layout report. This value is returned inline, not heap allocated.
#[derive(Clone, Copy, Debug)]
pub struct ParserLayoutReport {
    /// Fixed regions in backing allocation order.
    pub regions: [ParserRegionLayout; 8],
    /// Sum of element payload sizes, excluding padding.
    pub region_bytes: usize,
    /// Sum of alignment padding between regions.
    pub inter_region_padding: usize,
    /// Padding after the final region.
    pub trailing_padding: usize,
    /// Exact allocator-request size.
    pub total_bytes: usize,
    /// Alignment passed to the allocator.
    pub alignment: usize,
    /// Owner size when placed inline; not an additional slab allocation.
    pub inline_owner_bytes: usize,
}

impl Plan {
    fn checked_report(&self) -> Result<ParserLayoutReport, ParserAllocationError> {
        use std::mem::{align_of, size_of};
        fn description<T>(name: &'static str, region: Region) -> ParserRegionLayout {
            ParserRegionLayout {
                name,
                offset: region.offset,
                count: region.count,
                element_bytes: size_of::<T>(),
                alignment: align_of::<T>(),
                bytes: 0,
                preceding_padding: 0,
            }
        }
        // Independent of Layout::extend: this pass checks the result using
        // checked integer arithmetic and each element's actual ABI layout.
        let mut regions = [
            description::<u8>("carry", self.carry),
            description::<u8>("pending call", self.call),
            description::<u8>("decoded strings", self.decoded),
            description::<JsonFrame>("JSON stack", self.stack),
            description::<JsonKey>("duplicate keys", self.keys),
            description::<JsonNode>("JSON nodes", self.nodes),
            description::<u8>("proposal payload", self.payload),
            description::<ProposalSlot>("proposal descriptors", self.proposals),
        ];
        let mut cursor = 0usize;
        let mut region_bytes = 0usize;
        let mut padding = 0usize;
        let mut maximum_alignment = 1usize;
        for region in &mut regions {
            if region.count == 0 || !region.alignment.is_power_of_two() {
                return Err(ParserAllocationError::InvalidLayout);
            }
            let mask = region.alignment - 1;
            let expected = cursor
                .checked_add(mask)
                .map(|value| value & !mask)
                .ok_or(ParserAllocationError::LayoutOverflow)?;
            if region.offset != expected || region.offset % region.alignment != 0 {
                return Err(ParserAllocationError::InvalidLayout);
            }
            region.bytes = region
                .count
                .checked_mul(region.element_bytes)
                .ok_or(ParserAllocationError::LayoutOverflow)?;
            region.preceding_padding = region
                .offset
                .checked_sub(cursor)
                .ok_or(ParserAllocationError::InvalidLayout)?;
            padding = padding
                .checked_add(region.preceding_padding)
                .ok_or(ParserAllocationError::LayoutOverflow)?;
            region_bytes = region_bytes
                .checked_add(region.bytes)
                .ok_or(ParserAllocationError::LayoutOverflow)?;
            cursor = region
                .offset
                .checked_add(region.bytes)
                .ok_or(ParserAllocationError::LayoutOverflow)?;
            if cursor > self.layout.size() {
                return Err(ParserAllocationError::InvalidLayout);
            }
            maximum_alignment = maximum_alignment.max(region.alignment);
        }
        if maximum_alignment != self.layout.align() {
            return Err(ParserAllocationError::InvalidLayout);
        }
        let mask = maximum_alignment - 1;
        let total = cursor
            .checked_add(mask)
            .map(|value| value & !mask)
            .ok_or(ParserAllocationError::LayoutOverflow)?;
        if total != self.layout.size() || total > isize::MAX as usize {
            return Err(ParserAllocationError::InvalidLayout);
        }
        let trailing_padding = total
            .checked_sub(cursor)
            .ok_or(ParserAllocationError::InvalidLayout)?;
        let recomputed = region_bytes
            .checked_add(padding)
            .and_then(|value| value.checked_add(trailing_padding))
            .ok_or(ParserAllocationError::LayoutOverflow)?;
        if recomputed != total {
            return Err(ParserAllocationError::InvalidLayout);
        }
        Ok(ParserLayoutReport {
            regions,
            region_bytes,
            inter_region_padding: padding,
            trailing_padding,
            total_bytes: total,
            alignment: maximum_alignment,
            inline_owner_bytes: size_of::<ParserAllocation>(),
        })
    }
    fn validate_layout(&self) -> Result<(), ParserAllocationError> {
        self.checked_report().map(|_| ())
    }
}
impl ParserAllocation {
    /// Inspect every region and alignment byte before reserving/allocating.
    pub fn layout_report(
        limits: ParserAllocationLimits,
    ) -> Result<ParserLayoutReport, ParserAllocationError> {
        Plan::new(limits)?.checked_report()
    }
}

#[cfg(test)]
#[path = "../tests/unit/parser_allocation_layout_and_grant_integration.rs"]
mod layout_and_grant_integration_tests;

impl std::fmt::Debug for FrozenParserAllocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FrozenParserAllocation")
            .field("allocated_bytes", &self.allocated_bytes())
            .field("proposal_count", &self.proposal_count())
            .finish()
    }
}
