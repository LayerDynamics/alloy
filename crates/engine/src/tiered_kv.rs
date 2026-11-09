//! Bounded KV transfer and immutable, atomically published SSD snapshots.
//!
//! Native arrays stay on their execution worker. A separately admitted transfer
//! allowance covers one block and its serialized/native copies. The directory
//! catalog contains only paths; tensor files are read on demand. A snapshot is
//! visible only after every block and its manifest have been synced and the
//! temporary directory has been renamed. Cache failures are misses, never
//! permission to reuse unverified tensors.

use crate::{
    admission::{AdmissionController, MemoryReservation},
    error::{EngineError, MemoryLimitReason},
    family::{ExecutionControl, FamilyCache},
};
use inference_infra::kv_cache::{KvBlock, KvBlockLayer, KvCacheSpec};
use safetensors::{
    Dtype, SafeTensors,
    tensor::{TensorView, serialize_to_file},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::HashMap,
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
    time::{SystemTime, UNIX_EPOCH},
};

const FORMAT: u32 = 1;
const BLOCK_TOKENS: usize = 32;
const MAX_ENTRIES: usize = 64;
const MAX_TOKENS: usize = 131_072;
const MAX_MANIFEST_BYTES: u64 = 1024 * 1024;
const MAX_HEADER_BYTES: u64 = 64 * 1024;
const METADATA_ALLOWANCE: u64 = 8 * 1024 * 1024;
const PREFIX: &str = "kv-v1-";
static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);

/// Maximum retained idle KV reservation, independent of active execution needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HotCacheLimit {
    /// An absolute byte cap; zero keeps no unleased idle KV in RAM.
    Bytes(u64),
    /// An integer percentage of the current shared admission ceiling.
    Percent(u8),
}

impl HotCacheLimit {
    /// Parse bytes, KiB/MiB/GiB, or an integer percentage between 0 and 100.
    pub fn parse(value: &str) -> Result<Self, EngineError> {
        if let Some(number) = value.strip_suffix('%') {
            let percent = number
                .parse::<u8>()
                .map_err(|_| invalid("invalid hot-cache percentage"))?;
            return if percent <= 100 {
                Ok(Self::Percent(percent))
            } else {
                Err(invalid("hot-cache percentage exceeds 100"))
            };
        }
        let (number, multiplier) = [
            ("GiB", 1u64 << 30),
            ("MiB", 1u64 << 20),
            ("KiB", 1u64 << 10),
            ("B", 1),
        ]
        .into_iter()
        .find_map(|(suffix, factor)| value.strip_suffix(suffix).map(|number| (number, factor)))
        .unwrap_or((value, 1));
        let bytes = number
            .parse::<u64>()
            .ok()
            .and_then(|n| n.checked_mul(multiplier))
            .ok_or_else(|| invalid("invalid or overflowing hot-cache byte limit"))?;
        Ok(Self::Bytes(bytes))
    }

    /// Resolve the limit against a fresh admission-budget observation.
    pub fn bytes(self, budget: u64) -> u64 {
        match self {
            Self::Bytes(bytes) => bytes.min(budget),
            Self::Percent(percent) => {
                ((u128::from(budget) * u128::from(percent.min(100))) / 100) as u64
            }
        }
    }
}

/// Configuration applied before serving native execution requests.
#[derive(Clone, Debug)]
pub struct TieredKvConfig {
    /// Cold snapshot directory. None selects RAM-only caching.
    pub directory: Option<PathBuf>,
    /// Cap on retained idle KV; active and leased state remain owned and counted.
    pub hot_limit: HotCacheLimit,
    /// Disable SSD reads and writes while retaining the bounded RAM cache.
    pub hot_cache_only: bool,
    /// Dedicated admitted allowance for one transfer and its serialization.
    pub transfer_bytes: u64,
    /// Maximum total published cold-cache bytes, with oldest snapshots evicted.
    pub max_disk_bytes: u64,
}

impl TieredKvConfig {
    /// Configure 32-token blocks, a 64 MiB transfer allowance, and a 4 GiB SSD cap.
    pub fn new(directory: Option<PathBuf>, hot_limit: HotCacheLimit) -> Self {
        Self {
            directory,
            hot_limit,
            hot_cache_only: false,
            transfer_bytes: 64 * 1024 * 1024,
            max_disk_bytes: 4 * 1024 * 1024 * 1024,
        }
    }
}

/// Non-owning metadata published at native-worker boundaries.
#[derive(Clone, Debug, Default, Serialize)]
pub struct TieredKvSnapshot {
    /// Whether bounded hot retention has been configured.
    pub enabled: bool,
    /// Whether the SSD tier is enabled.
    pub disk_enabled: bool,
    /// Effective idle-retention limit at the last worker observation.
    pub hot_limit_bytes: u64,
    /// Current reservation bytes for retained native KV buffers.
    pub hot_reserved_bytes: u64,
    /// KV bytes protected by active work or a publication lease.
    pub protected_reserved_bytes: u64,
    /// Dedicated transfer allowance, held until the storage owner is dropped.
    pub transfer_reserved_bytes: u64,
    /// Complete cold snapshots discovered or published by this worker.
    pub cold_entries: usize,
    /// Complete snapshots successfully written during this process lifetime.
    pub spill_count: u64,
    /// Successful native KV restores during this process lifetime.
    pub restore_count: u64,
    /// Logical positions actually restored into native tensors.
    pub restored_tokens: u64,
    /// Idle native caches released by hot-limit or pressure reclamation.
    pub hot_evictions: u64,
    /// Last optional-cache failure; absence means no recorded failure.
    pub last_error: Option<String>,
}

#[derive(Clone)]
struct Entry {
    directory: PathBuf,
    modified: SystemTime,
    bytes: u64,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct StoredSpec {
    layers: usize,
    batch: usize,
    kv_heads: usize,
    head_dim: usize,
    capacity_tokens: usize,
}

impl From<KvCacheSpec> for StoredSpec {
    fn from(spec: KvCacheSpec) -> Self {
        Self {
            layers: spec.layers,
            batch: spec.batch,
            kv_heads: spec.kv_heads,
            head_dim: spec.head_dim,
            capacity_tokens: spec.capacity_tokens,
        }
    }
}
impl StoredSpec {
    fn native(&self) -> KvCacheSpec {
        KvCacheSpec {
            layers: self.layers,
            batch: self.batch,
            kv_heads: self.kv_heads,
            head_dim: self.head_dim,
            capacity_tokens: self.capacity_tokens,
        }
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct BlockRecord {
    start: usize,
    tokens: usize,
    bytes: u64,
    sha256: String,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    version: u32,
    identity: String,
    tokens: Vec<u32>,
    spec: StoredSpec,
    blocks: Vec<BlockRecord>,
}

/// Worker-local storage. The reservation drops after every owned catalog buffer.
pub(crate) struct TieredKvStore {
    pub(crate) config: TieredKvConfig,
    pub(crate) metrics: TieredKvSnapshot,
    directory: Option<PathBuf>,
    catalog: Vec<Entry>,
    _directory_lock: Option<File>,
    _transfer: Option<MemoryReservation>,
}

impl TieredKvStore {
    pub(crate) fn open(
        config: TieredKvConfig,
        admission: &AdmissionController,
    ) -> Result<Self, EngineError> {
        if matches!(config.hot_limit, HotCacheLimit::Percent(value) if value > 100) {
            return Err(invalid("hot-cache percentage exceeds 100"));
        }
        let disk = config.directory.is_some() && !config.hot_cache_only;
        if disk
            && (!(METADATA_ALLOWANCE + 128 * 1024..=128 * 1024 * 1024)
                .contains(&config.transfer_bytes)
                || config.max_disk_bytes == 0)
        {
            return Err(invalid(
                "SSD KV needs an 8.125–128 MiB transfer allowance and a positive disk cap",
            ));
        }
        // Acquire before allocating the directory catalog or transfer metadata.
        let transfer = if disk {
            Some(admission.reserve("tiered KV transfer allowance", config.transfer_bytes)?)
        } else {
            None
        };
        let directory = if disk {
            let path = config.directory.as_ref().expect("disk directory");
            fs::create_dir_all(path).map_err(io_error)?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                fs::set_permissions(path, fs::Permissions::from_mode(0o700)).map_err(io_error)?;
            }
            Some(path.canonicalize().map_err(io_error)?)
        } else {
            None
        };
        let directory_lock = if let Some(root) = &directory {
            let mut options = OpenOptions::new();
            options.read(true).write(true).create(true).truncate(false);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.mode(0o600);
            }
            let lock = options.open(root.join(".owner.lock")).map_err(io_error)?;
            lock.try_lock().map_err(|error| {
                invalid(format!(
                    "SSD KV directory already has an owner or cannot be locked: {error}"
                ))
            })?;
            // The exclusive process lock proves no live writer owns these
            // unpublished directories. A crash releases the lock, allowing
            // startup to remove its abandoned writes within the disk quota.
            for entry in fs::read_dir(root).map_err(io_error)? {
                let entry = entry.map_err(io_error)?;
                if entry
                    .file_name()
                    .to_str()
                    .and_then(|name| name.strip_prefix(".tmp-"))
                    .is_some_and(valid_entry_name)
                    && entry.file_type().map_err(io_error)?.is_dir()
                {
                    fs::remove_dir_all(entry.path()).map_err(io_error)?;
                }
            }
            Some(lock)
        } else {
            None
        };
        let mut store = Self {
            metrics: TieredKvSnapshot {
                enabled: true,
                disk_enabled: disk,
                transfer_reserved_bytes: transfer.as_ref().map_or(0, MemoryReservation::bytes),
                ..Default::default()
            },
            config,
            directory,
            catalog: Vec::new(),
            _directory_lock: directory_lock,
            _transfer: transfer,
        };
        store.scan()?;
        store.trim(0)?;
        Ok(store)
    }

    /// Reuse the same storage owner for idle-cap/quota changes. Different
    /// directories or transfer bounds need a newly admitted storage owner.
    pub(crate) fn update_limits(&mut self, config: &TieredKvConfig) -> Result<bool, EngineError> {
        if self.config.directory != config.directory
            || self.config.hot_cache_only != config.hot_cache_only
            || self.config.transfer_bytes != config.transfer_bytes
        {
            return Ok(false);
        }
        if matches!(config.hot_limit, HotCacheLimit::Percent(value) if value > 100)
            || (self.metrics.disk_enabled && config.max_disk_bytes == 0)
        {
            return Err(invalid("invalid hot-cache percentage or SSD quota"));
        }
        self.config.hot_limit = config.hot_limit;
        self.config.max_disk_bytes = config.max_disk_bytes;
        self.trim(0)?;
        Ok(true)
    }

    pub(crate) fn record_error(&mut self, error: impl std::fmt::Display) {
        self.metrics.last_error = Some(error.to_string().chars().take(1024).collect());
    }

    fn scan(&mut self) -> Result<(), EngineError> {
        let Some(root) = &self.directory else {
            return Ok(());
        };
        self.catalog.clear();
        for item in fs::read_dir(root).map_err(io_error)? {
            let item = item.map_err(io_error)?;
            let name = item.file_name();
            let Some(name) = name.to_str() else {
                continue;
            };
            if !valid_entry_name(name) || !item.file_type().map_err(io_error)?.is_dir() {
                continue;
            }
            let path = item.path();
            let Ok(metadata) = fs::symlink_metadata(path.join("manifest.json")) else {
                fs::remove_dir_all(&path).map_err(io_error)?;
                continue;
            };
            if !metadata.is_file() || metadata.len() > MAX_MANIFEST_BYTES {
                fs::remove_dir_all(&path).map_err(io_error)?;
                continue;
            }
            self.catalog.push(Entry {
                bytes: entry_bytes(&path)?,
                directory: path,
                modified: metadata.modified().unwrap_or(UNIX_EPOCH),
            });
            // Keep a bounded newest catalog even if another process left many entries.
            if self.catalog.len() > MAX_ENTRIES * 2 {
                self.catalog
                    .sort_by_key(|entry| std::cmp::Reverse(entry.modified));
                let old = self.catalog.split_off(MAX_ENTRIES);
                for entry in old {
                    fs::remove_dir_all(entry.directory).map_err(io_error)?;
                }
            }
        }
        self.catalog
            .sort_by_key(|entry| std::cmp::Reverse(entry.modified));
        self.metrics.cold_entries = self.catalog.len();
        Ok(())
    }

    fn trim(&mut self, incoming: u64) -> Result<(), EngineError> {
        loop {
            let total = self
                .catalog
                .iter()
                .try_fold(incoming, |sum, entry| sum.checked_add(entry.bytes))
                .ok_or_else(|| invalid("cold-cache size overflow"))?;
            if self.catalog.len() < MAX_ENTRIES && total <= self.config.max_disk_bytes {
                break;
            }
            let Some(oldest) = self.catalog.pop() else {
                break;
            };
            match fs::remove_dir_all(&oldest.directory) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(error) => return Err(io_error(error)),
            }
        }
        self.metrics.cold_entries = self.catalog.len();
        Ok(())
    }

    fn transfer_fits(&self, spec: KvCacheSpec, count: usize) -> Result<(), EngineError> {
        let plan = spec
            .block_payload_plan(count)
            .map_err(|error| invalid(error.to_string()))?;
        let bound = plan
            .host_bytes
            .checked_mul(4)
            .and_then(|n| n.checked_add(METADATA_ALLOWANCE))
            .ok_or_else(|| invalid("KV transfer bound overflow"))?;
        if bound > self.config.transfer_bytes {
            return Err(EngineError::MemoryLimitExceeded {
                operation: "tiered KV block transfer",
                required_bytes: bound,
                available_bytes: self.config.transfer_bytes,
                reason: MemoryLimitReason::BufferLimit,
            });
        }
        Ok(())
    }

    pub(crate) fn save(
        &mut self,
        identity: &str,
        tokens: &[u32],
        cache: &mut dyn FamilyCache,
    ) -> Result<bool, EngineError> {
        let Some(root) = self.directory.clone() else {
            return Ok(false);
        };
        let Some(spec) = cache.kv_block_spec() else {
            return Ok(false);
        };
        if tokens.is_empty()
            || tokens.len() != cache.len()
            || tokens.len() > MAX_TOKENS
            || !valid_digest(identity)
        {
            return Ok(false);
        }
        self.transfer_fits(spec, BLOCK_TOKENS.min(tokens.len()))?;
        let payload = spec
            .block_payload_plan(tokens.len())
            .map_err(|error| invalid(error.to_string()))?
            .host_bytes;
        let count = tokens.len().div_ceil(BLOCK_TOKENS);
        let estimate = payload
            .checked_add(count as u64 * MAX_HEADER_BYTES)
            .and_then(|n| n.checked_add(MAX_MANIFEST_BYTES))
            .ok_or_else(|| invalid("snapshot size overflow"))?;
        if estimate > self.config.max_disk_bytes {
            return Err(invalid("KV snapshot exceeds configured SSD cache cap"));
        }
        self.scan()?;
        self.trim(estimate)?;
        let nonce = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let id = digest(format!("{identity}:{now}:{}:{nonce}", std::process::id()).as_bytes());
        let temporary = root.join(format!(".tmp-{PREFIX}{id}"));
        fs::create_dir(&temporary).map_err(io_error)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&temporary, fs::Permissions::from_mode(0o700)).map_err(io_error)?;
        }
        let mut cleanup = TemporaryDirectory(Some(temporary.clone()));
        let mut blocks = Vec::with_capacity(count);
        for (index, start) in (0..tokens.len()).step_by(BLOCK_TOKENS).enumerate() {
            let end = (start + BLOCK_TOKENS).min(tokens.len());
            self.transfer_fits(spec, end - start)?;
            let block = cache.export_kv_block(start, end)?;
            let path = temporary.join(format!("{index:08}.safetensors"));
            let prefix_digest = token_digest(identity, &tokens[..end]);
            write_block(&path, &block, identity, &prefix_digest)?;
            let bytes = fs::metadata(&path).map_err(io_error)?.len();
            let hash = hash_file(&path, bytes)?;
            blocks.push(BlockRecord {
                start,
                tokens: end - start,
                bytes,
                sha256: hash,
            });
        }
        let manifest = Manifest {
            version: FORMAT,
            identity: identity.to_owned(),
            tokens: tokens.to_vec(),
            spec: spec.into(),
            blocks,
        };
        let encoded = serde_json::to_vec(&manifest).map_err(|error| invalid(error.to_string()))?;
        if encoded.len() as u64 > MAX_MANIFEST_BYTES {
            return Err(invalid("KV manifest exceeds metadata allowance"));
        }
        let manifest_path = temporary.join("manifest.json");
        let mut file = private_create(&manifest_path)?;
        file.write_all(&encoded).map_err(io_error)?;
        file.sync_all().map_err(io_error)?;
        File::open(&temporary)
            .map_err(io_error)?
            .sync_all()
            .map_err(io_error)?;
        let published = root.join(format!("{PREFIX}{id}"));
        fs::rename(&temporary, &published).map_err(io_error)?;
        cleanup.0 = None;
        File::open(&root)
            .map_err(io_error)?
            .sync_all()
            .map_err(io_error)?;
        self.catalog.insert(
            0,
            Entry {
                bytes: entry_bytes(&published)?,
                directory: published,
                modified: SystemTime::now(),
            },
        );
        self.metrics.cold_entries = self.catalog.len();
        self.metrics.spill_count = self.metrics.spill_count.saturating_add(1);
        Ok(true)
    }

    pub(crate) fn restore(
        &mut self,
        identity: &str,
        input: &[u32],
        cache: &mut dyn FamilyCache,
        control: &ExecutionControl,
    ) -> Result<usize, EngineError> {
        if self.directory.is_none() || input.len() < 2 || cache.len() != 0 {
            return Ok(0);
        }
        let Some(destination) = cache.kv_block_spec() else {
            return Ok(0);
        };
        // Keep one real input token for prefill, which produces fresh final logits.
        let mut best: Option<(PathBuf, Manifest, usize)> = None;
        for entry in &self.catalog {
            control.check_cancelled()?;
            let manifest = match read_manifest(&entry.directory.join("manifest.json")) {
                Ok(manifest) => manifest,
                Err(error) => {
                    self.metrics.last_error = Some(error.to_string().chars().take(1024).collect());
                    continue;
                }
            };
            if manifest.identity != identity
                || manifest.version != FORMAT
                || !same_layout(manifest.spec.native(), destination)
            {
                continue;
            }
            let common = manifest
                .tokens
                .iter()
                .zip(input)
                .take_while(|(left, right)| left == right)
                .count()
                .min(input.len() - 1);
            if common != 0 && best.as_ref().is_none_or(|(_, _, count)| common > *count) {
                best = Some((entry.directory.clone(), manifest, common));
            }
        }
        let Some((directory, manifest, matched)) = best else {
            return Ok(0);
        };
        let restored = (|| {
            for (index, record) in manifest.blocks.iter().enumerate() {
                if record.start >= matched {
                    break;
                }
                control.check_cancelled()?;
                self.transfer_fits(manifest.spec.native(), record.tokens)?;
                let end = record.start + record.tokens;
                let prefix_digest = token_digest(identity, &manifest.tokens[..end]);
                let mut block = read_block(
                    &directory.join(format!("{index:08}.safetensors")),
                    record,
                    manifest.spec.native(),
                    identity,
                    &prefix_digest,
                )?;
                if end > matched {
                    block
                        .truncate_tokens(matched - record.start)
                        .map_err(|error| invalid(error.to_string()))?;
                }
                cache.restore_kv_block(&block)?;
                control.check_cancelled()?;
            }
            if cache.len() != matched {
                return Err(invalid("restored KV length differs from verified prefix"));
            }
            Ok(matched)
        })();
        match restored {
            Ok(count) => {
                self.metrics.restore_count = self.metrics.restore_count.saturating_add(1);
                self.metrics.restored_tokens =
                    self.metrics.restored_tokens.saturating_add(count as u64);
                Ok(count)
            }
            Err(error) => {
                // A partial import must never be observed as a cache hit. Native
                // invalidation remains an error so the runtime discards that cache.
                cache.reset()?;
                if control.is_cancelled() {
                    return Err(error);
                }
                self.record_error(error);
                Ok(0)
            }
        }
    }
}

fn read_manifest(path: &Path) -> Result<Manifest, EngineError> {
    let bytes = bounded_read(path, MAX_MANIFEST_BYTES)?;
    let manifest: Manifest =
        serde_json::from_slice(&bytes).map_err(|error| invalid(error.to_string()))?;
    if manifest.version != FORMAT
        || !valid_digest(&manifest.identity)
        || manifest.tokens.is_empty()
        || manifest.tokens.len() > MAX_TOKENS
        || manifest.tokens.len() > manifest.spec.capacity_tokens
        || manifest.spec.layers == 0
        || manifest.spec.layers > 256
        || manifest.spec.batch == 0
        || manifest.spec.batch > 16
        || manifest.spec.kv_heads == 0
        || manifest.spec.kv_heads > 256
        || manifest.spec.head_dim == 0
        || manifest.spec.head_dim > 4096
    {
        return Err(invalid("invalid KV manifest bounds"));
    }
    if manifest.blocks.len() != manifest.tokens.len().div_ceil(BLOCK_TOKENS) {
        return Err(invalid("incomplete KV block table"));
    }
    let mut end = 0;
    for block in &manifest.blocks {
        if block.start != end
            || block.tokens == 0
            || block.tokens > BLOCK_TOKENS
            || !valid_digest(&block.sha256)
        {
            return Err(invalid("invalid KV block table"));
        }
        end = end
            .checked_add(block.tokens)
            .ok_or_else(|| invalid("KV block length overflow"))?;
        if end > manifest.tokens.len() {
            return Err(invalid("KV block exceeds token prefix"));
        }
    }
    if end != manifest.tokens.len() {
        return Err(invalid("incomplete KV prefix"));
    }
    Ok(manifest)
}

fn write_block(
    path: &Path,
    block: &KvBlock,
    identity: &str,
    prefix_digest: &str,
) -> Result<(), EngineError> {
    block
        .payload_plan()
        .map_err(|error| invalid(error.to_string()))?;
    let shape = vec![
        block.spec.batch,
        block.spec.kv_heads,
        block.tokens,
        block.spec.head_dim,
    ];
    let mut buffers = Vec::with_capacity(block.layers.len() * 2);
    for (index, layer) in block.layers.iter().enumerate() {
        for (kind, values) in [("keys", &layer.keys), ("values", &layer.values)] {
            buffers.push((
                format!("layer.{index}.{kind}"),
                values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<u8>>(),
            ));
        }
    }
    let views: Vec<_> = buffers
        .iter()
        .map(|(name, bytes)| {
            TensorView::new(Dtype::F32, shape.clone(), bytes).map(|view| (name.as_str(), view))
        })
        .collect::<Result<_, _>>()
        .map_err(|error| invalid(error.to_string()))?;
    let metadata = HashMap::from([
        ("format".to_owned(), FORMAT.to_string()),
        ("identity".to_owned(), identity.to_owned()),
        ("token_prefix_sha256".to_owned(), prefix_digest.to_owned()),
        ("start".to_owned(), block.start.to_string()),
    ]);
    // create_new establishes ownership; serialize_to_file opens only this newly
    // created path inside the private, unpublished snapshot directory.
    drop(private_create(path)?);
    serialize_to_file(views, Some(metadata), path).map_err(|error| invalid(error.to_string()))?;
    File::open(path)
        .map_err(io_error)?
        .sync_all()
        .map_err(io_error)?;
    Ok(())
}

fn read_block(
    path: &Path,
    record: &BlockRecord,
    spec: KvCacheSpec,
    identity: &str,
    prefix_digest: &str,
) -> Result<KvBlock, EngineError> {
    let payload = spec
        .block_payload_plan(record.tokens)
        .map_err(|error| invalid(error.to_string()))?
        .host_bytes;
    let limit = payload
        .checked_add(MAX_HEADER_BYTES + 8)
        .ok_or_else(|| invalid("KV file length overflow"))?;
    if record.bytes > limit {
        return Err(invalid("KV block file exceeds expected payload"));
    }
    let bytes = bounded_read(path, limit)?;
    if bytes.len() as u64 != record.bytes || digest(&bytes) != record.sha256 || bytes.len() < 8 {
        return Err(invalid("KV block checksum or size mismatch"));
    }
    let header = u64::from_le_bytes(bytes[..8].try_into().expect("eight bytes"));
    if header > MAX_HEADER_BYTES {
        return Err(invalid("KV tensor header exceeds allowance"));
    }
    let (_, metadata) =
        SafeTensors::read_metadata(&bytes).map_err(|error| invalid(error.to_string()))?;
    let fields = metadata
        .metadata()
        .as_ref()
        .ok_or_else(|| invalid("missing KV block identity"))?;
    if fields.get("identity").map(String::as_str) != Some(identity)
        || fields.get("token_prefix_sha256").map(String::as_str) != Some(prefix_digest)
        || fields.get("format") != Some(&FORMAT.to_string())
        || fields.get("start") != Some(&record.start.to_string())
    {
        return Err(invalid("KV block identity or prefix mismatch"));
    }
    let tensors = SafeTensors::deserialize(&bytes).map_err(|error| invalid(error.to_string()))?;
    if tensors.len() != spec.layers * 2 {
        return Err(invalid("KV block has incorrect layer count"));
    }
    let shape = [spec.batch, spec.kv_heads, record.tokens, spec.head_dim];
    let mut layers = Vec::with_capacity(spec.layers);
    for index in 0..spec.layers {
        let read = |kind: &str| -> Result<Vec<f32>, EngineError> {
            let tensor = tensors
                .tensor(&format!("layer.{index}.{kind}"))
                .map_err(|error| invalid(error.to_string()))?;
            if tensor.dtype() != Dtype::F32 || tensor.shape() != shape {
                return Err(invalid("KV tensor dtype or shape mismatch"));
            }
            let values: Vec<f32> = tensor
                .data()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|&chunk| f32::from_le_bytes(chunk))
                .collect();
            if values.iter().any(|value| !value.is_finite()) {
                return Err(invalid("KV tensor contains nonfinite values"));
            }
            Ok(values)
        };
        layers.push(KvBlockLayer {
            keys: read("keys")?,
            values: read("values")?,
        });
    }
    Ok(KvBlock {
        start: record.start,
        tokens: record.tokens,
        spec,
        layers,
    })
}

fn same_layout(left: KvCacheSpec, right: KvCacheSpec) -> bool {
    left.layers == right.layers
        && left.batch == right.batch
        && left.kv_heads == right.kv_heads
        && left.head_dim == right.head_dim
}
fn valid_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|value| value.is_ascii_digit() || (b'a'..=b'f').contains(&value))
}
fn valid_entry_name(name: &str) -> bool {
    name.strip_prefix(PREFIX).is_some_and(valid_digest)
}
pub(crate) fn digest(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
fn token_digest(identity: &str, tokens: &[u32]) -> String {
    let mut hash = Sha256::new();
    hash.update(b"alloy.persisted-kv-prefix.v1\0");
    hash.update(identity.as_bytes());
    for token in tokens {
        hash.update(token.to_le_bytes());
    }
    hash.finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
fn hash_file(path: &Path, limit: u64) -> Result<String, EngineError> {
    let mut file = File::open(path)
        .map_err(io_error)?
        .take(limit.saturating_add(1));
    let mut hash = Sha256::new();
    let mut buffer = [0u8; 16 * 1024];
    let mut total = 0u64;
    loop {
        let count = file.read(&mut buffer).map_err(io_error)?;
        if count == 0 {
            break;
        }
        total += count as u64;
        hash.update(&buffer[..count]);
    }
    if total != limit {
        return Err(invalid("KV file changed while hashing"));
    }
    Ok(hash
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}
fn private_create(path: &Path) -> Result<File, EngineError> {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options.open(path).map_err(io_error)
}
fn bounded_read(path: &Path, limit: u64) -> Result<Vec<u8>, EngineError> {
    let metadata = fs::symlink_metadata(path).map_err(io_error)?;
    if !metadata.is_file() || metadata.len() > limit {
        return Err(invalid(
            "KV cache file exceeds bound or is not a regular file",
        ));
    }
    let file = File::open(path).map_err(io_error)?;
    let length = file.metadata().map_err(io_error)?.len();
    if length > limit {
        return Err(invalid("KV cache file grew beyond its bound"));
    }
    let mut bytes = Vec::with_capacity(length as usize);
    file.take(limit + 1)
        .read_to_end(&mut bytes)
        .map_err(io_error)?;
    if bytes.len() as u64 > limit {
        return Err(invalid("KV cache file grew during read"));
    }
    Ok(bytes)
}
fn entry_bytes(path: &Path) -> Result<u64, EngineError> {
    fs::read_dir(path)
        .map_err(io_error)?
        .try_fold(0u64, |total, entry| {
            let entry = entry.map_err(io_error)?;
            let metadata = fs::symlink_metadata(entry.path()).map_err(io_error)?;
            total
                .checked_add(metadata.len())
                .ok_or_else(|| invalid("cold-cache byte count overflow"))
        })
}
struct TemporaryDirectory(Option<PathBuf>);
impl Drop for TemporaryDirectory {
    fn drop(&mut self) {
        if let Some(path) = &self.0 {
            let _ = fs::remove_dir_all(path);
        }
    }
}
fn invalid(message: impl Into<String>) -> EngineError {
    EngineError::invalid_request("tiered_kv", message)
}
fn io_error(error: std::io::Error) -> EngineError {
    EngineError::ExecutionFailed {
        stage: "tiered KV storage",
        message: error.to_string(),
    }
}

#[cfg(test)]
#[path = "../tests/unit/tiered_kv_tests.rs"]
mod tests;
