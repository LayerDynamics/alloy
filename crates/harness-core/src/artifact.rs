//! Bounded W3 artifacts. Filesystem publication precedes SQLite registration.
//!
//! The configured local root and its parent are daemon-owned. This module rejects
//! symlinks/aliases at managed entries; it does not defend against malicious
//! same-user replacement of directories between system calls. No transaction or
//! SessionStore borrow is retained while streaming or synchronizing file bytes.

use crate::{
    approval::CONTRACT_VERSION,
    error::{HarnessError, HarnessResult},
    store::SessionStore,
};
use rusqlite::{OptionalExtension, params};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File, OpenOptions},
    io::{self, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex, MutexGuard},
    time::{SystemTime, UNIX_EPOCH},
};

/// Frozen maximum payload bytes per artifact.
pub const MAX_ARTIFACT_BYTES: u64 = 16 * 1024 * 1024;
/// Frozen source-byte ceiling for a preview.
pub const MAX_PREVIEW_BYTES: usize = 4096;
/// Fixed streaming/verification buffer size.
pub const COPY_BUFFER_BYTES: usize = 64 * 1024;

/// Initial defaults and upper bounds. Configuration may only lower them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArtifactLimits {
    /// Per-object ceiling, also tightened by the invocation's output limit.
    pub max_artifact_bytes: u64,
    /// Total logical managed bytes including active publication reservations.
    pub max_managed_bytes: u64,
    /// Managed file entries, including temporary/publication reservations.
    pub max_managed_files: u64,
    /// Active streaming writers.
    pub max_writers: u64,
}
impl Default for ArtifactLimits {
    fn default() -> Self {
        Self {
            max_artifact_bytes: MAX_ARTIFACT_BYTES,
            max_managed_bytes: 1024 * 1024 * 1024,
            max_managed_files: 4096,
            max_writers: 8,
        }
    }
}
impl ArtifactLimits {
    fn validate(self) -> HarnessResult<Self> {
        let max = Self::default();
        if self.max_artifact_bytes > max.max_artifact_bytes
            || self.max_managed_bytes == 0
            || self.max_managed_bytes > max.max_managed_bytes
            || self.max_managed_files == 0
            || self.max_managed_files > max.max_managed_files
            || self.max_writers == 0
            || self.max_writers > max.max_writers
        {
            return Err(error(
                "artifact limits exceed frozen ceilings or have zero capacity",
            ));
        }
        Ok(self)
    }
}

/// Audit metadata, not an authorization capability or a ToolResult wire envelope.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ArtifactMetadata {
    /// Opaque random identity, never an input pathname.
    pub artifact_ref: String,
    /// Owning session.
    pub session_id: String,
    /// Originating daemon invocation.
    pub call_id: String,
    /// Complete accepted byte count, never a truncation claim.
    pub byte_length: u64,
    /// SHA-256 of the complete bytes.
    pub sha256: String,
    /// Bounded text/base64 encoding of preview source bytes; untrusted and partial.
    pub preview_bytes: String,
    /// Either utf8 or base64.
    pub preview_encoding: String,
    /// Unix epoch audit timestamp in milliseconds.
    pub created_at_unix_ms: i64,
    /// W3 metadata interpretation.
    pub contract_version: String,
}

#[derive(Default)]
struct Budget {
    bytes: u64,
    files: u64,
    writers: u64,
    poisoned: bool,
}
/// Shared per-store state; retains exclusive daemon ownership through writers.
pub(crate) struct Shared {
    root: PathBuf,
    db: PathBuf,
    store_id: String,
    limits: ArtifactLimits,
    budget: Mutex<Budget>,
    _lease: Arc<dyn Send + Sync>,
    #[cfg(test)]
    fault: Mutex<Option<&'static str>>,
}
impl Shared {
    fn budget(&self) -> HarnessResult<MutexGuard<'_, Budget>> {
        let guard = self
            .budget
            .lock()
            .map_err(|_| error("artifact accounting lock poisoned"))?;
        if guard.poisoned {
            return Err(error("artifact store requires reopen/reconciliation"));
        }
        Ok(guard)
    }
    fn poison(&self) {
        if let Ok(mut b) = self.budget.lock() {
            b.poisoned = true;
        }
    }
    #[cfg(test)]
    fn checkpoint(&self, stage: &'static str) -> HarnessResult<()> {
        if std::env::var("ALLOY_ARTIFACT_CRASH_STAGE").ok().as_deref() == Some(stage) {
            std::process::exit(77);
        }
        if *self.fault.lock().unwrap() == Some(stage) {
            return Err(error(format!("injected {stage}")));
        }
        Ok(())
    }
}

/// Cloneable admission facade. Clones share quotas, writer limits and ownership.
#[derive(Clone)]
pub struct ArtifactStore {
    shared: Arc<Shared>,
}
/// Private-file stream. Any write error permanently prevents publication.
pub struct ArtifactWriter {
    shared: Arc<Shared>,
    session: String,
    call: String,
    epoch: String,
    reference: String,
    temp: PathBuf,
    file: Option<File>,
    length: u64,
    max_bytes: u64,
    reserved_bytes: u64,
    hash: Sha256,
    preview: Vec<u8>,
    failed: bool,
    published: bool,
    settled: bool,
}
/// Complete-length/digest verified read handle. Verification precedes exposure.
pub struct VerifiedArtifact {
    file: File,
    metadata: ArtifactMetadata,
    _lease: Arc<Shared>,
}
/// Ownership-checked lookup, not yet integrity-verified. Release the store lock
/// before calling verify; this value owns no SessionStore borrow.
pub struct UnverifiedArtifact {
    shared: Arc<Shared>,
    metadata: ArtifactMetadata,
}
impl UnverifiedArtifact {
    /// Read/hash the complete object without a database/store lock, then expose it.
    pub fn verify(self) -> HarnessResult<VerifiedArtifact> {
        {
            let _budget = self.shared.budget()?;
        }
        let file = verify(
            &self
                .shared
                .root
                .join("objects")
                .join(&self.metadata.artifact_ref),
            &self.metadata,
        )?;
        Ok(VerifiedArtifact {
            file,
            metadata: self.metadata,
            _lease: self.shared,
        })
    }
}
/// Fully synchronized publication awaiting a short database metadata transaction.
/// This value exposes no usable artifact reference. Dropping it requires recovery.
pub struct PendingArtifact {
    writer: ArtifactWriter,
    metadata: ArtifactMetadata,
}
impl VerifiedArtifact {
    /// Metadata whose complete object was verified before returning this handle.
    pub fn metadata(&self) -> &ArtifactMetadata {
        &self.metadata
    }
}
impl Read for VerifiedArtifact {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        self.file.read(bytes)
    }
}

fn error(message: impl Into<String>) -> HarnessError {
    HarnessError::Store(message.into())
}
fn ioe(error: io::Error) -> HarnessError {
    self::error(format!("artifact I/O: {error}"))
}
fn valid_ref(value: &str) -> bool {
    value.len() == 37
        && value.starts_with("art1_")
        && value.as_bytes()[5..]
            .iter()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(b))
}
fn valid_temp(value: &str) -> bool {
    value.len() == 34
        && value.starts_with("t_")
        && value.as_bytes()[2..]
            .iter()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(b))
}
fn random_hex() -> HarnessResult<String> {
    let mut bytes = [0u8; 16];
    File::open("/dev/urandom")
        .map_err(ioe)?
        .read_exact(&mut bytes)
        .map_err(ioe)?;
    Ok(bytes.iter().map(|b| format!("{b:02x}")).collect())
}
fn timestamp() -> HarnessResult<i64> {
    i64::try_from(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|_| error("system clock before epoch"))?
            .as_millis(),
    )
    .map_err(|_| error("artifact timestamp overflow"))
}
fn sync_dir(path: &Path) -> HarnessResult<()> {
    File::open(path).map_err(ioe)?.sync_all().map_err(ioe)
}
fn open_regular(path: &Path) -> HarnessResult<File> {
    open_regular_pair(path, false)
}
fn open_regular_pair(path: &Path, pair: bool) -> HarnessResult<File> {
    let before = fs::symlink_metadata(path).map_err(ioe)?;
    if !before.is_file() || before.file_type().is_symlink() {
        return Err(error("nonregular artifact object"));
    }
    let file = File::open(path).map_err(ioe)?;
    let after = file.metadata().map_err(ioe)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        let links = if pair { 2 } else { 1 };
        if before.dev() != after.dev()
            || before.ino() != after.ino()
            || before.nlink() != links
            || after.nlink() != links
        {
            return Err(error("managed artifact aliased or changed during open"));
        }
    }
    #[cfg(not(unix))]
    let _ = (before, after, pair);
    Ok(file)
}
fn private_file(path: &Path) -> HarnessResult<File> {
    let mut options = OpenOptions::new();
    options.write(true).read(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options.open(path).map_err(ioe)
}
fn directory(path: &Path) -> HarnessResult<()> {
    match fs::symlink_metadata(path) {
        Ok(meta) => {
            if !meta.is_dir() || meta.file_type().is_symlink() {
                return Err(error("artifact root/subdirectory is not a real directory"));
            }
        }
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            let mut builder = fs::DirBuilder::new();
            #[cfg(unix)]
            {
                use std::os::unix::fs::DirBuilderExt;
                builder.mode(0o700);
            }
            builder.create(path).map_err(ioe)?;
            sync_dir(
                path.parent()
                    .ok_or_else(|| error("directory has no parent"))?,
            )?;
        }
        Err(e) => return Err(ioe(e)),
    }
    Ok(())
}
fn marker(path: &Path, expected: &[u8]) -> HarnessResult<()> {
    if path.exists() || fs::symlink_metadata(path).is_ok() {
        let mut file = open_regular(path)?;
        let mut data = Vec::new();
        Read::by_ref(&mut file)
            .take(16_385)
            .read_to_end(&mut data)
            .map_err(ioe)?;
        if data != expected {
            return Err(error(
                "artifact root/database binding mismatch or incomplete marker",
            ));
        }
    } else {
        let mut file = private_file(path)?;
        file.write_all(expected).map_err(ioe)?;
        file.sync_all().map_err(ioe)?;
        sync_dir(path.parent().ok_or_else(|| error("marker has no parent"))?)?;
    }
    Ok(())
}
fn encode64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let a = chunk[0];
        let b = chunk.get(1).copied().unwrap_or(0);
        let c = chunk.get(2).copied().unwrap_or(0);
        out.push(ALPHABET[(a >> 2) as usize] as char);
        out.push(ALPHABET[(((a & 3) << 4) | (b >> 4)) as usize] as char);
        out.push(if chunk.len() > 1 {
            ALPHABET[(((b & 15) << 2) | (c >> 6)) as usize] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            ALPHABET[(c & 63) as usize] as char
        } else {
            '='
        });
    }
    out
}
fn preview(bytes: &[u8]) -> (String, String) {
    match std::str::from_utf8(bytes) {
        Ok(s) => (s.to_owned(), "utf8".into()),
        Err(e) if e.error_len().is_none() => (
            String::from_utf8(bytes[..e.valid_up_to()].to_vec()).expect("validated prefix"),
            "utf8".into(),
        ),
        Err(_) => (encode64(bytes), "base64".into()),
    }
}
fn verify(path: &Path, metadata: &ArtifactMetadata) -> HarnessResult<File> {
    verify_pair(path, metadata, false)
}
fn verify_pair(path: &Path, metadata: &ArtifactMetadata, pair: bool) -> HarnessResult<File> {
    if !valid_ref(&metadata.artifact_ref)
        || metadata.contract_version != CONTRACT_VERSION
        || metadata.byte_length > MAX_ARTIFACT_BYTES
    {
        return Err(error("invalid committed artifact metadata"));
    }
    let mut file = open_regular_pair(path, pair)?;
    if file.metadata().map_err(ioe)?.len() != metadata.byte_length {
        return Err(error("artifact length mismatch"));
    }
    let mut buffer = [0u8; COPY_BUFFER_BYTES];
    let mut hash = Sha256::new();
    let mut length = 0u64;
    let mut prefix = Vec::with_capacity(MAX_PREVIEW_BYTES);
    loop {
        let n = file.read(&mut buffer).map_err(ioe)?;
        if n == 0 {
            break;
        }
        length += n as u64;
        if length > metadata.byte_length {
            return Err(error("artifact changed while verifying"));
        }
        hash.update(&buffer[..n]);
        let count = (MAX_PREVIEW_BYTES - prefix.len()).min(n);
        prefix.extend_from_slice(&buffer[..count]);
    }
    if length != metadata.byte_length || format!("{:x}", hash.finalize()) != metadata.sha256 {
        return Err(error("artifact digest mismatch"));
    }
    let (bytes, encoding) = preview(&prefix);
    if bytes != metadata.preview_bytes || encoding != metadata.preview_encoding {
        return Err(error("artifact preview metadata mismatch"));
    }
    file.seek(SeekFrom::Start(0)).map_err(ioe)?;
    Ok(file)
}
fn metadata_rows(store: &SessionStore) -> HarnessResult<BTreeMap<String, ArtifactMetadata>> {
    let mut stmt = store
        .conn
        .prepare("SELECT artifact_ref,session_id,call_id,metadata_json FROM tool_artifacts")?;
    let mut rows = stmt.query([])?;
    let mut result = BTreeMap::new();
    while let Some(row) = rows.next()? {
        if result.len() >= 4096 {
            return Err(error("artifact metadata exceeds frozen file ceiling"));
        }
        let (reference, session, call, json): (String, String, String, String) =
            (row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?);
        if json.len() > 1_048_576 {
            return Err(error("artifact metadata envelope too large"));
        }
        let m: ArtifactMetadata = serde_json::from_str(&json)?;
        if reference != m.artifact_ref
            || session != m.session_id
            || call != m.call_id
            || !valid_ref(&reference)
        {
            return Err(error("artifact ownership metadata mismatch"));
        }
        result.insert(reference, m);
    }
    drop(rows);
    drop(stmt);
    // New W3 artifact events and metadata must agree before cleanup is authorized.
    let mut stmt = store
        .conn
        .prepare("SELECT payload_json,artifact_refs FROM events WHERE kind='artifact.created'")?;
    let mut rows = stmt.query([])?;
    let mut seen = BTreeSet::new();
    while let Some(row) = rows.next()? {
        let json: String = row.get(0)?;
        if json.len() > 1_048_576 {
            return Err(error("artifact event envelope too large"));
        }
        let value: serde_json::Value = serde_json::from_str(&json)?;
        if value["contract_version"].as_str() != Some(CONTRACT_VERSION) {
            continue;
        }
        let m: ArtifactMetadata = serde_json::from_value(value)?;
        let refs: serde_json::Value = serde_json::from_str(&row.get::<_, String>(1)?)?;
        if result.get(&m.artifact_ref) != Some(&m)
            || !seen.insert(m.artifact_ref.clone())
            || refs != serde_json::json!([m.artifact_ref])
        {
            return Err(error(
                "artifact event/metadata proof mismatch; cleanup refused",
            ));
        }
    }
    if seen.len() != result.len() {
        return Err(error("artifact metadata has no unique creation event"));
    }
    Ok(result)
}
#[cfg(unix)]
type InodeIdentity = (u64, u64);

// Managed filename, temporary-name flag, and observed link count.
#[cfg(unix)]
type ManagedInodeEntry = (String, bool, u64);

#[cfg(unix)]
type InodeInventory = BTreeMap<InodeIdentity, Vec<ManagedInodeEntry>>;

struct Inventory {
    temps: Vec<PathBuf>,
    objects: BTreeSet<String>,
    paired: BTreeSet<String>,
}
fn inventory(root: &Path) -> HarnessResult<Inventory> {
    for entry in fs::read_dir(root).map_err(ioe)? {
        let name = entry.map_err(ioe)?.file_name();
        if name != ".alloy-store-id" && name != "tmp" && name != "objects" {
            return Err(error("unknown artifact root entry; cleanup refused"));
        }
    }
    let mut temps = Vec::new();
    let mut objects = BTreeSet::new();
    let mut bytes = 0u64;
    let mut files = 0u64;
    let mut paired = BTreeSet::new();
    #[cfg(unix)]
    let mut inodes: InodeInventory = BTreeMap::new();
    for (sub, temporary) in [("tmp", true), ("objects", false)] {
        for entry in fs::read_dir(root.join(sub)).map_err(ioe)? {
            let entry = entry.map_err(ioe)?;
            let name = entry
                .file_name()
                .into_string()
                .map_err(|_| error("non-UTF8 managed filename"))?;
            if !(if temporary {
                valid_temp(&name)
            } else {
                valid_ref(&name)
            }) {
                return Err(error("unknown managed filename; cleanup refused"));
            }
            let m = fs::symlink_metadata(entry.path()).map_err(ioe)?;
            if !m.is_file() || m.file_type().is_symlink() {
                return Err(error("nonregular managed entry; cleanup refused"));
            }
            bytes = bytes
                .checked_add(m.len())
                .ok_or_else(|| error("artifact size overflow"))?;
            files += 1;
            if files > 4096 || bytes > 1024 * 1024 * 1024 {
                return Err(error(
                    "on-disk artifact quota exceeded; manual inspection required",
                ));
            }
            #[cfg(unix)]
            {
                use std::os::unix::fs::MetadataExt;
                if m.nlink() > 2 {
                    return Err(error("external artifact hardlinks; cleanup refused"));
                }
                inodes.entry((m.dev(), m.ino())).or_default().push((
                    name.clone(),
                    temporary,
                    m.nlink(),
                ));
            }
            if temporary {
                temps.push(entry.path());
            } else {
                objects.insert(name);
            }
        }
    }
    #[cfg(unix)]
    for group in inodes.values() {
        if group.len() == 1 && group[0].2 == 1 {
            continue;
        }
        if group.len() != 2 || group.iter().any(|v| v.2 != 2) || group[0].1 == group[1].1 {
            return Err(error("unproven hardlink alias; cleanup refused"));
        }
        paired.insert(group.iter().find(|v| !v.1).expect("one object").0.clone());
    }
    Ok(Inventory {
        temps,
        objects,
        paired,
    })
}

impl ArtifactStore {
    /// Bind one explicit root to this exclusively owned database and reconcile it.
    /// Repeated opens in the same epoch share the same accounting state.
    pub fn open(
        store: &mut SessionStore,
        root: impl AsRef<Path>,
        limits: ArtifactLimits,
    ) -> HarnessResult<Self> {
        if !cfg!(unix) {
            return Err(error(
                "artifact publication currently supports Unix local filesystems only",
            ));
        }
        let limits = limits.validate()?;
        let root = root.as_ref();
        let name = root
            .file_name()
            .ok_or_else(|| error("artifact root name required"))?;
        let parent = root
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        let parent = fs::canonicalize(parent).map_err(ioe)?;
        let root = parent.join(name);
        if let Some(shared) = &store.artifact_state {
            if shared.root != root || shared.limits != limits {
                return Err(error("artifact root/limits cannot change in a live epoch"));
            }
            drop(shared.budget()?);
            return Ok(Self {
                shared: Arc::clone(shared),
            });
        }
        directory(&root)?;
        let store_id: String =
            store
                .conn
                .query_row("SELECT store_id FROM runtime_metadata", [], |r| r.get(0))?;
        marker(&root.join(".alloy-store-id"), store_id.as_bytes())?;
        let mut bound = store.path().as_os_str().to_os_string();
        bound.push(".artifact-root");
        let binding = serde_json::to_vec(
            &serde_json::json!({"store_id":store_id,"root":root.to_str().ok_or_else(||error("root must be UTF8"))?}),
        )?;
        if binding.len() > 16_384 {
            return Err(error("artifact root binding too large"));
        }
        marker(&PathBuf::from(bound), &binding)?;
        directory(&root.join("tmp"))?;
        directory(&root.join("objects"))?;
        // Prove ALL committed state and disk entries before deleting any leftovers.
        let metadata = metadata_rows(store)?;
        let disk = inventory(&root)?;
        for (reference, m) in &metadata {
            if !disk.objects.contains(reference) {
                return Err(error("committed artifact missing; cleanup refused"));
            }
            verify_pair(
                &root.join("objects").join(reference),
                m,
                disk.paired.contains(reference),
            )?;
        }
        for temp in disk.temps {
            fs::remove_file(temp).map_err(ioe)?;
        }
        for reference in disk.objects {
            if !metadata.contains_key(&reference) {
                fs::remove_file(root.join("objects").join(reference)).map_err(ioe)?;
            }
        }
        sync_dir(&root.join("tmp"))?;
        sync_dir(&root.join("objects"))?;
        let bytes = metadata.values().try_fold(0u64, |n, m| {
            n.checked_add(m.byte_length)
                .ok_or_else(|| error("quota overflow"))
        })?;
        if bytes > limits.max_managed_bytes || metadata.len() as u64 > limits.max_managed_files {
            return Err(error("committed artifacts exceed configured quota"));
        }
        let shared = Arc::new(Shared {
            root,
            db: store.path().to_owned(),
            store_id,
            limits,
            budget: Mutex::new(Budget {
                bytes,
                files: metadata.len() as u64,
                ..Budget::default()
            }),
            _lease: store.artifact_lease(),
            #[cfg(test)]
            fault: Mutex::new(None),
        });
        store.artifact_state = Some(Arc::clone(&shared));
        Ok(Self { shared })
    }

    fn check_store(&self, store: &SessionStore) -> HarnessResult<()> {
        if store.path() != self.shared.db {
            return Err(error("wrong artifact database"));
        }
        let id: String =
            store
                .conn
                .query_row("SELECT store_id FROM runtime_metadata", [], |r| r.get(0))?;
        if id != self.shared.store_id {
            return Err(error("artifact store identity mismatch"));
        }
        Ok(())
    }

    /// Reserve a bounded writer for a Started invocation. No store borrow escapes.
    /// requested_max may tighten, never widen, the bound invocation's collection limit.
    pub fn begin(
        &self,
        store: &SessionStore,
        session: &str,
        call: &str,
        requested_max: u64,
    ) -> HarnessResult<ArtifactWriter> {
        self.check_store(store)?;
        let (state,epoch,json):(String,String,String)=store.conn.query_row("SELECT state,daemon_epoch,binding_json FROM tool_intents WHERE session_id=?1 AND call_id=?2",params![session,call],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?)))?;
        if state != "Started" || epoch != store.daemon_epoch() {
            return Err(error("artifact writer requires current Started invocation"));
        }
        let binding: serde_json::Value = serde_json::from_str(&json)?;
        let limit = binding["effective_max_output_bytes"]
            .as_u64()
            .ok_or_else(|| error("missing bound output ceiling"))?;
        let max_bytes = requested_max
            .min(limit)
            .min(self.shared.limits.max_artifact_bytes);
        // hard-link publication briefly uses two names: reserve both entries and
        // both logical byte extents up front. Completed objects count once.
        let reserved_bytes = max_bytes
            .checked_mul(2)
            .ok_or_else(|| error("reservation overflow"))?;
        {
            let mut budget = self.shared.budget()?;
            if budget.writers >= self.shared.limits.max_writers
                || reserved_bytes
                    > self
                        .shared
                        .limits
                        .max_managed_bytes
                        .saturating_sub(budget.bytes)
                || 2 > self
                    .shared
                    .limits
                    .max_managed_files
                    .saturating_sub(budget.files)
            {
                return Err(error("artifact admission quota exhausted"));
            }
            budget.bytes += reserved_bytes;
            budget.files += 2;
            budget.writers += 1;
        }
        let allocated: HarnessResult<(String, PathBuf, File)> = (|| {
            let reference = format!("art1_{}", random_hex()?);
            let temp = self
                .shared
                .root
                .join("tmp")
                .join(format!("t_{}", random_hex()?));
            let file = private_file(&temp)?;
            Ok((reference, temp, file))
        })();
        let (reference, temp, file) = match allocated {
            Ok(v) => v,
            Err(e) => {
                if let Ok(mut b) = self.shared.budget.lock() {
                    b.bytes -= reserved_bytes;
                    b.files -= 2;
                    b.writers -= 1;
                }
                return Err(e);
            }
        };
        Ok(ArtifactWriter {
            shared: Arc::clone(&self.shared),
            session: session.into(),
            call: call.into(),
            epoch,
            max_bytes,
            reserved_bytes,
            reference,
            temp,
            file: Some(file),
            length: 0,
            hash: Sha256::new(),
            preview: Vec::with_capacity(MAX_PREVIEW_BYTES),
            failed: false,
            published: false,
            settled: false,
        })
    }

    /// Check ownership in a short database read. Release the store lock, then call
    /// verify on the returned value before reading any bytes.
    pub fn lookup(
        &self,
        store: &SessionStore,
        session: &str,
        call: &str,
        reference: &str,
    ) -> HarnessResult<UnverifiedArtifact> {
        self.check_store(store)?;
        {
            let _budget = self.shared.budget()?;
        }
        if !valid_ref(reference) {
            return Err(error("invalid artifact reference"));
        }
        let row:Option<String>=store.conn.query_row("SELECT metadata_json FROM tool_artifacts WHERE artifact_ref=?1 AND session_id=?2 AND call_id=?3",params![reference,session,call],|r|r.get(0)).optional()?;
        let json = row.ok_or_else(|| error("unknown or unauthorized artifact"))?;
        if json.len() > 1_048_576 {
            return Err(error("metadata too large"));
        }
        let metadata: ArtifactMetadata = serde_json::from_str(&json)?;
        if metadata.artifact_ref != reference
            || metadata.session_id != session
            || metadata.call_id != call
        {
            return Err(error("artifact ownership mismatch"));
        }
        Ok(UnverifiedArtifact {
            shared: Arc::clone(&self.shared),
            metadata,
        })
    }
}

impl ArtifactWriter {
    /// Stream a reader with the frozen 64 KiB buffer and at most one byte read
    /// beyond the remaining accepted limit to detect overflow. No truncation publish.
    pub fn copy_from(&mut self, reader: &mut impl Read) -> HarnessResult<u64> {
        let mut buffer = [0u8; COPY_BUFFER_BYTES];
        loop {
            let remaining = self.max_bytes.saturating_sub(self.length);
            let count = buffer.len().min(remaining.saturating_add(1) as usize);
            let n = match reader.read(&mut buffer[..count]) {
                Ok(n) => n,
                Err(e) => {
                    self.failed = true;
                    return Err(ioe(e));
                }
            };
            if n == 0 {
                return Ok(self.length);
            }
            self.write_all(&buffer[..n]).map_err(ioe)?;
        }
    }

    /// Finalize and synchronize publication WITHOUT a SessionStore argument.
    /// Release all database/session locks before this method. A returned pending
    /// object still has no publicly usable reference until its commit succeeds.
    pub fn finish(mut self) -> HarnessResult<PendingArtifact> {
        let result = self.publish();
        match result {
            Ok(metadata) => Ok(PendingArtifact {
                writer: self,
                metadata,
            }),
            Err(e) => {
                if self.published {
                    self.shared.poison();
                }
                Err(e)
            }
        }
    }

    fn publish(&mut self) -> HarnessResult<ArtifactMetadata> {
        if self.failed {
            return Err(error("failed artifact stream cannot publish"));
        }
        {
            let _budget = self.shared.budget()?;
        }
        #[cfg(test)]
        self.shared.checkpoint("before_sync")?;
        self.file
            .as_ref()
            .ok_or_else(|| error("writer file missing"))?
            .sync_all()
            .map_err(ioe)?;
        #[cfg(test)]
        self.shared.checkpoint("after_sync")?;
        let target = self.shared.root.join("objects").join(&self.reference);
        fs::hard_link(&self.temp, &target).map_err(ioe)?;
        self.published = true;
        #[cfg(test)]
        self.shared.checkpoint("after_publish")?;
        sync_dir(&self.shared.root.join("objects"))?;
        fs::remove_file(&self.temp).map_err(ioe)?;
        sync_dir(&self.shared.root.join("tmp"))?;
        #[cfg(test)]
        self.shared.checkpoint("after_directory_sync")?;
        let (preview_bytes, preview_encoding) = preview(&self.preview);
        let metadata = ArtifactMetadata {
            artifact_ref: self.reference.clone(),
            session_id: self.session.clone(),
            call_id: self.call.clone(),
            byte_length: self.length,
            sha256: format!("{:x}", self.hash.clone().finalize()),
            preview_bytes,
            preview_encoding,
            created_at_unix_ms: timestamp()?,
            contract_version: CONTRACT_VERSION.into(),
        };
        if serde_json::to_vec(&metadata)?.len() > 1_048_576 {
            return Err(error("metadata exceeds bounded envelope"));
        }
        Ok(metadata)
    }
}
impl PendingArtifact {
    /// Atomically register metadata plus artifact.created. No filesystem streaming,
    /// hashing or synchronization occurs while this method borrows SessionStore.
    pub fn commit(mut self, store: &mut SessionStore) -> HarnessResult<ArtifactMetadata> {
        let facade = ArtifactStore {
            shared: Arc::clone(&self.writer.shared),
        };
        facade.check_store(store)?;
        {
            let _budget = self.writer.shared.budget()?;
        }
        #[cfg(test)]
        self.writer.shared.checkpoint("before_metadata_commit")?;
        let json = serde_json::to_string(&self.metadata)?;
        crate::store::commit_artifact(store, &self.metadata, &json, &self.writer.epoch)?;
        #[cfg(test)]
        self.writer.shared.checkpoint("after_metadata_commit")?;
        {
            let mut budget = self.writer.shared.budget()?;
            budget.bytes = budget.bytes - self.writer.reserved_bytes + self.writer.length;
            budget.files -= 1;
            budget.writers -= 1;
        }
        self.writer.settled = true;
        Ok(self.metadata)
    }
}

impl Write for ArtifactWriter {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.failed {
            return Err(io::Error::other("artifact stream already failed"));
        }
        if bytes.len() as u64 > self.max_bytes.saturating_sub(self.length) {
            self.failed = true;
            return Err(io::Error::other("artifact payload ceiling exceeded"));
        }
        let result = self
            .file
            .as_mut()
            .ok_or_else(|| io::Error::other("artifact file closed"))?
            .write(bytes);
        match result {
            Ok(n) => {
                self.hash.update(&bytes[..n]);
                self.length += n as u64;
                let count = (MAX_PREVIEW_BYTES - self.preview.len()).min(n);
                self.preview.extend_from_slice(&bytes[..count]);
                Ok(n)
            }
            Err(e) => {
                self.failed = true;
                Err(e)
            }
        }
    }
    fn flush(&mut self) -> io::Result<()> {
        let result = self
            .file
            .as_mut()
            .ok_or_else(|| io::Error::other("artifact file closed"))?
            .flush();
        if result.is_err() {
            self.failed = true;
        }
        result
    }
}
impl Drop for ArtifactWriter {
    fn drop(&mut self) {
        self.file.take();
        if self.settled {
            return;
        }
        if self.published {
            // Commit may have succeeded: NEVER blindly delete the published file.
            // Retain conservative reservations and block admission until reopen.
            self.shared.poison();
            return;
        }
        let clean = match fs::remove_file(&self.temp) {
            Ok(()) => sync_dir(&self.shared.root.join("tmp")).is_ok(),
            Err(e) if e.kind() == io::ErrorKind::NotFound => true,
            Err(_) => false,
        };
        if let Ok(mut budget) = self.shared.budget.lock() {
            if clean {
                budget.bytes -= self.reserved_bytes;
                budget.files -= 2;
                budget.writers -= 1;
            } else {
                budget.poisoned = true;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture_at(dir: &Path) -> (SessionStore, ArtifactStore) {
        let mut store = SessionStore::open(dir.join("events.db")).unwrap();
        store.create_session("s", "fixture").unwrap();
        // Test-only Started projection; public integration tests use real policy.
        store.conn.execute("INSERT INTO tool_intents(session_id,call_id,daemon_epoch,state,binding_json,created_ts) VALUES('s','c',?1,'Started',?2,0)",params![store.daemon_epoch(),serde_json::json!({"effective_max_output_bytes":MAX_ARTIFACT_BYTES}).to_string()]).unwrap();
        let artifacts =
            ArtifactStore::open(&mut store, dir.join("artifacts"), ArtifactLimits::default())
                .unwrap();
        (store, artifacts)
    }
    #[test]
    fn base64_vectors_and_preview_source_ceiling() {
        for (input, expected) in [
            (b"".as_slice(), ""),
            (b"f", "Zg=="),
            (b"fo", "Zm8="),
            (b"foo", "Zm9v"),
            (b"foob", "Zm9vYg=="),
        ] {
            assert_eq!(encode64(input), expected);
        }
        assert_eq!(encode64(&[0xff; 4096]).len(), 5464);
        assert_eq!(preview(&[0xe2]), (String::new(), "utf8".into()));
    }
    #[test]
    fn collision_never_overwrites_a_published_object() {
        let dir = tempfile::tempdir().unwrap();
        let (mut store, a) = fixture_at(dir.path());
        let mut first = a.begin(&store, "s", "c", 3).unwrap();
        first.write_all(b"old").unwrap();
        let m = first.finish().unwrap().commit(&mut store).unwrap();
        let mut second = a.begin(&store, "s", "c", 3).unwrap();
        second.reference = m.artifact_ref.clone();
        second.write_all(b"new").unwrap();
        assert!(second.finish().is_err());
        let mut verified = a
            .lookup(&store, "s", "c", &m.artifact_ref)
            .unwrap()
            .verify()
            .unwrap();
        let mut bytes = Vec::new();
        verified.read_to_end(&mut bytes).unwrap();
        assert_eq!(bytes, b"old");
    }
    #[test]
    fn all_injected_publication_boundaries_reconcile_without_false_references() {
        for stage in [
            "before_sync",
            "after_sync",
            "after_publish",
            "after_directory_sync",
            "before_metadata_commit",
            "after_metadata_commit",
        ] {
            let dir = tempfile::tempdir().unwrap();
            let (mut store, a) = fixture_at(dir.path());
            *a.shared.fault.lock().unwrap() = Some(stage);
            let mut writer = a.begin(&store, "s", "c", 4).unwrap();
            writer.write_all(b"data").unwrap();
            let result = match writer.finish() {
                Ok(p) => p.commit(&mut store),
                Err(e) => Err(e),
            };
            assert!(result.is_err(), "{stage}");
            let count: i64 = store
                .conn
                .query_row("SELECT COUNT(*) FROM tool_artifacts", [], |r| r.get(0))
                .unwrap();
            assert_eq!(count, i64::from(stage == "after_metadata_commit"));
            drop(a);
            drop(store);
            let mut store = SessionStore::open(dir.path().join("events.db")).unwrap();
            let _a = ArtifactStore::open(
                &mut store,
                dir.path().join("artifacts"),
                ArtifactLimits::default(),
            )
            .unwrap();
            assert_eq!(
                fs::read_dir(dir.path().join("artifacts/objects"))
                    .unwrap()
                    .count(),
                count as usize
            );
            assert_eq!(
                fs::read_dir(dir.path().join("artifacts/tmp"))
                    .unwrap()
                    .count(),
                0
            );
        }
    }
    #[test]
    fn crash_stage_probe() {
        let Some(dir) = std::env::var_os("ALLOY_ARTIFACT_CRASH_DIR") else {
            return;
        };
        let (mut store, a) = fixture_at(Path::new(&dir));
        let mut writer = a.begin(&store, "s", "c", 4).unwrap();
        writer.write_all(b"data").unwrap();
        let _ = writer.finish().unwrap().commit(&mut store);
        panic!("child failed to exit at configured boundary");
    }
    #[test]
    fn real_process_exit_at_each_publication_boundary_recovers_safely() {
        for stage in [
            "before_sync",
            "after_sync",
            "after_publish",
            "after_directory_sync",
            "before_metadata_commit",
            "after_metadata_commit",
        ] {
            let dir = tempfile::tempdir().unwrap();
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "artifact::tests::crash_stage_probe",
                    "--nocapture",
                ])
                .env("ALLOY_ARTIFACT_CRASH_DIR", dir.path())
                .env("ALLOY_ARTIFACT_CRASH_STAGE", stage)
                .output()
                .unwrap();
            assert_eq!(
                status.status.code(),
                Some(77),
                "{stage}: {}",
                String::from_utf8_lossy(&status.stderr)
            );
            let mut store = SessionStore::open(dir.path().join("events.db")).unwrap();
            let a = ArtifactStore::open(
                &mut store,
                dir.path().join("artifacts"),
                ArtifactLimits::default(),
            )
            .unwrap();
            let rows = metadata_rows(&store).unwrap();
            assert_eq!(rows.len(), usize::from(stage == "after_metadata_commit"));
            for reference in rows.keys() {
                let mut v = a
                    .lookup(&store, "s", "c", reference)
                    .unwrap()
                    .verify()
                    .unwrap();
                let mut bytes = Vec::new();
                v.read_to_end(&mut bytes).unwrap();
                assert_eq!(bytes, b"data");
            }
            assert_eq!(
                fs::read_dir(dir.path().join("artifacts/tmp"))
                    .unwrap()
                    .count(),
                0
            );
            assert_eq!(
                fs::read_dir(dir.path().join("artifacts/objects"))
                    .unwrap()
                    .count(),
                rows.len()
            );
        }
    }
    #[test]
    fn concurrent_streamers_share_eight_writer_reservations() {
        let dir = tempfile::tempdir().unwrap();
        let (store, a) = fixture_at(dir.path());
        let mut writers = Vec::new();
        for _ in 0..8 {
            writers.push(a.begin(&store, "s", "c", 100).unwrap());
        }
        assert!(a.begin(&store, "s", "c", 1).is_err());
        let gate = Arc::new(std::sync::Barrier::new(9));
        let threads: Vec<_> = writers
            .into_iter()
            .map(|mut writer| {
                let gate = Arc::clone(&gate);
                std::thread::spawn(move || {
                    gate.wait();
                    writer.write_all(&[1u8; 100]).unwrap();
                    writer
                })
            })
            .collect();
        gate.wait();
        for thread in threads {
            drop(thread.join().unwrap());
        }
        let budget = a.shared.budget().unwrap();
        assert_eq!((budget.bytes, budget.files, budget.writers), (0, 0, 0));
    }
    #[test]
    fn dropping_pending_publication_requires_reconciliation() {
        let dir = tempfile::tempdir().unwrap();
        let (store, a) = fixture_at(dir.path());
        let mut writer = a.begin(&store, "s", "c", 4).unwrap();
        writer.write_all(b"data").unwrap();
        drop(writer.finish().unwrap());
        assert!(a.begin(&store, "s", "c", 1).is_err());
        let count: i64 = store
            .conn
            .query_row("SELECT COUNT(*) FROM tool_artifacts", [], |r| r.get(0))
            .unwrap();
        assert_eq!(count, 0);
    }
    #[test]
    fn copy_buffer_is_bounded_and_tighter_invocation_limit_is_enforced() {
        struct Observed {
            remaining: usize,
            max: usize,
        }
        impl Read for Observed {
            fn read(&mut self, b: &mut [u8]) -> io::Result<usize> {
                self.max = self.max.max(b.len());
                let n = b.len().min(self.remaining);
                b[..n].fill(1);
                self.remaining -= n;
                Ok(n)
            }
        }
        let dir = tempfile::tempdir().unwrap();
        let (store, a) = fixture_at(dir.path());
        let mut observed = Observed {
            remaining: 100_001,
            max: 0,
        };
        let mut writer = a.begin(&store, "s", "c", 100_000).unwrap();
        assert!(writer.copy_from(&mut observed).is_err());
        assert!(observed.max <= COPY_BUFFER_BYTES);
        assert!(writer.finish().is_err());
    }
    #[test]
    fn configuration_cannot_raise_frozen_limits() {
        let defaults = ArtifactLimits::default();
        for changed in [
            ArtifactLimits {
                max_artifact_bytes: MAX_ARTIFACT_BYTES + 1,
                ..defaults
            },
            ArtifactLimits {
                max_writers: 9,
                ..defaults
            },
            ArtifactLimits {
                max_managed_files: 4097,
                ..defaults
            },
            ArtifactLimits {
                max_managed_bytes: 1024 * 1024 * 1024 + 1,
                ..defaults
            },
        ] {
            assert!(changed.validate().is_err());
        }
    }
}
