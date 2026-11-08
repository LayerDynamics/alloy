// Physical extraction: intentionally shares qwen.rs scope.

/// A file identity obtained through the caller's trusted artifact policy.
/// Do not create these pins from an untrusted manifest and call that verification.
#[derive(Clone, Debug)]
pub struct FilePin {
    /// Exact source-file length.
    pub bytes: u64,
    /// SHA-256 of the complete file contents.
    pub sha256: [u8; 32],
}

/// Explicitly pinned local artifact and template selection.
#[derive(Clone, Debug)]
pub struct Qwen3Artifact {
    /// Identifier that profiles will reference.
    pub id: ArtifactId,
    /// Existing artifact directory; canonicalized during construction.
    pub directory: PathBuf,
    /// Basenames only. Include config.json, tokenizer.json, every consumed
    /// weight shard/index, and any selected EOS configuration file.
    pub files: BTreeMap<String, FilePin>,
    /// Trusted digest of qwen3::template::CHAT_TEMPLATE for this profile path.
    pub template_sha256: [u8; 32],
}

impl Qwen3Artifact {
    /// Fingerprint the trusted manifest record used for render identity.
    ///
    /// Encoding: the NUL-terminated domain below, u64-LE file count, then
    /// each BTreeMap entry in UTF-8 name order as u64-LE name byte length,
    /// UTF-8 name, u64-LE pinned file size, and the 32-byte SHA-256 pin.
    /// Finally append the 32-byte trusted compiled-template digest.
    ///
    /// The artifact ID is framed separately by the shared renderer. Local
    /// directory paths are intentionally excluded. This function performs
    /// no I/O and does not itself verify that files match these pins.
    pub fn manifest_sha256(&self) -> Result<[u8; 32], EngineError> {
        let mut hash = Sha256::new();
        hash.update(b"alloy.qwen3.artifact-manifest.v1\0");
        hash.update(as_u64(self.files.len())?.to_le_bytes());
        for (name, pin) in &self.files {
            hash.update(as_u64(name.len())?.to_le_bytes());
            hash.update(name.as_bytes());
            hash.update(pin.bytes.to_le_bytes());
            hash.update(pin.sha256);
        }
        hash.update(self.template_sha256);
        Ok(hash.finalize().into())
    }
}

struct ArtifactData {
    specification: Qwen3Artifact,
    config: Qwen3Config,
    tokenizer: Arc<PinnedTokenizer>,
    special_tokens: Arc<BTreeSet<u32>>,
    byte_decoder: Arc<BTreeMap<char, u8>>,
    manifest_sha256: [u8; 32],
    eos: Vec<u32>,
    source_bytes: u64,
    resident_bound: u64,
    decoded_bound: u64,
    checksums: BTreeMap<PathBuf, [u8; 32]>,
}

fn component_path(spec: &Qwen3Artifact, name: &str) -> Result<PathBuf, EngineError> {
    let path = Path::new(name);
    if path.components().count() != 1
        || !matches!(path.components().next(), Some(Component::Normal(_)))
    {
        return Err(validation(
            Some(&spec.id),
            "manifest paths must be simple basenames",
        ));
    }
    let supplied = spec.directory.join(path);
    let resolved = supplied
        .canonicalize()
        .map_err(|e| EngineError::checkpoint_io("canonicalize pinned component", &supplied, e))?;
    if resolved.parent() != Some(spec.directory.as_path()) {
        return Err(validation(
            Some(&spec.id),
            "artifact component escapes its directory",
        ));
    }
    Ok(resolved)
}

fn read_metadata(spec: &Qwen3Artifact, name: &str, limit: usize) -> Result<Vec<u8>, EngineError> {
    let pin = spec
        .files
        .get(name)
        .ok_or_else(|| validation(Some(&spec.id), "required metadata lacks a trusted pin"))?;
    if pin.bytes > as_u64(limit)? {
        return Err(EngineError::MemoryLimitExceeded {
            operation: "read Qwen3 metadata",
            required_bytes: pin.bytes,
            available_bytes: as_u64(limit)?,
            reason: MemoryLimitReason::BufferLimit,
        });
    }
    let path = component_path(spec, name)?;
    let mut file = File::open(&path)
        .map_err(|e| EngineError::checkpoint_io("open Qwen3 metadata", &path, e))?;
    let meta = file
        .metadata()
        .map_err(|e| EngineError::checkpoint_io("stat Qwen3 metadata", &path, e))?;
    if !meta.is_file() || meta.len() != pin.bytes {
        return Err(validation(
            Some(&spec.id),
            "metadata length or type disagrees with its pin",
        ));
    }
    let length = as_usize(pin.bytes)?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(length)
        .map_err(|_| allocation("Qwen3 metadata", Some(length)))?;
    bytes.resize(length, 0);
    file.read_exact(&mut bytes)
        .map_err(|e| EngineError::checkpoint_io("read Qwen3 metadata", &path, e))?;
    let mut probe = [0_u8; 1];
    if file
        .read(&mut probe)
        .map_err(|e| EngineError::checkpoint_io("check Qwen3 metadata length", &path, e))?
        != 0
        || digest(&bytes) != pin.sha256
    {
        return Err(validation(
            Some(&spec.id),
            "metadata changed or failed its trusted digest",
        ));
    }
    Ok(bytes)
}

fn parse_json(path: &Path, bytes: &[u8]) -> Result<Value, EngineError> {
    serde_json::from_slice(bytes)
        .map_err(|e| EngineError::checkpoint_format("json", Some(path.to_owned()), e))
}

fn parse_config(value: &Value) -> Result<Qwen3Config, EngineError> {
    if value.get("model_type").and_then(Value::as_str) != Some("qwen3") {
        return Err(EngineError::unsupported(
            "architecture",
            "only dense Qwen3 causal LM is implemented",
        ));
    }
    let architectures = value
        .get("architectures")
        .and_then(Value::as_array)
        .ok_or_else(|| validation(None, "missing architectures"))?;
    if architectures.len() != 1 || architectures[0].as_str() != Some("Qwen3ForCausalLM") {
        return Err(EngineError::unsupported(
            "architecture",
            "expected Qwen3ForCausalLM",
        ));
    }
    if value.get("rope_scaling").is_some_and(|v| !v.is_null()) {
        return Err(EngineError::unsupported(
            "model configuration",
            "rope_scaling is not implemented",
        ));
    }
    for key in ["attention_bias", "mlp_bias", "use_sliding_window"] {
        if value.get(key).is_some_and(|v| v.as_bool() != Some(false)) {
            return Err(EngineError::unsupported(
                "model configuration",
                format!("{key} must be false when present"),
            ));
        }
    }
    if value
        .get("hidden_act")
        .is_some_and(|v| v.as_str() != Some("silu"))
    {
        return Err(EngineError::unsupported(
            "hidden_act",
            "only silu is implemented",
        ));
    }
    if value.get("rms_norm_eps").is_some_and(|v| {
        v.as_f64()
            .is_none_or(|n| !n.is_finite() || (n - 1e-6).abs() > 1e-12)
    }) {
        return Err(EngineError::unsupported(
            "rms_norm_eps",
            "only 1e-6 is implemented",
        ));
    }
    let dimension = |name: &str| -> Result<usize, EngineError> {
        value
            .get(name)
            .and_then(Value::as_u64)
            .and_then(|v| usize::try_from(v).ok())
            .filter(|&v| v > 0)
            .ok_or_else(|| {
                validation(
                    None,
                    format!("missing or invalid positive dimension: {name}"),
                )
            })
    };
    let theta = value
        .get("rope_theta")
        .and_then(Value::as_f64)
        .filter(|v| v.is_finite() && *v > 0.0)
        .ok_or_else(|| validation(None, "missing rope_theta"))? as f32;
    if !theta.is_finite() || theta <= 0.0 {
        return Err(validation(None, "rope_theta is not representable as f32"));
    }
    let config = Qwen3Config {
        hidden_size: dimension("hidden_size")?,
        num_layers: dimension("num_hidden_layers")?,
        q_heads: dimension("num_attention_heads")?,
        kv_heads: dimension("num_key_value_heads")?,
        head_dim: dimension("head_dim")?,
        intermediate_size: dimension("intermediate_size")?,
        vocab_size: dimension("vocab_size")?,
        max_context: dimension("max_position_embeddings")?,
        rope_theta: theta,
        tie_embeddings: value
            .get("tie_word_embeddings")
            .and_then(Value::as_bool)
            .ok_or_else(|| validation(None, "tie_word_embeddings must be explicit"))?,
    };
    if !config.q_heads.is_multiple_of(config.kv_heads)
        || !config.head_dim.is_multiple_of(2)
        || as_u64(config.vocab_size)? > u64::from(u32::MAX) + 1
    {
        return Err(validation(
            None,
            "invalid attention dimensions or token-ID range",
        ));
    }
    for dimension in [
        config.hidden_size,
        config.num_layers,
        config.q_heads,
        config.kv_heads,
        config.head_dim,
        config.intermediate_size,
        config.vocab_size,
        config.max_context,
    ] {
        if i32::try_from(dimension).is_err() {
            return Err(validation(None, "MLX model dimension exceeds C int"));
        }
    }
    let q = product(as_u64(config.q_heads)?, as_u64(config.head_dim)?)?;
    let kv = product(as_u64(config.kv_heads)?, as_u64(config.head_dim)?)?;
    if q > i32::MAX as u64 || kv > i32::MAX as u64 {
        return Err(validation(None, "MLX projection width exceeds C int"));
    }
    resident_bound(&config)?;
    cache_bound(&config, 1)?;
    Ok(config)
}

fn parse_eos(value: Option<&Value>) -> Result<Vec<u32>, EngineError> {
    let value = value.ok_or_else(|| {
        validation(
            None,
            "EOS must be explicit in pinned config or generation_config",
        )
    })?;
    let raw = match value {
        Value::Array(values) => values.clone(),
        Value::Number(_) => vec![value.clone()],
        _ => {
            return Err(validation(
                None,
                "EOS must be an integer or an explicit integer array",
            ));
        }
    };
    let mut eos = Vec::new();
    for value in raw {
        let id = value
            .as_u64()
            .and_then(|v| u32::try_from(v).ok())
            .ok_or_else(|| validation(None, "invalid EOS token ID"))?;
        if !eos.contains(&id) {
            eos.push(id);
        }
    }
    Ok(eos)
}
