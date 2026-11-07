// Physical extraction: intentionally shares qwen.rs scope.

fn layer_projection_elements(config: &Qwen3Config) -> Result<u64, EngineError> {
    let h = as_u64(config.hidden_size)?;
    let q = product(as_u64(config.q_heads)?, as_u64(config.head_dim)?)?;
    let kv = product(as_u64(config.kv_heads)?, as_u64(config.head_dim)?)?;
    let i = as_u64(config.intermediate_size)?;
    sum(&[
        product(2, product(q, h)?)?,  // q, o
        product(2, product(kv, h)?)?, // k, v
        product(3, product(i, h)?)?,  // gate, up, down
    ])
}

fn norm_bytes(config: &Qwen3Config) -> Result<u64, EngineError> {
    let h = as_u64(config.hidden_size)?;
    let head = as_u64(config.head_dim)?;
    let per_layer = sum(&[
        product(2, h)?,    // input + post attention
        product(2, head)?, // q/k norm
    ])?;
    let elements = sum(&[
        product(as_u64(config.num_layers)?, per_layer)?,
        h, // final layernorm
    ])?;
    product(elements, 4)
}

/// Host-side Affine4 payload estimate (packed codes + scales/biases for every
/// matrix, including embed/lm_head). Used for loader charge before MLX upload.
fn host_affine4_bound(config: &Qwen3Config) -> Result<u64, EngineError> {
    let h = as_u64(config.hidden_size)?;
    let v = as_u64(config.vocab_size)?;
    let matrix_elements = sum(&[
        product(v, h)?, // embed
        product(
            as_u64(config.num_layers)?,
            layer_projection_elements(config)?,
        )?,
        if config.tie_embeddings {
            0
        } else {
            product(v, h)?
        },
    ])?;
    let matrix_bytes = affine4_matrix_bytes(matrix_elements, AFFINE4_GROUP_SIZE)?;
    let workspace = product(matrix_bytes / 32 + 1, 1)?;
    sum(&[norm_bytes(config)?, matrix_bytes, workspace])
}

/// MLX resident estimate after upload: every Affine4 matrix stays packed.
/// Embedding lookup dequantizes only the requested rows, and the output
/// projection consumes packed weights through quantized matmul.
fn resident_bound(config: &Qwen3Config) -> Result<u64, EngineError> {
    host_affine4_bound(config)
}

/// Largest possible individual source vector copied during native upload.
/// Use dense float32 widths even for a quantized checkpoint: mixed dense
/// matrices can be accepted within the decoded-weight allowance. Each
/// vector is dropped as soon as `mlx_array_new_data` has copied it.
fn largest_upload_buffer_bound(config: &Qwen3Config) -> Result<u64, EngineError> {
    let hidden = as_u64(config.hidden_size)?;
    let query = product(as_u64(config.q_heads)?, as_u64(config.head_dim)?)?;
    let kv = product(as_u64(config.kv_heads)?, as_u64(config.head_dim)?)?;
    let elements = [
        product(as_u64(config.vocab_size)?, hidden)?,
        product(hidden, query)?,
        product(hidden, kv)?,
        product(hidden, as_u64(config.intermediate_size)?)?,
        hidden,
        as_u64(config.head_dim)?,
    ]
    .into_iter()
    .max()
    .ok_or_else(|| internal("missing Qwen3 upload dimensions"))?;
    product(elements, 4)
}

/// Source decoding and native upload are separate ownership phases.
///
/// The streaming loader retains bounded metadata/scratch plus decoded
/// vectors, verifies every consumed byte, and drops its source state before
/// returning. Upload then replaces one host vector at a time with its native
/// array; only that vector's copy can overlap both representations. Native
/// vocabulary matrices remain packed rather than creating dense graphs.
fn loading_peak_bound(
    data: &ArtifactData,
    limits: &Qwen3BackendLimits,
) -> Result<u64, EngineError> {
    let host_limit = sum(&[data.decoded_bound, limits.loader_headroom_bytes])?;
    let workspace = qwen3::loader::streaming_workspace_bound(as_usize(data.source_bytes)?)
        .map_err(|error| model_error(error, "plan streaming Qwen loader"))?;
    let decode_peak = sum(&[host_limit, as_u64(workspace)?])?;
    let upload_peak = sum(&[
        host_limit.max(data.resident_bound),
        largest_upload_buffer_bound(&data.config)?,
    ])?;
    sum(&[decode_peak.max(upload_peak), limits.loader_headroom_bytes])
}

fn affine4_matrix_bytes(elements: u64, group_size: u64) -> Result<u64, EngineError> {
    if group_size == 0 {
        return Err(EngineError::invalid_request(
            "group_size",
            "affine4 group size must be positive",
        ));
    }
    // packed u32: 8 nibbles per word → elements/8 words * 4 bytes = elements/2
    let packed = elements / 2;
    // scales + biases: 2 * (elements / group_size) * 4 bytes
    let params = product(elements / group_size, 8)?;
    sum(&[packed, params])
}

// Dense matrix payload, shared by host decoding and MLX residency.
fn dense_weight_payload_bound(config: &Qwen3Config) -> Result<u64, EngineError> {
    let embedding = product(as_u64(config.vocab_size)?, as_u64(config.hidden_size)?)?;
    let projections = product(
        as_u64(config.num_layers)?,
        layer_projection_elements(config)?,
    )?;
    let elements = sum(&[
        embedding,
        projections,
        if config.tie_embeddings { 0 } else { embedding },
    ])?;
    sum(&[product(elements, 4)?, norm_bytes(config)?])
}

fn dense_decoded_weight_bound(config: &Qwen3Config) -> Result<u64, EngineError> {
    sum(&[
        dense_weight_payload_bound(config)?,
        product(
            as_u64(config.num_layers)?,
            as_u64(std::mem::size_of::<LayerWeights>())?,
        )?,
    ])
}

fn decoded_weight_bound(config: &Qwen3Config) -> Result<u64, EngineError> {
    // Loader charges host Affine4 (or f32) buffers before MLX upload.
    sum(&[
        host_affine4_bound(config)?,
        product(
            as_u64(config.num_layers)?,
            as_u64(std::mem::size_of::<LayerWeights>())?,
        )?,
    ])
}

fn weight_memory_add_f32(
    allocated: &mut u64,
    used: &mut u64,
    values: &Vec<f32>,
) -> Result<(), EngineError> {
    *allocated = sum(&[*allocated, product(as_u64(values.capacity())?, 4)?])?;
    *used = sum(&[*used, product(as_u64(values.len())?, 4)?])?;
    Ok(())
}

fn weight_memory_add_bytes(
    allocated: &mut u64,
    used: &mut u64,
    bytes: u64,
) -> Result<(), EngineError> {
    *allocated = sum(&[*allocated, bytes])?;
    *used = sum(&[*used, bytes])?;
    Ok(())
}

fn weight_memory(weights: &Qwen3Weights) -> Result<BufferMemoryUsage, EngineError> {
    let mut allocated = product(
        as_u64(weights.layers.capacity())?,
        as_u64(std::mem::size_of::<LayerWeights>())?,
    )?;
    let mut used = product(
        as_u64(weights.layers.len())?,
        as_u64(std::mem::size_of::<LayerWeights>())?,
    )?;
    weight_memory_add_bytes(
        &mut allocated,
        &mut used,
        as_u64(weights.embed_tokens.resident_bytes())?,
    )?;
    weight_memory_add_f32(&mut allocated, &mut used, &weights.final_layernorm)?;
    if let Some(head) = &weights.lm_head {
        weight_memory_add_bytes(&mut allocated, &mut used, as_u64(head.resident_bytes())?)?;
    }
    for layer in &weights.layers {
        weight_memory_add_f32(&mut allocated, &mut used, &layer.input_layernorm)?;
        weight_memory_add_bytes(
            &mut allocated,
            &mut used,
            as_u64(layer.q_proj.resident_bytes())?,
        )?;
        weight_memory_add_bytes(
            &mut allocated,
            &mut used,
            as_u64(layer.k_proj.resident_bytes())?,
        )?;
        weight_memory_add_bytes(
            &mut allocated,
            &mut used,
            as_u64(layer.v_proj.resident_bytes())?,
        )?;
        weight_memory_add_f32(&mut allocated, &mut used, &layer.q_norm)?;
        weight_memory_add_f32(&mut allocated, &mut used, &layer.k_norm)?;
        weight_memory_add_bytes(
            &mut allocated,
            &mut used,
            as_u64(layer.o_proj.resident_bytes())?,
        )?;
        weight_memory_add_f32(&mut allocated, &mut used, &layer.post_attention_layernorm)?;
        weight_memory_add_bytes(
            &mut allocated,
            &mut used,
            as_u64(layer.gate_proj.resident_bytes())?,
        )?;
        weight_memory_add_bytes(
            &mut allocated,
            &mut used,
            as_u64(layer.up_proj.resident_bytes())?,
        )?;
        weight_memory_add_bytes(
            &mut allocated,
            &mut used,
            as_u64(layer.down_proj.resident_bytes())?,
        )?;
    }
    BufferMemoryUsage::new(allocated, used)
}

fn logits_payload_bound(config: &Qwen3Config) -> Result<u64, EngineError> {
    // The model's last-row API admits projection, possible contiguous
    // output, and host result. Four rows' bytes leave capacity headroom.
    product(as_u64(config.vocab_size)?, 16)
}

/// Conservative f32 graph-payload allowance for the selected chunk bound.
/// Counts per-layer projection/normalization/MLP intermediates, possible
/// attention score/softmax buffers, and prefix-layout temporaries.
///
/// KV old/new overlap is separately reserved in cache_bound, not here.
/// MLX allocator internals, kernels' hidden workspace, caches and alignment
/// are not a certified hard bound: execution_headroom_bytes must be sized
/// and calibrated for the supported MLX pin/device/model. This adapter does
/// not claim OS OOM prevention or allocator-pressure discovery.
fn workspace_bound(
    config: &Qwen3Config,
    capacity: usize,
    chunk_tokens: usize,
    token_bytes: usize,
) -> Result<u64, EngineError> {
    let layers = as_u64(config.num_layers)?;
    let t = as_u64(chunk_tokens.min(capacity).max(1))?;
    let s = as_u64(capacity)?;
    let h = as_u64(config.hidden_size)?;
    let q = product(as_u64(config.q_heads)?, as_u64(config.head_dim)?)?;
    let kv = product(as_u64(config.kv_heads)?, as_u64(config.head_dim)?)?;
    let i = as_u64(config.intermediate_size)?;
    let per_layer_elements = sum(&[
        product(12, product(t, h)?)?,
        product(12, product(t, q)?)?,
        product(12, product(t, kv)?)?,
        product(10, product(t, i)?)?,
        product(8, product(product(as_u64(config.q_heads)?, t)?, s)?)?,
        product(4, product(s, kv)?)?,
    ])?;
    sum(&[
        product(product(layers, per_layer_elements)?, 4)?,
        product(product(t, h)?, 16)?,
        logits_payload_bound(config)?,
        product(as_u64(token_bytes)?, 8)?,
        product(sum(&[layers, 64])?, 256)?,
        32,
    ])
}

fn output_bytes(limits: GenerationLimits) -> Result<u64, EngineError> {
    sum(&[
        as_u64(limits.max_buffered_text_bytes())?,
        as_u64(limits.max_text_delta_bytes())?,
        product(
            as_u64(limits.event_capacity())?,
            as_u64(std::mem::size_of::<String>())?,
        )?,
    ])
}

fn sum(values: &[u64]) -> Result<u64, EngineError> {
    values.iter().try_fold(0_u64, |a, &b| {
        a.checked_add(b)
            .ok_or_else(|| internal("Qwen3 size arithmetic overflow"))
    })
}

fn product(a: u64, b: u64) -> Result<u64, EngineError> {
    a.checked_mul(b)
        .ok_or_else(|| internal("Qwen3 size arithmetic overflow"))
}

fn as_u64(value: usize) -> Result<u64, EngineError> {
    u64::try_from(value).map_err(|_| internal("Qwen3 size exceeds u64"))
}

fn as_usize(value: u64) -> Result<usize, EngineError> {
    usize::try_from(value).map_err(|_| internal("Qwen3 size exceeds platform usize"))
}
