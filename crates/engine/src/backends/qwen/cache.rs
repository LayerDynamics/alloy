// Physical extraction: intentionally shares qwen.rs scope.

struct Qwen3NativeCache {
    inner: MlxKvCache,
    owner: ModelInstanceId,
    mlx_owner: MlxModelId,
    mode: CacheMode,
    capacity: usize,
    stream: Rc<Stream>,
}

impl FamilyCache for Qwen3NativeCache {
    fn kv_block_spec(&self) -> Option<KvCacheSpec> {
        Some(self.inner.spec())
    }
    fn export_kv_block(
        &mut self,
        start: usize,
        end: usize,
    ) -> Result<inference_infra::kv_cache::KvBlock, EngineError> {
        self.inner
            .export_block(&self.mlx_owner, start, end, self.stream.as_ref())
            .map_err(|error| cache_error(error, "export MLX Qwen KV block"))
    }
    fn restore_kv_block(
        &mut self,
        block: &inference_infra::kv_cache::KvBlock,
    ) -> Result<usize, EngineError> {
        self.inner
            .restore_block(&self.mlx_owner, block, self.stream.as_ref())
            .map_err(|error| cache_error(error, "restore MLX Qwen KV block"))
    }
    fn owner(&self) -> &ModelInstanceId {
        &self.owner
    }
    fn representation(&self) -> &CacheRepresentationId {
        &self.mode.representation
    }
    fn mode(&self) -> &CacheMode {
        &self.mode
    }
    fn len(&self) -> usize {
        self.inner.len()
    }
    fn retained_tokens(&self) -> usize {
        self.inner.len()
    }
    fn capacity_tokens(&self) -> usize {
        self.capacity
    }
    fn memory_usage(&self) -> Result<BufferMemoryUsage, EngineError> {
        // Measured current backing buffers, not allocator cache or
        // older external views. Used bytes remain logical KV payload.
        let allocated = self
            .inner
            .buffer_bytes()
            .map_err(|e| cache_error(e, "measure committed MLX Qwen KV buffers"))?;
        let spec = self.inner.spec();
        let used = product(
            product(
                product(
                    product(as_u64(spec.layers)?, as_u64(spec.kv_heads)?)?,
                    as_u64(spec.head_dim)?,
                )?,
                as_u64(self.inner.len())?,
            )?,
            8,
        )?;
        BufferMemoryUsage::new(allocated, used)
    }
    fn supports_truncation(&self) -> bool {
        true
    }
    fn reset(&mut self) -> Result<(), EngineError> {
        self.inner
            .reset(&self.mlx_owner)
            .map_err(|e| cache_error(e, "reset MLX Qwen KV"))
    }
    fn truncate(&mut self, tokens: usize) -> Result<(), EngineError> {
        self.inner
            .truncate(&self.mlx_owner, tokens)
            .map_err(|e| cache_error(e, "truncate MLX Qwen KV"))
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl Drop for Qwen3NativeCache {
    fn drop(&mut self) {
        let _ = self.stream.synchronize();
    }
}

fn cache_mode() -> Result<CacheMode, EngineError> {
    Ok(CacheMode {
        representation: CacheRepresentationId::new(QWEN3_MLX_CACHE_REPRESENTATION)?,
        kind: CacheKind::Full,
    })
}

fn cache_spec(config: &Qwen3Config, capacity: usize) -> KvCacheSpec {
    KvCacheSpec {
        layers: config.num_layers,
        batch: 1,
        kv_heads: config.kv_heads,
        capacity_tokens: capacity,
        head_dim: config.head_dim,
    }
}

fn cache_bound(config: &Qwen3Config, capacity: usize) -> Result<u64, EngineError> {
    cache_spec(config, capacity)
        .payload_plan()
        .map(|plan| plan.update_overlap_bytes)
        .map_err(|e| cache_error(e, "plan MLX Qwen KV overlap"))
}
