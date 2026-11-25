// Worker-local Gemma 4 text model. Its cache owns one K/V pair per independent layer.
use inference_infra::decoder::gemma4::model::{Gemma4Cache as NativeGemma4Cache, Gemma4Model};

impl Qwen3Backend {
    fn load_gemma4(
        &self,
        request: ModelLoadRequest<'_>,
        data: &ArtifactData,
    ) -> Result<Box<dyn LoadedModel>, EngineError> {
        let control = request.control();
        control.check_cancelled()?;
        let peak = loading_peak_bound(data, &self.limits)?;
        for (operation, required_bytes, available_bytes) in [
            (
                "Gemma 4 checkpoint",
                data.source_bytes,
                request.limits().max_source_bytes(),
            ),
            (
                "Gemma 4 resident weights",
                data.resident_bound,
                request.limits().max_resident_bytes(),
            ),
            (
                "Gemma 4 load overlap",
                peak,
                request.limits().max_load_peak_bytes(),
            ),
        ] {
            if required_bytes > available_bytes {
                return Err(EngineError::MemoryLimitExceeded {
                    operation,
                    required_bytes,
                    available_bytes,
                    reason: MemoryLimitReason::BufferLimit,
                });
            }
        }
        let device = Device::new(
            match self.execution {
                Qwen3Execution::MlxCpu => DeviceType::Cpu,
                Qwen3Execution::MlxMetal => DeviceType::Gpu,
            },
            0,
        )
        .map_err(|e| execution_error("create Gemma4 device", e.to_string()))?;
        if !device
            .is_available()
            .map_err(|e| execution_error("query Gemma4 device", e.to_string()))?
        {
            return Err(EngineError::unsupported(
                "Gemma4 execution device",
                "requested device is unavailable; no fallback",
            ));
        }
        let stream = Rc::new(
            Stream::new(&device)
                .map_err(|e| execution_error("create Gemma4 stream", e.to_string()))?,
        );
        let cancelled = || control.is_cancelled();
        let model = Rc::new(
            Gemma4Model::load_verified(
                &data.specification.directory,
                LoadOptions {
                    max_file_bytes: as_usize(data.source_bytes)?,
                    max_weight_bytes: as_usize(data.resident_bound)?,
                },
                &CheckpointVerification {
                    sha256: &data.checksums,
                    cancelled: &cancelled,
                },
                &stream,
            )
            .map_err(|e| mlx_model_error(e, control, "load Gemma 4 verified model"))?,
        );
        control.check_cancelled()?;
        let memory = BufferMemoryUsage::new(model.weight_buffer_bytes(), data.decoded_bound)?;
        let mut capabilities = self.registration.capabilities().clone();
        capabilities.context_limit_tokens = capabilities
            .context_limit_tokens
            .min(model.config().context());
        Ok(Box::new(Gemma4Loaded {
            model,
            stream,
            artifact: request.artifact().id().clone(),
            instance: ModelInstanceId::new(),
            registration: self.registration.clone(),
            capabilities,
            memory,
            max_prefill_tokens: self.limits.prefill_chunk_tokens.get(),
        }))
    }
}

struct Gemma4Loaded {
    model: Rc<Gemma4Model>,
    stream: Rc<Stream>,
    artifact: ArtifactId,
    instance: ModelInstanceId,
    registration: FamilyRegistration,
    capabilities: Capabilities,
    memory: BufferMemoryUsage,
    max_prefill_tokens: usize,
}
impl LoadedModel for Gemma4Loaded {
    fn instance_id(&self) -> &ModelInstanceId {
        &self.instance
    }
    fn artifact_id(&self) -> &ArtifactId {
        &self.artifact
    }
    fn registration(&self) -> &FamilyRegistration {
        &self.registration
    }
    fn capabilities(&self) -> &Capabilities {
        &self.capabilities
    }
    fn memory_usage(&self) -> Result<BufferMemoryUsage, EngineError> {
        Ok(self.memory)
    }
    fn autoregressive(&mut self) -> Option<&mut dyn AutoregressiveModel> {
        Some(self)
    }
}
impl Drop for Gemma4Loaded {
    fn drop(&mut self) {
        let _ = self.stream.synchronize();
    }
}

struct Gemma4Cache {
    inner: NativeGemma4Cache,
    model: Rc<Gemma4Model>,
    stream: Rc<Stream>,
    owner: ModelInstanceId,
    mode: CacheMode,
}
impl FamilyCache for Gemma4Cache {
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
        self.inner.retained_tokens()
    }
    fn capacity_tokens(&self) -> usize {
        self.inner.capacity()
    }
    fn memory_usage(&self) -> Result<BufferMemoryUsage, EngineError> {
        let used = self
            .model
            .config()
            .cache_payload(self.inner.len())
            .map_err(|e| model_error(e, "count Gemma 4 cache payload"))?;
        BufferMemoryUsage::new(
            self.inner
                .buffer_bytes()
                .map_err(|e| execution_error("measure Gemma4 cache", e.to_string()))?,
            used,
        )
    }
    fn supports_truncation(&self) -> bool {
        false
    }
    fn reset(&mut self) -> Result<(), EngineError> {
        self.inner.reset();
        Ok(())
    }
    fn truncate(&mut self, _: usize) -> Result<(), EngineError> {
        Err(EngineError::unsupported(
            "Gemma4 truncation",
            "evicted local prefixes require reset and rebuild",
        ))
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    // Full-only export omits local state. Restart explicitly rebuilds tokens
    // until a mixed-state persistence format is implemented.
}
impl Drop for Gemma4Cache {
    fn drop(&mut self) {
        let _ = self.stream.synchronize();
    }
}

impl Gemma4Loaded {
    fn forward(
        &self,
        tokens: &[u32],
        cache: &mut dyn FamilyCache,
        control: &ExecutionControl,
    ) -> Result<TokenLogits, EngineError> {
        control.check_cancelled()?;
        if cache.owner() != &self.instance
            || cache.mode() != &family_cache_mode(BackendFamily::Gemma4)?
        {
            return Err(EngineError::invalid_request(
                "Gemma4 cache",
                "foreign model instance or representation",
            ));
        }
        let cache = cache
            .as_any_mut()
            .downcast_mut::<Gemma4Cache>()
            .ok_or_else(|| {
                EngineError::invalid_request("Gemma4 cache", "wrong native cache type")
            })?;
        if !Rc::ptr_eq(&cache.model, &self.model) {
            return Err(EngineError::invalid_request(
                "Gemma4 cache",
                "foreign native model",
            ));
        }
        if tokens.is_empty() || tokens.len() > self.max_prefill_tokens {
            return Err(EngineError::invalid_request(
                "tokens",
                "Gemma4 prefill exceeds its admitted chunk",
            ));
        }
        let end = cache
            .inner
            .len()
            .checked_add(tokens.len())
            .ok_or_else(|| internal("Gemma4 position overflow"))?;
        if end > cache.inner.capacity() {
            return Err(EngineError::ContextLimitExceeded {
                input_tokens: end,
                output_reserve_tokens: 0,
                safety_margin_tokens: 0,
                limit_tokens: cache.inner.capacity(),
            });
        }
        let workspace = self
            .model
            .config()
            .workspace_bytes(tokens.len(), end)
            .map_err(|e| model_error(e, "plan Gemma 4 execution"))?;
        let output = self
            .model
            .step(
                tokens,
                &mut cache.inner,
                workspace,
                &self.stream,
                &mut || control.is_cancelled(),
            )
            .map_err(|e| mlx_model_error(e, control, "execute Gemma 4 decoder"))?;
        TokenLogits::new(output.into_values())
    }
}
impl AutoregressiveModel for Gemma4Loaded {
    fn vocabulary_size(&self) -> usize {
        self.model.config().vocabulary()
    }
    fn allocate_cache(
        &mut self,
        request: &CacheAllocationRequest,
        control: &ExecutionControl,
    ) -> Result<Box<dyn FamilyCache>, EngineError> {
        control.check_cancelled()?;
        if request.mode() != &family_cache_mode(BackendFamily::Gemma4)? {
            return Err(EngineError::unsupported(
                "Gemma4 cache mode",
                "requires mixed full and sliding KV state",
            ));
        }
        if request.capacity_tokens() > self.capabilities.context_limit_tokens {
            return Err(EngineError::ContextLimitExceeded {
                input_tokens: request.capacity_tokens(),
                output_reserve_tokens: 0,
                safety_margin_tokens: 0,
                limit_tokens: self.capabilities.context_limit_tokens,
            });
        }
        let required = self
            .model
            .config()
            .cache_overlap_bound(request.capacity_tokens(), self.max_prefill_tokens)
            .map_err(|e| model_error(e, "plan Gemma 4 cache allocation"))?;
        if required > request.max_cache_bytes() {
            return Err(EngineError::MemoryLimitExceeded {
                operation: "Gemma 4 cache allocation",
                required_bytes: required,
                available_bytes: request.max_cache_bytes(),
                reason: MemoryLimitReason::BufferLimit,
            });
        }
        let inner = self
            .model
            .new_cache(request.capacity_tokens(), request.max_cache_bytes())
            .map_err(|e| mlx_model_error(e, control, "allocate Gemma4 state"))?;
        control.check_cancelled()?;
        Ok(Box::new(Gemma4Cache {
            inner,
            model: Rc::clone(&self.model),
            stream: Rc::clone(&self.stream),
            owner: self.instance.clone(),
            mode: request.mode().clone(),
        }))
    }
    fn prefill(
        &mut self,
        tokens: &[u32],
        cache: &mut dyn FamilyCache,
        control: &ExecutionControl,
    ) -> Result<PrefillOutput, EngineError> {
        PrefillOutput::new(tokens.len(), self.forward(tokens, cache, control)?)
    }
    fn decode(
        &mut self,
        token: u32,
        cache: &mut dyn FamilyCache,
        control: &ExecutionControl,
    ) -> Result<TokenLogits, EngineError> {
        if cache.len() == 0 {
            return Err(EngineError::invalid_request(
                "Gemma4 cache",
                "prefill before decode",
            ));
        }
        self.forward(&[token], cache, control)
    }
}
